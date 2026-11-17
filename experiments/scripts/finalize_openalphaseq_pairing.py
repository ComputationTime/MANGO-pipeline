"""Freeze domain-normalized, training-disjoint OpenAlphaSeq evaluation queries."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def target_id(chain, sequence):
    return hashlib.sha256(f'{chain}:{sequence}'.encode()).hexdigest()


class TrainingMatcher:
    """Find any complete training domain inside a longer expressed component."""
    def __init__(self, sequences):
        self.edges, self.failure, self.output = [{}], [0], [False]
        for sequence in set(sequences):
            if not sequence:
                continue
            node = 0
            for aa in sequence:
                child = self.edges[node].get(aa)
                if child is None:
                    child = len(self.edges)
                    self.edges[node][aa] = child
                    self.edges.append({}); self.failure.append(0); self.output.append(False)
                node = child
            self.output[node] = True
        pending = deque(self.edges[0].values())
        while pending:
            node = pending.popleft()
            for aa, child in self.edges[node].items():
                pending.append(child)
                fallback = self.failure[node]
                while fallback and aa not in self.edges[fallback]:
                    fallback = self.failure[fallback]
                self.failure[child] = self.edges[fallback].get(aa, 0)
                self.output[child] |= self.output[self.failure[child]]

    def contains(self, sequence):
        node = 0
        for aa in sequence:
            while node and aa not in self.edges[node]:
                node = self.failure[node]
            node = self.edges[node].get(aa, 0)
            if self.output[node]:
                return True
        return False


def binding_label(positives, negatives):
    """Any reliable positive is binding; unmeasured/uncertain is not negative."""
    return 'positive' if positives else ('negative' if negatives else 'uncertain')


def prepare_training_fasta(study):
    root = Path(study['experiment_root']) / 'training_numbering'
    root.mkdir(parents=True, exist_ok=True)
    with Path(study['training_records']).open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == 'train']
    sequences = sorted({r[c] for r in rows for c in ('resolved_H_seq', 'resolved_L_seq') if r[c]})
    contents = ''.join(f'>{hashlib.sha256(s.encode()).hexdigest()}\n{s}\n' for s in sequences)
    destination = root / 'components.fasta'
    if destination.exists() and destination.read_text() != contents:
        raise ValueError('Training numbering input differs; preserve the existing annotation cache')
    destination.write_text(contents)
    write_json(root / 'inputs.json', dict(training_records=study['training_records'],
        training_sha256=digest(study['training_records']), sequences=len(sequences), split='train'))
    return root


def finalize(study):
    root = Path(study['experiment_root'])
    for marker in ('numbering_complete.json', 'training_numbering/numbering_complete.json'):
        if not json.loads((root / marker).read_text())['complete']:
            raise ValueError(f'Incomplete annotation: {marker}')
    with Path(study['training_records']).open() as handle:
        original_training = {r[c] for r in csv.DictReader(handle) if r['split'] == 'train'
                             for c in ('resolved_H_seq', 'resolved_L_seq') if r[c]}
    with sqlite3.connect(root / 'training_numbering/domains.sqlite') as db:
        training_rows = db.execute('SELECT sequence,status,target FROM domains').fetchall()
    if {r[0] for r in training_rows} != original_training:
        raise ValueError('Training annotation does not cover exactly the train split')
    if any(r[1] != 'valid' for r in training_rows):
        raise ValueError('A training sequence could not be domain-normalized')
    training = original_training | {r[2] for r in training_rows}
    matcher = TrainingMatcher(training)
    with sqlite3.connect(root / 'domains.sqlite') as db:
        domains = {r[0]: dict(raw=r[1], status=r[2], chain=r[3], sequence=r[4], cdr3=r[5])
                   for r in db.execute('SELECT component_id,sequence,status,chain,target,cdr3_indices FROM domains')}
    seen_targets = set()
    for cid, domain in domains.items():
        seen = domain['status'] == 'valid' and (domain['sequence'] in training or matcher.contains(domain['raw']))
        if seen:
            seen_targets.add(target_id(domain['chain'], domain['sequence']))
    del matcher
    source = sqlite3.connect(f'file:{(root / "raw.sqlite").resolve()}?mode=ro', uri=True)
    temporary = root / 'cohort.tmp.sqlite'; temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript('''
    CREATE TABLE targets (id TEXT PRIMARY KEY,chain TEXT,sequence TEXT,cdr3_indices TEXT,
                          ntokens INTEGER,ncdr3 INTEGER,seen_training INTEGER);
    CREATE TABLE queries (id TEXT PRIMARY KEY,heavy TEXT,light TEXT,heavy_novel INTEGER,
                          light_novel INTEGER,strict_unseen INTEGER,families TEXT,raw_constructs INTEGER);
    CREATE TABLE raw_map (raw_id TEXT PRIMARY KEY,query_id TEXT);
    CREATE TABLE exclusions (raw_id TEXT PRIMARY KEY,reason TEXT);
    CREATE TABLE labels (query_id TEXT,antigen_id TEXT,label TEXT,positive_observations INTEGER,
                         negative_observations INTEGER,uncertain_observations INTEGER,conflict INTEGER,
                         experiments TEXT,PRIMARY KEY(query_id,antigen_id));
    ''')
    targets = {}
    for d in domains.values():
        if d['status'] != 'valid' or len(d['raw']) > 170 or not 60 <= len(d['sequence']) <= 170:
            continue
        tid = target_id(d['chain'], d['sequence'])
        targets[tid] = (tid, d['chain'], d['sequence'], d['cdr3'], len(d['sequence']) + 1,
                        len(json.loads(d['cdr3'])), int(tid in seen_targets))
    db.executemany('INSERT INTO targets VALUES (?,?,?,?,?,?,?)', targets.values())
    queries, raw_map, exclusions = {}, {}, []
    for rid, components_json, kind, families_json in source.execute('SELECT id,components,kind,families FROM antibodies'):
        components = json.loads(components_json)
        reason = None
        if kind not in ('scfv', 'single_domain'):
            reason = kind
        elif any(domains[c]['status'] != 'valid' for c in components):
            reason = 'unassigned_or_multiple_domains'
        else:
            parts = [domains[c] for c in components]
            types = Counter(d['chain'] for d in parts)
            if (kind == 'scfv' and types != Counter(heavy=1, light=1)) or (kind == 'single_domain' and types != Counter(heavy=1)):
                reason = 'unexpected_domain_types'
            elif any(not 60 <= len(d['sequence']) <= 170 for d in parts):
                reason = 'domain_length'
        if reason:
            exclusions.append((rid, reason)); continue
        typed = {d['chain']: target_id(d['chain'], d['sequence']) for d in parts}
        heavy, light = typed.get('heavy'), typed.get('light')
        hnew = bool(heavy and heavy not in seen_targets); lnew = bool(light and light not in seen_targets)
        if not (hnew or lnew):
            exclusions.append((rid, 'all_domains_seen_in_training')); continue
        qid = hashlib.sha256(json.dumps([heavy, light], separators=(',', ':')).encode()).hexdigest()
        if qid not in queries:
            queries[qid] = dict(id=qid, heavy=heavy, light=light, heavy_novel=hnew, light_novel=lnew,
                strict_unseen=all(t not in seen_targets for t in typed.values()), families=set(), raw_constructs=0)
        queries[qid]['families'].update(json.loads(families_json)); queries[qid]['raw_constructs'] += 1
        raw_map[rid] = qid
    db.executemany('INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)',
        [(q['id'], q['heavy'], q['light'], q['heavy_novel'], q['light_novel'], q['strict_unseen'],
          json.dumps(sorted(q['families'])), q['raw_constructs']) for q in queries.values()])
    db.executemany('INSERT INTO raw_map VALUES (?,?)', raw_map.items())
    db.executemany('INSERT INTO exclusions VALUES (?,?)', exclusions)
    accumulated = {}
    for rid, aid, experiment, affinity, above, sufficient in source.execute('SELECT * FROM observations'):
        if rid not in raw_map:
            continue
        key = (raw_map[rid], aid)
        entry = accumulated.setdefault(key, [0, 0, 0, set()])
        # A finite affinity alone does not make a binder; retain the assay flags.
        index = (0 if above else 1) if affinity is not None and sufficient else 2
        entry[index] += 1; entry[3].add(experiment)
    db.executemany('INSERT INTO labels VALUES (?,?,?,?,?,?,?,?)',
        [(qid, aid, binding_label(p, n), p, n, u, bool(p and n), json.dumps(sorted(experiments)))
         for (qid, aid), (p, n, u, experiments) in accumulated.items()])
    db.executescript('CREATE INDEX labels_query ON labels(query_id);')
    db.commit()
    audit = dict(complete=True, training_original_sequences=len(original_training),
        training_domain_sequences=len({r[2] for r in training_rows}),
        training_records_sha256=digest(study['training_records']),
        raw_constructs=source.execute('SELECT COUNT(*) FROM antibodies').fetchone()[0],
        retained_raw_constructs=len(raw_map), unique_queries=len(queries),
        heavy_novel_queries=sum(q['heavy_novel'] for q in queries.values()),
        strict_unseen_queries=sum(q['strict_unseen'] for q in queries.values()),
        exclusions=dict(Counter(reason for _, reason in exclusions)),
        labels=dict(db.execute('SELECT label,COUNT(*) FROM labels GROUP BY label')),
        conflicting_pairs=db.execute('SELECT COUNT(*) FROM labels WHERE conflict=1').fetchone()[0],
        families={family:dict(queries=sum(family in q['families'] for q in queries.values()),
            heavy_novel=sum(q['heavy_novel'] and family in q['families'] for q in queries.values()),
            strict_unseen=sum(q['strict_unseen'] and family in q['families'] for q in queries.values()))
            for family in ('RBD', 'TIGIT', 'PD1', 'HER2')},
        novelty_protocol='exact_IMGT_domains_and_complete_training_domains_as_expression_substrings_v1',
        lineage_disjoint=False,
        canonical_training_overlap_in_scored_domains=0,
        implementation_sha256=digest(__file__), raw_manifest_sha256=digest(root / 'raw_manifest.json'),
        numbering_source_sha256=json.loads((root / 'numbering_complete.json').read_text())['source_sha256'])
    novel_targets = {t for q in queries.values() for t, novel in
                     ((q['heavy'], q['heavy_novel']), (q['light'], q['light_novel'])) if novel}
    assert not any(targets[t][2] in training for t in novel_targets)
    assert not (novel_targets & seen_targets)
    audit['unique_novel_targets'] = len(novel_targets)
    exports = {
        'heldout_antibodies.csv': '''SELECT q.*,h.sequence AS heavy_sequence,l.sequence AS light_sequence
            FROM queries q LEFT JOIN targets h ON q.heavy=h.id LEFT JOIN targets l ON q.light=l.id ORDER BY q.id''',
        'excluded.csv': 'SELECT * FROM exclusions ORDER BY raw_id',
        'labels.csv': 'SELECT * FROM labels ORDER BY query_id,antigen_id',
    }
    for filename, statement in exports.items():
        cursor = db.execute(statement)
        with (root / filename).open('w', newline='') as handle:
            writer = csv.writer(handle); writer.writerow([col[0] for col in cursor.description]); writer.writerows(cursor)
    with (root / 'heldout_chains.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['id', 'chain', 'sequence', 'cdr3_indices', 'ntokens', 'ncdr3'])
        writer.writerows(targets[t][:6] for t in sorted(novel_targets))
    db.close(); source.close(); temporary.replace(root / 'cohort.sqlite')
    audit['cohort_sha256'] = digest(root / 'cohort.sqlite')
    write_json(root / 'cohort_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/config/openalphaseq_pairing.yaml')
    parser.add_argument('--training-fasta-only', action='store_true')
    args = parser.parse_args()
    if Path(args.config).suffix == '.json':
        study = json.loads(Path(args.config).read_text())
    else:
        import yaml
        study = yaml.safe_load(Path(args.config).read_text())
    if args.training_fasta_only:
        prepare_training_fasta(study)
    else:
        finalize(study)
