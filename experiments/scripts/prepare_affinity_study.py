"""Audit independent sequence cohorts before any improvement-study training."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3
import subprocess

from affinity_study_common import (digest, fingerprint, freeze, read_csv, read_json,
                                   row_key, write_csv, write_json)
from prepare_abrank_transfer import UnionFind, assert_disjoint
from prepare_asd_transfer import select_examples


def fasta(path, sequences):
    Path(path).write_text(''.join(f'>{fingerprint(s)}\n{s}\n' for s in sorted(set(sequences)) if s))


def search(settings, root, name, queries, targets, antibody):
    queries, targets = set(queries)-{''}, set(targets)-{''}
    if not queries or not targets:
        return set()
    directory = root / 'searches' / name; directory.mkdir(parents=True, exist_ok=True)
    fasta(directory / 'query.fa', queries); fasta(directory / 'target.fa', targets)
    identity = settings['antibody_identity' if antibody else 'antigen_identity']
    coverage = settings['antibody_coverage' if antibody else 'antigen_coverage']
    command = [settings['mmseqs'], 'easy-search', str(directory / 'query.fa'), str(directory / 'target.fa'),
        str(directory / 'hits.tsv'), str(directory / 'tmp'), '--min-seq-id', str(identity),
        '-c', str(coverage), '--cov-mode', '0', '-e', '0.001' if antibody else str(settings['antigen_evalue']),
        '--threads', '4', '--max-seqs', '100000', '-s', '7.5', '--format-output', 'query,target,fident,qcov,tcov,evalue']
    with (directory / 'search.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {line.split('\t')[0] for line in (directory / 'hits.tsv').read_text().splitlines()}


def length_signature(row):
    from chain_examples import example_chains
    return tuple(len(row[f'resolved_{c}_seq']) if name in example_chains(row) else 0
                 for c,name in [('H','heavy'),('L','light')])


def diversity_match(base, pool):
    """Match chain masks and lengths exactly, preserving training token budget."""
    bins = defaultdict(list)
    for row in pool:
        bins[length_signature(row)].append(row)
    counts = Counter(); chosen = []
    for row in base:
        bucket = bins[length_signature(row)]
        if not bucket:
            return None
        best = min(bucket, key=lambda r:(counts[r['resolved_ag_seq']], row_key(r)))
        bucket.remove(best); chosen.append(best); counts[best['resolved_ag_seq']] += 1
    return chosen


def affinity_audit(settings, pool, root):
    path = settings.get('verified_affinity_records')
    if not path:
        return dict(feasible=False, reason='no_independent_curator_verified_assay_table',
                    aggregate_abrank_not_used='Source and antigen identity alone do not establish assay comparability')
    by_id = {r['id']:r for r in pool}; retained = []
    for record in read_csv(path):
        # The optional table references audited canonical records, never supplies
        # alternative sequences or a different split assignment.
        required = {'id','assay_id','assay_provenance_url','log10_kd_nm','affinity_operator'}
        if not required.issubset(record):
            raise ValueError('Verified affinity table lacks assay provenance or Kd contract')
        if record['id'] not in by_id or record['affinity_operator'] != '=':
            continue
        value = float(record['log10_kd_nm'])
        if not math.isfinite(value) or not record['assay_id'] or not record['assay_provenance_url'].startswith('https://'):
            raise ValueError('Invalid verified affinity measurement')
        retained.append(dict(record, **{k:by_id[record['id']][k] for k in ('split','resolved_ag_seq','ab_ag_cluster')}))
    write_csv(root / 'verified_affinities.csv', retained)
    groups = defaultdict(list)
    for r in retained:
        groups[r['split'],r['assay_id'],r['assay_provenance_url'],r['resolved_ag_seq']].append(r)
    comparisons = []
    for key, members in groups.items():
        members = sorted(members, key=lambda r:(float(r['log10_kd_nm']),r['id']))
        for i,strong in enumerate(members):
            # Bound each strong target to one sufficiently separated weaker
            # target. Avoid quadratic expansion of large affinity libraries.
            weak = next((w for w in members[i+1:] if float(w['log10_kd_nm'])-float(strong['log10_kd_nm'])
                         >= math.log10(settings['minimum_affinity_ratio'])), None)
            if weak:
                comparisons.append(dict(split=key[0], assay_id=key[1], strong=strong['id'], weak=weak['id']))
    write_csv(root / 'ranking_pairs.csv', comparisons)
    counts = Counter(r['split'] for r in comparisons)
    feasible = all(counts[s] >= 10 for s in ('train','val','test'))
    return dict(feasible=feasible, reason='verified_pairs_available' if feasible else 'insufficient_verified_pairs_per_split',
                comparisons_by_split=dict(counts), source_sha256=digest(path))


def prepare(config):
    import yaml
    settings = yaml.safe_load(Path(config).read_text()); root = Path(settings['experiment_root'])
    root.mkdir(parents=True, exist_ok=True)
    if settings.get('smoke'):
        # Real cached ESM2 inputs, tiny cohorts; deliberately no scientific audit
        # claims or external benchmark evaluation in this plumbing fixture.
        rows=read_csv('artifacts/experiments/asd_scaling_2k_4k_v2/n2000/cohort/asd_records.csv')
        tiny=[]
        for split in ('train','val','test'):
            selected=sorted((r for r in rows if r['split']==split),key=row_key)[:4]
            for r in selected:r['example_target_chains']='heavy;light'
            tiny.extend(selected)
        # Honor the production mask encoding rather than inventing a new one.
        tiny=select_examples(tiny)
        for phase in ('pilot','confirmation','large'):
            for variant in ('base','diversity'):write_csv(root/f'{phase}_{variant}.csv',tiny)
        write_csv(root/'pool.csv',tiny)
        variants={phase:dict(records=4,base_antigens=4,diversity_antigens=4,diversity_feasible=False)
                  for phase in ('pilot','confirmation','large')}
        write_json(root/'cohort_audit.json',dict(complete=True,smoke_only=True,training_feasible=True,
            variants=variants,development_groups=dict(val=0,test=0),independent_affinity=dict(feasible=False,reason='smoke_fixture'),
            files_sha256={p.name:digest(p) for p in root.glob('*.csv')}))
        return
    source = Path(settings['source_records']); external = Path(settings['external_root'])
    inputs = [source, Path(settings['source_audit']), Path(settings['native_records']),
              external / 'numbering_complete.json', external / 'domains.sqlite', external / 'antigens.csv']
    if settings.get('verified_affinity_records'):
        inputs.append(Path(settings['verified_affinity_records']))
    implementation = [Path(__file__), Path(__file__).with_name('affinity_study_common.py')]
    manifest = dict(settings=settings, source_sha256={str(p):digest(p) for p in inputs},
                    implementation_sha256={str(p):digest(p) for p in implementation})
    freeze(root / 'input_manifest.json', manifest)
    if (root / 'cohort_audit.json').exists():
        audit = read_json(root / 'cohort_audit.json')
        for name, expected in audit['files_sha256'].items():
            if digest(root / name) != expected:
                raise ValueError('Frozen improvement-study cohort changed')
        return
    original_audit = read_json(settings['source_audit'])
    if original_audit['sha256'].get(str(source)) != digest(source):
        raise ValueError('ASD source differs from its original protection audit')
    source_rows = read_csv(source)
    # Never repurpose an upstream holdout for training; reject source groups
    # crossing splits before any selection or new exclusion.
    memberships = defaultdict(set)
    for row in source_rows:
        memberships[row['ab_ag_cluster']].add(row['split'])
    if any(len(v)>1 for v in memberships.values()):
        raise ValueError('An upstream ab_ag_cluster crosses splits')
    eligible = [r for r in source_rows if len(r['resolved_ag_seq']) <= settings['max_antigen_tokens']
                and r.get('has_buzz') != 'True' and 'alpha' not in r.get('source','').lower()
                and not any(s.startswith('flab_') for s in r.get('source','').lower().split(';'))]
    train = sorted((r for r in eligible if r['split']=='train'), key=row_key)[:settings['candidate_records']]
    heldout = [r for r in eligible if r['split']!='train']
    # Limit heldouts by complete source groups, not individual row splits.
    def bounded(rows, limit=400):
        buckets = defaultdict(list)
        for r in rows: buckets[r['ab_ag_cluster']].append(r)
        result = []
        for group in sorted(buckets, key=fingerprint):
            if len(result) >= limit: break
            result.extend(buckets[group])
        return result
    heldout = bounded([r for r in heldout if r['split']=='val']) + bounded([r for r in heldout if r['split']=='test'])
    candidates = train + heldout
    external_marker = read_json(external / 'numbering_complete.json')
    if not external_marker['complete'] or external_marker['source_sha256'] != digest(external / 'components.fasta'):
        raise ValueError('External sequence annotations lack valid source provenance')
    with sqlite3.connect(f'file:{(external / "domains.sqlite").resolve()}?mode=ro', uri=True) as db:
        blocked_chains = {r[0] for r in db.execute("SELECT target FROM domains WHERE status='valid'")}
    blocked_antigens = {r['resolved_ag_seq'] for r in read_csv(external / 'antigens.csv')}
    chains = {r[f'resolved_{c}_seq'] for r in candidates for c in ('H','L')}
    ag_hits = search(settings,root,'external_antigens',[r['resolved_ag_seq'] for r in candidates],blocked_antigens,False)
    ab_hits = search(settings,root,'external_antibodies',chains,blocked_chains,True)
    rejected = []
    def external_safe(row):
        reasons = []
        if row['resolved_ag_seq'] in blocked_antigens or fingerprint(row['resolved_ag_seq']) in ag_hits:
            reasons.append('external_antigen_homology')
        if any(row[f'resolved_{c}_seq'] in blocked_chains or fingerprint(row[f'resolved_{c}_seq']) in ab_hits for c in ('H','L')):
            reasons.append('external_antibody_lineage')
        for reason in reasons: rejected.append(dict(id=row['id'], reason=reason))
        return not reasons
    candidates = [r for r in candidates if external_safe(r)]
    reserved = [r for r in source_rows if r['split']!='train']
    native_reserved = [r for r in read_csv(settings['native_records']) if r['split']!='train']
    # Screen training against ALL upstream and native reservations, including
    # heldout rows not selected for these bounded development panels.
    protected = reserved + native_reserved
    near = search(settings,root,'reserved_antibody_lineages',
        [r[f'resolved_{c}_seq'] for r in candidates if r['split']=='train' for c in ('H','L')],
        [r[f'resolved_{c}_seq'] for r in protected for c in ('H','L')],True)
    safe = []
    for row in candidates:
        if row['split']=='train' and any(fingerprint(row[f'resolved_{c}_seq']) in near for c in ('H','L')):
            rejected.append(dict(id=row['id'],reason='reserved_antibody_lineage')); continue
        safe.append(row)
    # Development partitions must themselves be lineage-disjoint. Remove whole
    # validation source groups having a near-match to the confirmation panel.
    near_dev = search(settings,root,'development_lineages',
        [r[f'resolved_{c}_seq'] for r in safe if r['split']=='val' for c in ('H','L')],
        [r[f'resolved_{c}_seq'] for r in safe if r['split']=='test' for c in ('H','L')],True)
    excluded_groups = {r['ab_ag_cluster'] for r in safe if r['split']=='val' and
        any(fingerprint(r[f'resolved_{c}_seq']) in near_dev for c in ('H','L'))}
    safe = [r for r in safe if r['ab_ag_cluster'] not in excluded_groups]
    train = select_examples(sorted((r for r in safe if r['split']=='train'), key=row_key))
    holdouts = select_examples([r for r in safe if r['split']!='train'])
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert_disjoint([r for r in train+holdouts if r['split']==a], [r for r in train+holdouts if r['split']==b])
    pool = train + holdouts
    write_csv(root / 'pool.csv', pool)
    variants = {}; capacity = min(settings['pilot_records'],len(train))
    for phase, size in [('pilot',capacity),('confirmation',min(settings['confirmation_records'],len(train))),
                        ('large',min(settings['large_records'],len(train)))]:
        base = train[:size]; diverse = diversity_match(base,train)
        n_ag = lambda rows: len({r['resolved_ag_seq'] for r in rows})
        for name, members in [('base',base),('diversity',diverse or base)]:
            write_csv(root / f'{phase}_{name}.csv', members + holdouts)
        variants[phase] = dict(records=size, base_antigens=n_ag(base), diversity_antigens=n_ag(diverse or base),
            diversity_feasible=diverse is not None and n_ag(diverse)>n_ag(base))
    labels = affinity_audit(settings,pool,root)
    fasta(root / 'sequences.fa',[r[f'resolved_{c}_seq'] for r in pool for c in ('H','L')])
    write_csv(root / 'exclusions.csv', rejected)
    audit = dict(complete=True, training_feasible=len(train)>=settings['minimum_training_records'] and
        all(any(r['split']==s for r in holdouts) for s in ('val','test')),
        source_audit_sha256=digest(settings['source_audit']), variants=variants, independent_affinity=labels,
        development=dict(Counter(r['split'] for r in holdouts)),
        development_groups={s:len({r['ab_ag_cluster'] for r in holdouts if r['split']==s}) for s in ('val','test')},
        excluded=dict(Counter(r['reason'] for r in rejected)), excluded_validation_groups=len(excluded_groups),
        candidate_selection='32k bounded hash sample; no new split assignments',
        lineage_rule='90% identity, 80% coverage of both chains, E<=0.001; conservative sequence proxy',
        antigen_rule='20% identity, 60% coverage of both sequences, E<=0.001; inherited CATH exclusions also retained',
        limitations=['Sequence-based exclusions do not prove remote-homology absence.',
                    'Parent ab_ag_cluster is the biological uncertainty unit; seed runs are not biological replicates.'])
    audit['source_quarantine']='BUZZ, explicit AlphaSeq, and all FLAb-derived source tags are excluded before sequence screening'
    audit['files_sha256'] = {p.name:digest(p) for p in root.glob('*.csv')}
    write_json(root / 'cohort_audit.json',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='experiments/config/affinity_improvement_study.yaml')
    prepare(**vars(p.parse_args()))
