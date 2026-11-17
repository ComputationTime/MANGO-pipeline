"""Freeze the three local binding CSVs and reuse exact, audited ANARCI assignments."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3

from finalize_openalphaseq_pairing import digest, prepare_training_fasta, write_json
from prepare_openalphaseq_pairing import split_construct


def sid(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def finite_affinity(value):
    if value.strip().lower() in ('', 'nan', 'none', 'na'):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def flag(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Unexpected assay flag: {value!r}')
    return int(value == 'True')


def main(config):
    import yaml
    settings = yaml.safe_load(Path(config).read_text())
    root = Path(settings['experiment_root']); root.mkdir(parents=True, exist_ok=True)
    cache = Path(settings['cache_root'])
    study = json.loads((cache / 'study.json').read_text())
    files = [Path(settings['source_dir']) / f'{name}_data.csv' for name in ('HER2', 'PD1', 'TIGIT')]
    inputs = dict(source_sha256={str(p): digest(p) for p in files},
                  training_records_sha256=digest(study['training_records']),
                  implementation_sha256=digest(__file__))
    marker = root / 'raw_manifest.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['inputs'] != inputs:
            raise ValueError('Frozen local inputs changed; use a new experiment root')
        for name, expected in previous['files_sha256'].items():
            if digest(root / name) != expected:
                raise ValueError(f'Frozen preparation changed: {name}')
        print('Reusing frozen local binding inputs', flush=True)
        return
    models = study['models']
    for name, spec in models.items():
        prefix = 'iglm_scratch' if name == 'iglm' else name
        candidates = [p for p in Path('artifacts/runs').glob(f'{prefix}__*/checkpoints/best.pt')
                      if (p.parent.parent / 'eval.json').exists()]
        selected = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        spec['directory'] = str(selected.parent.parent)
        spec['checkpoint_sha256'] = digest(selected)
        spec['checkpoint_epoch'] = json.loads((selected.parent.parent / 'eval.json').read_text())['checkpoint_epoch']
    study.update(experiment_root=str(root), batch_size=settings['batch_size'],
                 cache_root=str(cache), local_binding_inputs=inputs)
    write_json(root / 'study.json', study)
    constructs, components, antigens, observations, counts = {}, {}, {}, [], {}
    for path in files:
        family = path.name.split('_')[0]; count = 0; blank = 0
        for row in csv.DictReader(path.open()):
            count += 1
            antigen, sequence = row['matalpha_sequence'], row['mata_sequence']
            aid = sid(antigen)
            if not antigen or set(antigen) - set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError('Invalid antigen sequence')
            antigens[aid] = (aid, antigen, family, json.dumps([row['matalpha_description']]))
            if not sequence:
                blank += 1
                continue
            parts, kind = split_construct(sequence)
            ids = [sid(s) for s in parts]
            constructs[sid(sequence)] = (sid(sequence), sequence, json.dumps(ids), kind, json.dumps([family]))
            components.update(zip(ids, parts))
            observations.append((sid(sequence), aid, path.name, finite_affinity(row['alphaseq_affinity']),
                                 flag(row['above_background']), flag(row['sufficient_replicate_observations'])))
        counts[family] = dict(rows=count, blank_antibody_rows=blank)
    if len(antigens) != 3 or len({a[2] for a in antigens.values()}) != 3:
        raise ValueError('Expected exactly one antigen per local CSV')
    with sqlite3.connect(root / 'raw.sqlite') as db:
        db.executescript('''
            CREATE TABLE components (id TEXT PRIMARY KEY,sequence TEXT,frequency INTEGER);
            CREATE TABLE antibodies (id TEXT PRIMARY KEY,sequence TEXT,components TEXT,kind TEXT,families TEXT);
            CREATE TABLE antigens (id TEXT PRIMARY KEY,sequence TEXT,family TEXT,aliases TEXT);
            CREATE TABLE observations (antibody_id TEXT,antigen_id TEXT,experiment TEXT,
                affinity REAL,above_background INTEGER,sufficient_replicates INTEGER);
        ''')
        db.executemany('INSERT INTO components VALUES (?,?,1)', sorted(components.items()))
        db.executemany('INSERT INTO antibodies VALUES (?,?,?,?,?)', sorted(constructs.values()))
        db.executemany('INSERT INTO antigens VALUES (?,?,?,?)', sorted(antigens.values()))
        db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?)', observations)
    fasta = root / 'components.fasta'
    fasta.write_text(''.join(f'>{k}\n{s}\n' for k,s in sorted(components.items())))
    cached_marker = json.loads((cache / 'numbering_complete.json').read_text())
    if not cached_marker['complete'] or cached_marker['source_sha256'] != digest(cache / 'components.fasta'):
        raise ValueError('Cached ANARCI source changed')
    with sqlite3.connect(f'file:{(cache / "domains.sqlite").resolve()}?mode=ro', uri=True) as source:
        schema = source.execute("SELECT sql FROM sqlite_master WHERE name='domains'").fetchone()[0]
        rows = []
        for key, sequence in sorted(components.items()):
            row = source.execute('SELECT * FROM domains WHERE component_id=?', (key,)).fetchone()
            if row is None or row[1] != sequence:
                raise ValueError(f'No exact cached domain assignment for {key}')
            rows.append(row)
    with sqlite3.connect(root / 'domains.sqlite') as db:
        db.execute(schema)
        db.executemany('INSERT INTO domains VALUES (?,?,?,?,?,?,?,?)', rows)
    write_json(root / 'numbering_complete.json', dict(complete=True, total=len(rows),
        source_sha256=digest(fasta), counts=dict(Counter(r[2] for r in rows)),
        reused_from=str(cache), cached_domains_sha256=digest(cache / 'domains.sqlite')))
    training = prepare_training_fasta(study)
    old_training = cache / 'training_numbering'
    numbered = json.loads((old_training / 'numbering_complete.json').read_text())
    if not numbered['complete'] or numbered['source_sha256'] != digest(training / 'components.fasta'):
        raise ValueError('Training ANARCI cache differs from current train split')
    for name in ('domains.sqlite', 'numbering_complete.json'):
        shutil.copyfile(old_training / name, training / name)
    records = list(csv.DictReader(Path(study['training_records']).open()))
    splits = {}
    for row in records:
        splits.setdefault(row['ab_ag_cluster'], set()).add(row['split'])
    if any(len(s) != 1 for s in splits.values()):
        raise ValueError('A cluster crosses training splits')
    audit = dict(complete=True, training_records_sha256=inputs['training_records_sha256'],
                 clusters_crossing_splits=0, splits=dict(Counter(r['split'] for r in records)))
    write_json(root / 'training_audit.json', audit)
    antigen_rows = [dict(id=k, split='test', antigen_chains='A', resolved_ag_seq=s,
                        family=f, aliases=';'.join(json.loads(a))) for k,s,f,a in sorted(antigens.values())]
    for filename in ('antigens.csv', 'antigen_records.csv'):
        rows = [dict(r) for r in antigen_rows]
        if filename == 'antigen_records.csv':
            for row in rows:
                row['pdb_path'] = str(root / 'structures' / row['id'] / 'antigen.cif')
        with (root / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Freeze only artifacts used to prepare this local analysis, never full-release observations.
    names = ['study.json', 'raw.sqlite', 'components.fasta', 'domains.sqlite', 'numbering_complete.json',
             'training_numbering/domains.sqlite', 'training_numbering/numbering_complete.json',
             'antigens.csv', 'antigen_records.csv', 'training_audit.json']
    write_json(marker, dict(inputs=inputs, source_rows=sum(c['rows'] for c in counts.values()),
        targets=counts, raw_constructs=len(constructs), components=len(components),
        files_sha256={name:digest(root / name) for name in names}))
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/config/mango_binding_affinity.yaml')
    main(**vars(parser.parse_args()))
