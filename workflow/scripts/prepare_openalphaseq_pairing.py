"""Freeze raw OpenAlphaSeq expression constructs and the 12-protein panel."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from prepare_asd_transfer import sha256
from antigen_diagnostics import atomic_json

FAMILIES = dict(YM_0005='RBD', YM_0549='RBD', YM_1068='RBD', YM_0693='TIGIT',
                YM_0988='TIGIT', YM_0852='PD1', YM_0985='PD1', YM_0989='HER2', YM_0990='HER2')
AA = set('ACDEFGHIKLMNPQRSTVWY')


def sid(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def split_construct(sequence):
    """Split the explicitly expressed scFv linker, leaving typing to ANARCI."""
    if not sequence or set(sequence) - AA:
        return [], 'invalid_amino_acids'
    matches = [m for m in re.finditer('GGGGSGGGGSGGGGS', sequence)
               if 70 <= m.start() <= 170 and 70 <= len(sequence) - m.end() <= 170]
    if len(matches) == 1:
        m = matches[0]
        return [sequence[:m.start()], sequence[m.end():]], 'scfv'
    if 70 <= len(sequence) <= 170:
        return [sequence], 'single_domain'
    return [sequence], 'full_sequence_fallback'


def prepare(config):
    import pyarrow.parquet as pq
    if Path(config).suffix == '.json':
        study = json.loads(Path(config).read_text())
    else:
        import yaml
        study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root']); root.mkdir(parents=True, exist_ok=True)
    source = Path(study['dataset_dir'])
    release = json.loads((source / 'download_manifest.json').read_text())
    if not release['complete'] or release['repository'] != 'aalphabio/open-alphaseq':
        raise ValueError('A complete official OpenAlphaSeq snapshot is required')
    files = sorted(source.glob('data/*/data.parquet'))
    for path in files:
        if sha256(path) != release['files'][path.relative_to(source).as_posix()]['sha256']:
            raise ValueError('OpenAlphaSeq source changed')
    audit = json.loads(Path(study['training_audit']).read_text())
    if sha256(study['training_records']) != audit['sha256'][study['training_records']]:
        raise ValueError('16k training cohort changed')
    inputs = dict(configuration=study, repository=release['repository'], revision=release['revision'],
                  source_files={str(p): sha256(p) for p in files},
                  training_sha256=sha256(study['training_records']), preparation_sha256=sha256(__file__))
    manifest_path = root / 'raw_manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous['inputs'] != inputs:
            raise ValueError('Pairing inputs changed; use a new experiment_root')
        if (sha256(root / 'raw.sqlite') != previous['raw_sha256'] or
                sha256(root / 'components.fasta') != previous['components_sha256']):
            raise ValueError('Frozen raw pairing data changed')
        print('Reusing frozen OpenAlphaSeq pairing inputs', flush=True)
        return previous
    panel, aliases, agfamily = set(), defaultdict(set), {}
    panel_mode = study.get('panel', 'optimisation_experiments_all_targets')
    if panel_mode not in {'optimisation_experiments_all_targets', 'all_experiments_all_targets'}:
        raise ValueError(f'Unknown antigen panel: {panel_mode}')
    for path in files:
        if path.parent.name == 'YM_0005' and panel_mode != 'all_experiments_all_targets':
            continue
        t = pq.read_table(path, columns=['matalpha_sequence', 'matalpha_description']).to_pydict()
        for seq, label in zip(t['matalpha_sequence'], t['matalpha_description']):
            if seq:
                if set(seq) - AA:
                    raise ValueError('Noncanonical target protein in candidate panel')
                panel.add(seq); aliases[seq].add(label); agfamily[seq] = FAMILIES[path.parent.name]
    database = root / 'raw.tmp.sqlite'; database.unlink(missing_ok=True)
    db = sqlite3.connect(database)
    db.executescript('''
    PRAGMA journal_mode=OFF;
    CREATE TABLE components (id TEXT PRIMARY KEY, sequence TEXT, frequency INTEGER);
    CREATE TABLE antibodies (id TEXT PRIMARY KEY, sequence TEXT, components TEXT, kind TEXT, families TEXT);
    CREATE TABLE antigens (id TEXT PRIMARY KEY, sequence TEXT, family TEXT, aliases TEXT);
    CREATE TABLE observations (antibody_id TEXT, antigen_id TEXT, experiment TEXT,
                               affinity REAL, above_background INTEGER, sufficient_replicates INTEGER);
    ''')
    db.executemany('INSERT INTO antigens VALUES (?,?,?,?)',
                   [(sid(s), s, agfamily[s], json.dumps(sorted(aliases[s]))) for s in sorted(panel)])
    constructs, components, frequency, parent = {}, {}, Counter(), defaultdict(set)
    counts = {}
    for path in files:
        experiment = path.parent.name
        t = pq.read_table(path).to_pydict()
        for seq in {s for s in t['mata_sequence'] if s}:
            key = sid(seq); parent[key].add(FAMILIES[experiment])
            if key not in constructs:
                parts, kind = split_construct(seq)
                ids = [sid(s) for s in parts]
                constructs[key] = (seq, ids, kind)
                for component, component_id in zip(parts, ids):
                    components[component_id] = component; frequency[component_id] += 1
        observations = []
        n = len(t['mata_sequence'])
        for i in range(n):
            a, b = t['mata_sequence'][i], t['matalpha_sequence'][i]
            if not a or b not in panel:
                continue
            value = t['alphaseq_affinity'][i]
            affinity = float(value) if value is not None and math.isfinite(value) else None
            observations.append((sid(a), sid(b), experiment, affinity,
                                 int(bool(t['above_background'][i])), int(bool(t['sufficient_replicate_observations'][i]))))
        db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?)', observations)
        counts[experiment] = dict(source_rows=n, panel_observations=len(observations))
        db.commit()
        print(experiment, counts[experiment], flush=True)
    db.executemany('INSERT INTO antibodies VALUES (?,?,?,?,?)',
        [(key, seq, json.dumps(ids), kind, json.dumps(sorted(parent[key])))
         for key, (seq, ids, kind) in sorted(constructs.items())])
    db.executemany('INSERT INTO components VALUES (?,?,?)',
        [(key, seq, frequency[key]) for key, seq in sorted(components.items())])
    db.executescript('CREATE INDEX observations_query ON observations(antibody_id,antigen_id);')
    db.commit(); db.close(); database.replace(root / 'raw.sqlite')
    with (root / 'components.fasta').open('w') as handle:
        for key in sorted(components, key=lambda k: (-frequency[k], k)):
            handle.write(f'>{key}\n{components[key]}\n')
    antigens = [dict(id=sid(s), split='test', antigen_chains='AG', resolved_ag_seq=s,
                    family=agfamily[s], aliases=';'.join(sorted(aliases[s]))) for s in sorted(panel)]
    with (root / 'antigens.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(antigens[0])); writer.writeheader(); writer.writerows(antigens)
    result = dict(inputs=inputs, source_rows=sum(c['source_rows'] for c in counts.values()),
                  raw_constructs=len(constructs), components=len(components), candidate_antigens=len(panel),
                  kinds=dict(Counter(v[2] for v in constructs.values())), experiments=counts,
                  raw_sha256=sha256(root / 'raw.sqlite'), components_sha256=sha256(root / 'components.fasta'),
                  antigens_sha256=sha256(root / 'antigens.csv'))
    atomic_json(manifest_path, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','experiments')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/openalphaseq_pairing.yaml')
    prepare(parser.parse_args().config)
