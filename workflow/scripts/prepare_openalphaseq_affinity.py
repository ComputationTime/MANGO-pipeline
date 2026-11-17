"""Prepare the complete AlphaSeq affinity cohort against the native study train split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from finalize_openalphaseq_pairing import digest, prepare_training_fasta, write_json
from prepare_openalphaseq_pairing import prepare


def merge_config(paths):
    def update(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                update(target[key], value)
            else:
                target[key] = value
    result = {}
    for path in paths:
        update(result, yaml.safe_load(Path(path).read_text()))
    return result


def model_directories(config):
    """Use the launcher's scientific identities, independent of execution policy."""
    from iglm_baseline import experiment_hash as iglm_hash
    c = merge_config(config['training_configs'])
    result = {}
    for tag in c['active_embedders']:
        model = {**c['model'], **c['embedders'][tag].get('model_overrides', {})}
        for key in ('retrain', 'loss_plot_interval'):
            model.pop(key, None)
        payload = dict(dataset=c['dataset'], processing=c['processing'], embedding=c['embedding'],
                       embedder=c['embedders'][tag], model=model, weights=c['weights'],
                       experiment_seed=c['experiment']['seed'])
        identity = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
        result[tag] = dict(directory=f'artifacts/runs/{tag}__{identity}', family='mango',
                           spec=c['embedders'][tag])
    result['iglm'] = dict(directory=f'artifacts/runs/iglm_scratch__{iglm_hash(c)}', family='iglm')
    return result


def main(config):
    study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root']); root.mkdir(parents=True, exist_ok=True)
    records = Path(study['training_records'])
    rows = list(csv.DictReader(records.open()))
    splits = {}
    for row in rows:
        splits.setdefault(row['ab_ag_cluster'], set()).add(row['split'])
    if any(len(values) != 1 for values in splits.values()):
        raise ValueError('A training antigen/antibody cluster crosses data splits')
    if set(r['split'] for r in rows) != {'train', 'val', 'test'}:
        raise ValueError('Expected the full cluster-separated native cohort')
    audit = dict(complete=True, sha256={str(records): digest(records)},
                 records=len(rows), clusters_crossing_splits=0,
                 splits={s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')})
    write_json(study['training_audit'], audit)
    prepare(config)
    source = Path(study['annotation_source'])
    source_marker = json.loads((source / 'numbering_complete.json').read_text())
    if not source_marker['complete'] or source_marker['source_sha256'] != digest(root / 'components.fasta'):
        raise ValueError('Cached expression-domain annotations do not match the full source snapshot')
    for name in ('domains.sqlite', 'numbering_complete.json'):
        destination = root / name
        if not destination.exists():
            shutil.copyfile(source / name, destination)
        elif digest(destination) != digest(source / name):
            raise ValueError(f'Cached annotations changed: {name}')
    prepare_training_fasta(study)
    antigen_rows = list(csv.DictReader((root / 'antigens.csv').open()))
    for row in antigen_rows:
        row['antigen_chains'] = 'A'
        row['pdb_path'] = str(root / 'structures' / row['id'] / 'antigen.cif')
    with (root / 'antigen_records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(antigen_rows[0]))
        writer.writeheader(); writer.writerows(antigen_rows)
    resolved = {**study, 'models': model_directories(study)}
    write_json(root / 'study.json', resolved)
    print(json.dumps(dict(root=str(root), models=resolved['models'], training_audit=audit), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/openalphaseq_affinity.yaml')
    main(**vars(parser.parse_args()))
