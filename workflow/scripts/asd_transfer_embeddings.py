"""Shared frozen antigen tensors and compact indexes for large ASD libraries."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import torch

from prepare_asd_transfer import sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_index(path, assignments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.sqlite')
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.execute('CREATE TABLE records (split TEXT, id TEXT, antigen_key TEXT, length INTEGER, PRIMARY KEY(split,id)) WITHOUT ROWID')
        db.executemany('INSERT INTO records VALUES (?,?,?,?)', assignments)
    temporary.replace(path)


def finish(output, marker, config, records, count):
    write_json(output / 'store.json', dict(format='mango_antigen_store_v1', **config,
                                         records_sha256=sha256(records)))
    write_json(marker, dict(status='complete', records=count, records_sha256=sha256(records),
                            store=str(output / 'store.json'), store_sha256=sha256(output / 'store.json'),
                            index_sha256=sha256(config['index'])))


def embed_esm2(study, phase, records, marker):
    import embed_common as ec
    import embed_antigen_esm2 as esm
    root = Path(study['experiment_root'])
    output = root / phase / 'embeddings' / 'antigen' / 'esm2'
    cache = Path(study.get('esm2_cache_dir', root / 'cache' / 'esm2'))
    rows = list(csv.DictReader(open(records)))
    unique, assignments = {}, []
    for row in rows:
        sequences = list(ec.antigen_sequences(row, 'resolved').items())
        key = hashlib.sha256(json.dumps(sequences).encode()).hexdigest()
        length = sum(len(s) for _, s in sequences) + len(sequences) - 1
        unique.setdefault(key, (row, length))
        assignments.append((row['split'], row['id'], key, length))
    runtime = None
    model_name = f"esm2_{study['esm2_size']}_UR50D"
    for index, (key, (row, length)) in enumerate(sorted(unique.items()), 1):
        path = cache / f'{key}.pt'
        if not path.is_file():
            if runtime is None:
                runtime = esm.load_runtime(study['esm2_size'])
            temporary = path.with_suffix('.tmp.pt')
            esm.esm2(records, row['id'], str(temporary), 'resolved', study['esm2_size'],
                     'esm2', row=row, runtime=runtime)
            temporary.replace(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('embedder') != 'esm2' or payload.get('model_name') != model_name or payload.get('shape') != [length, 1280]:
            raise ValueError(f'Invalid shared frozen ESM2 tensor: {path}')
        if not isinstance(payload.get('embedding'), torch.Tensor) or list(payload['embedding'].shape) != [length, 1280]:
            raise ValueError(f'Missing shared ESM2 tensor: {path}')
        if index % 20 == 0 or index == len(unique):
            print(f'{phase}: {index}/{len(unique)} unique antigens ready ({len(rows):,} records)', flush=True)
    index_path = output / 'index.sqlite'
    write_index(index_path, assignments)
    finish(output, marker, dict(mode='esm2', width=1280, index=str(index_path.resolve()),
                               cache_dir=str(cache.resolve()), model_name=model_name), records, len(rows))


def embed_controls(study, phase, tag, records, marker):
    root = Path(study['experiment_root']) / phase
    source = root / 'embeddings' / 'antigen' / 'esm2'
    output = root / 'embeddings' / 'antigen' / tag
    config = json.loads((source / 'store.json').read_text())
    if config['records_sha256'] != sha256(records):
        raise ValueError('Source antigen-store cohort differs')
    with sqlite3.connect(f"file:{config['index']}?mode=ro", uri=True) as db:
        assignments = list(db.execute('SELECT split,id,antigen_key,length FROM records ORDER BY split,id'))
    spec = dict(width=1280, index=config['index'], cache_dir=config['cache_dir'], seed=0)
    if tag in {'control_pooled_esm2', 'control_shuffled'}:
        from model_common import comparison_cohort
        baseline = json.loads((root / 'runs' / 'esm2' / 'eval.json').read_text())
        if baseline['cohort_sha256'] != comparison_cohort(records) or baseline['embedder'] != 'esm2':
            raise ValueError('Source ESM2 evaluation differs from the ASD cohort')
    if tag == 'control_pooled_esm2':
        pooled_cache = Path(study['experiment_root']) / 'cache' / 'pooled_esm2'
        pooled_cache.mkdir(parents=True, exist_ok=True)
        for key in sorted({r[2] for r in assignments}):
            source_path = Path(config['cache_dir']) / f'{key}.pt'
            payload = torch.load(source_path, map_location='cpu', weights_only=False)
            matrix = payload['embedding'].mean(dim=0, keepdim=True)
            temporary = pooled_cache / f'{key}.tmp.pt'
            torch.save(dict(embedding=matrix, shape=list(matrix.shape), source_sha256=sha256(source_path)), temporary)
            temporary.replace(pooled_cache / f'{key}.pt')
        spec.update(mode='pooled', cache_dir=str(pooled_cache.resolve()))
    elif tag == 'control_shuffled':
        from shuffle_antigen_embeddings import derangement
        output.mkdir(parents=True, exist_ok=True)
        donor_assignments, pairing = [], {}
        mapping_path = output / 'mapping.csv'
        with mapping_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'target_id', 'donor_id', 'target_equals_donor', 'same_antigen_sequence'])
            for split in ('train', 'val', 'test'):
                members = {r[1]: (r[2], r[3]) for r in assignments if r[0] == split}
                ids = sorted(members)
                donors = derangement(ids, 0, split)
                correct = 0
                for target, donor in zip(ids, donors):
                    key, length = members[donor]
                    same = key == members[target][0]
                    correct += same
                    donor_assignments.append((split, target, key, length))
                    writer.writerow([split, target, donor, False, same])
                pairing[split] = dict(records=len(ids), same_antigen_sequence=correct)
        index_path = output / 'index.sqlite'
        write_index(index_path, donor_assignments)
        write_json(output / 'biological_pairing_audit.json', pairing)
        spec.update(mode='shuffled', index=str(index_path.resolve()), mapping_sha256=sha256(mapping_path))
    elif tag in {'control_constant', 'control_random', 'control_constant_l1', 'control_random_l1'}:
        spec.update(mode='constant' if 'constant' in tag else 'random',
                    tokens=1 if tag.endswith('_l1') else None)
    else:
        raise ValueError(f'Unsupported compact ASD control: {tag}')
    finish(output, marker, spec, records, len(assignments))
