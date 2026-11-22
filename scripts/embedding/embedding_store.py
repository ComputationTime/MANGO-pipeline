"""Read compact antigen caches without storing a tensor per antibody variant.

An optional store.json beside the split directories maps virtual record paths
through a read-only SQLite index. Existing per-record .pt files still work.
Only antigen identities, lengths and frozen tensors enter this store.
"""
from functools import lru_cache
import json
from pathlib import Path
import sqlite3

import torch


def embedding_available(path):
    path = Path(path)
    return path.is_file() or (path.parent.parent / 'store.json').is_file()


@lru_cache(maxsize=32)
def _store(path):
    config = json.loads(Path(path).read_text())
    if config.get('format') != 'mango_antigen_store_v1':
        raise ValueError(f'Unknown antigen store: {path}')
    db = sqlite3.connect(f"file:{config['index']}?mode=ro", uri=True)
    return config, db


@lru_cache(maxsize=64)
def _matrix(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    matrix = payload.get('embedding')
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or payload.get('shape') != list(matrix.shape):
        raise ValueError(f'Invalid frozen antigen cache: {path}')
    return matrix


def load_stored_embedding(path):
    path = Path(path)
    config, db = _store(str((path.parent.parent / 'store.json').resolve()))
    row = db.execute('SELECT antigen_key,length FROM records WHERE split=? AND id=?',
                     (path.parent.name, path.stem)).fetchone()
    if row is None:
        raise FileNotFoundError(f'Record absent from antigen store: {path}')
    antigen_key, length = row
    mode = config['mode']
    if mode in {'constant', 'random'}:
        from embed_antigen_control import control_matrix
        fixed = config.get('tokens')
        spec = dict(mode=mode, width=config['width'], seed=config['seed'],
                    shape_mode='fixed' if fixed else 'match_source', tokens=fixed or 1)
        matrix = control_matrix(path.stem, spec, shape=(length, config['width']))
    elif mode in {'esm2', 'pooled', 'shuffled'}:
        matrix = _matrix(str(Path(config['cache_dir']) / f'{antigen_key}.pt'))
        if list(matrix.shape) != [1 if mode == 'pooled' else length, config['width']]:
            raise ValueError(f'Antigen-store shape mismatch: {path}')
    else:
        raise ValueError(f'Unknown antigen-store mode: {mode}')
    return matrix.unsqueeze(0)
