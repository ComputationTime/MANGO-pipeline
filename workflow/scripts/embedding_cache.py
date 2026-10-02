"""Explicit, content-verified reuse of an audited antigen cache.

An optional index authorizes existing tensors after target-only table changes.
It cannot authorize reuse after antigen, implementation, configuration or tensor
changes. Reused records are still reported as reused by the benchmark runner.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def antigen_identity(row):
    fields = ('id', 'split', 'antigen_chains', 'resolved_ag_seq', 'expected_ag_seq', 'pdb_path')
    return {key: row.get(key, '') for key in fields}


def read_index(output_dir, signature):
    path = Path(output_dir) / 'audited_cache.json'
    try:
        index = json.loads(path.read_text())
        if index['signature'] != signature:
            return {}
        if any(digest(p) != sha for p, sha in index['implementation'].items()):
            return {}
        return index['records']
    except (OSError, ValueError, KeyError):
        return {}


def verified_record(index, row, output, dependencies):
    try:
        entry = index[f"{row['split']}/{row['id']}"]
        if entry['antigen_identity'] != antigen_identity(row):
            return False
        if digest(output) != entry['tensor_sha256']:
            return False
        # Only the records table (first dependency) may have a newer mtime.
        # Preserve ordinary invalidation for explicit structure/weight inputs.
        output_mtime = Path(output).stat().st_mtime_ns
        for name in dependencies[1:]:
            asset = Path(name)
            if asset.suffix != ".py" and asset.stat().st_mtime_ns > output_mtime:
                return False
        # Structures and explicit weight files retain their original identities.
        for name, expected in entry.get('assets', {}).items():
            stat = Path(name).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False
