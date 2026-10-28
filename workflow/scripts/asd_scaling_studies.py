"""Frozen reference traversal and the attainable marginal-preserving shuffle."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path


def studies_in_order(study):
    import yaml
    result, seen = [], {}
    def visit(current, ancestors):
        root = current['experiment_root']
        if root in ancestors:
            raise ValueError('Cycle in ASD scaling references')
        if root in seen:
            if seen[root] != current:
                raise ValueError('Conflicting configurations for one scaling root')
            return
        reference = current.get('reference_config')
        if reference:
            visit(yaml.safe_load(Path(reference).read_text()), ancestors | {root})
        seen[root] = current
        result.append(current)
    visit(study, set())
    names = [run for current in result for run in current['runs']]
    if len(names) != len(set(names)):
        raise ValueError('Scaling references contain duplicate run names')
    return result


def maximal_record_shuffle(rows, seed=0):
    """Minimize unchanged antigen identities while preserving record marginals.

    If the largest antigen occupies L of N records, at least max(0,2L-N)
    records must retain that antigen. Report this fraction rather than claiming
    that all pairs are mismatched when the requested control is impossible.
    """
    splits = defaultdict(list)
    for row in rows:
        splits[row['split']].append(row)
    result = {}
    for split, members in splits.items():
        ordered = sorted(members, key=lambda r: (
            hashlib.sha256(f'{seed}:{r["resolved_ag_seq"]}'.encode()).digest(), r['id']))
        largest = max(Counter(r['resolved_ag_seq'] for r in ordered).values())
        if largest == len(ordered):
            raise ValueError(f'No antigen identity can be changed in {split}')
        donors = ordered[largest:] + ordered[:largest]
        same = sum(t['resolved_ag_seq'] == d['resolved_ag_seq'] for t, d in zip(ordered, donors))
        if same != max(0, 2 * largest - len(ordered)):
            raise ValueError('Training shuffle failed to attain the minimum unchanged count')
        result.update((t['id'], d) for t, d in zip(ordered, donors))
    return result
