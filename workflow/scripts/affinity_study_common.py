"""Frozen protocol, comparisons and decisions for the unattended affinity study."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

INTERVENTIONS = ('balance', 'diversity', 'decoder', 'ranking', 'cdr')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def freeze(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen inputs changed: {path}; use a new experiment root')
    else:
        write_json(path, value)


def fingerprint(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def row_key(row, seed=1729):
    return fingerprint(f'{seed}:{row["id"]}')


def training_weights(examples, balanced=False):
    counts = Counter(x['antigen_id'] for x in examples)
    # Normalize so the average example weight is exactly one; this changes
    # relative exposure, not overall gradient scale or target membership.
    return [len(examples) / (len(counts) * counts[x['antigen_id']]) if balanced else 1.
            for x in examples]


def advancement(differences, settings, groups, affinity=False):
    if len(differences) != len(settings['pilot_seeds']):
        return dict(advance=False, reason='missing_seed_results')
    if groups < settings['minimum_development_groups']:
        return dict(advance=False, reason='insufficient_independent_development_groups')
    # Differences are improvement-oriented: positive always favors candidate.
    threshold = settings['affinity_improvement'] if affinity else settings['nll_improvement']
    mean = sum(differences) / len(differences)
    return dict(advance=mean >= threshold and sum(d > 0 for d in differences) >= 2,
                reason='affinity_screen' if affinity else 'sequence_modeling_screen',
                mean_improvement=mean, threshold=threshold, seed_improvements=differences)


def holm(pvalues, alpha=.05):
    """Five planned intervention hypotheses, including infeasible/null arms."""
    ordered = sorted(pvalues, key=pvalues.get)
    rejected = {key:False for key in pvalues}
    for i, key in enumerate(ordered):
        if pvalues[key] > alpha / (len(ordered) - i):
            break
        rejected[key] = True
    return rejected


def paired_group_bootstrap(rows, repeats=2000, seed=1729):
    """Token-weighted NLL improvement; identical biological groups across seeds.

    Each row contains paired loss sums and token counts for one biological
    group, already averaged over matched seeds. Seeds are not biological units.
    """
    import numpy as np
    if len(rows) < 2:
        return dict(low=None, high=None, pvalue=1., groups=len(rows))
    a = np.asarray([[r['improvement_sum'], r['ntokens']] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        draw = a[rng.integers(0, len(a), size=len(a))].sum(axis=0)
        values.append(float(draw[0] / draw[1]))
    low, high = np.quantile(values, [.025, .975])
    # Centered bootstrap null distribution, two-sided, with finite-sample floor.
    observed = a[:,0].sum() / a[:,1].sum()
    pvalue = (1 + sum(abs(x-observed) >= abs(observed) for x in values)) / (repeats+1)
    return dict(low=float(low), high=float(high), pvalue=float(pvalue), groups=len(rows),
                improvement=float(observed))


def retry_kind(stderr):
    text = stderr.lower()
    if 'cuda out of memory' in text or 'cuda error: out of memory' in text:
        return 'oom'
    if any(x in text for x in ('connection reset', 'temporary failure', 'timed out', 'http 503', 'http 502')):
        return 'transient'
    return None
