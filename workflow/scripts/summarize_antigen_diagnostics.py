"""Cluster-paired uncertainty, region scores, and antigen sensitivity probes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antigen_diagnostics import atomic_json, read_csv, sha256


REGIONS = ('all', 'framework', 'cdr', 'cdr1', 'cdr2', 'cdr3', 'eos', 'unassigned')


def scored_examples(path, numbered, primary):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            item = json.loads(line)
            losses = item.pop('token_nll')
            sequence = item.pop('sequence')
            annotation = numbered['sequences'][hashlib.sha256(sequence.encode()).hexdigest()]
            if annotation['sequence'] != sequence or len(annotation['labels']) != len(losses):
                raise ValueError('Region labels and original target token losses differ')
            if not np.isfinite(losses).all():
                raise ValueError('Non-finite token scores')
            item['primary'] = item['id'] in primary
            for region in REGIONS:
                selected = [value for label, value in zip(annotation['labels'], losses)
                            if region == 'all' or label == region or (region == 'cdr' and label.startswith('cdr'))]
                item[f'{region}_sum'] = sum(selected)
                item[f'{region}_tokens'] = len(selected)
            rows.append(item)
    return rows


def paired_difference(left, right, region, repeats, seed):
    """Token-mean NLL(left)-NLL(right), resampling whole ab_ag_cluster groups."""
    key = lambda r: (r['id'], r['chain'])
    lhs, rhs = {key(r): r for r in left}, {key(r): r for r in right}
    if lhs.keys() != rhs.keys() or len(lhs) != len(left) or len(rhs) != len(right):
        raise ValueError('Paired diagnostic comparison has different or duplicate targets')
    groups = defaultdict(lambda: [0., 0])
    for k, a in lhs.items():
        b = rhs[k]
        if a['cluster'] != b['cluster'] or a[f'{region}_tokens'] != b[f'{region}_tokens']:
            raise ValueError('Paired diagnostic target group/token counts differ')
        groups[a['cluster']][0] += a[f'{region}_sum'] - b[f'{region}_sum']
        groups[a['cluster']][1] += a[f'{region}_tokens']
    array = np.asarray([groups[k] for k in sorted(groups) if groups[k][1]], dtype=float)
    if not len(array):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(repeats, len(array)))
    bootstrap = array[draws].sum(axis=1)
    values = bootstrap[:, 0] / bootstrap[:, 1]
    interval = np.quantile(values, [.025, .975])
    return dict(delta_nll=float(array[:, 0].sum() / array[:, 1].sum()),
                cluster_bootstrap_95ci=[float(interval[0]), float(interval[1])],
                n_clusters=len(array), n_examples=len(left), n_tokens=int(array[:, 1].sum()),
                equal_cluster_mean_delta=float(np.mean(array[:, 0] / array[:, 1])))


def summarize(config):
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root'])
    numbered = json.loads((root / 'regions.json').read_text())
    records_path = Path(study['cohort_dir']) / 'sabdab2_records.csv'
    if numbered['records_sha256'] != sha256(records_path):
        raise ValueError('Region annotations come from another cohort')
    primary = {r['id'] for r in read_csv(Path(study['cohort_dir']) / 'sabdab2_test_coverage.csv')
               if r['cath_assigned_all_antigen_chains'] == 'True'}
    all_rows = []
    source_audit = {}
    for name in study['probes']:
        folder = root / 'probes' / name
        marker = json.loads((folder / 'complete.json').read_text())
        scores_path = folder / 'token_scores.jsonl'
        if marker['token_scores_sha256'] != sha256(scores_path):
            raise ValueError('Probe per-token outputs changed')
        source_audit[name] = marker['source']
        all_rows.extend(scored_examples(scores_path, numbered, primary))
    with (root / 'per_example.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    scores, comparisons = [], []
    scopes = dict(train_probe=lambda r: r['split'] == 'train', val=lambda r: r['split'] == 'val',
                  test_primary=lambda r: r['split'] == 'test' and r['primary'], test_all=lambda r: r['split'] == 'test')
    for scope, include in scopes.items():
        selected = [r for r in all_rows if include(r)]
        runs = sorted({(r['model'], r['variant']) for r in selected})
        for model, variant in runs:
            rows = [r for r in selected if r['model'] == model and r['variant'] == variant]
            for region in REGIONS:
                tokens = sum(r[f'{region}_tokens'] for r in rows)
                if tokens:
                    scores.append(dict(model=model, variant=variant, scope=scope, region=region,
                                       nll=sum(r[f'{region}_sum'] for r in rows) / tokens, n_tokens=tokens,
                                       n_examples=len(rows), n_clusters=len({r['cluster'] for r in rows})))
        pairs = []
        for model in study['probes']:
            if study['probes'][model]['family'] == 'mango':
                pairs += [(model, variant, model, 'correct') for m, variant in runs if m == model and variant != 'correct']
        for suffix in ('scratch', 'pretrained'):
            pairs += [(f'{architecture}_{suffix}', 'correct', f'iglm_{suffix}', 'correct')
                      for architecture in ('prefix', 'cross')]
        for left_model, left_variant, right_model, right_variant in pairs:
            left = [r for r in selected if (r['model'], r['variant']) == (left_model, left_variant)]
            right = [r for r in selected if (r['model'], r['variant']) == (right_model, right_variant)]
            if not left or not right:
                continue
            for region in ('all', 'framework', 'cdr', 'cdr3'):
                effect = paired_difference(left, right, region, study['bootstrap_repeats'], study['seed'])
                if effect:
                    comparisons.append(dict(scope=scope, region=region, left_model=left_model, left_variant=left_variant,
                                            right_model=right_model, right_variant=right_variant, **effect))
    # Observational length bins, always paired against IgLM on the same targets.
    lengths = []
    selected = [r for r in all_rows if scopes['test_primary'](r) and r['variant'] == 'correct']
    for low, high in ((0, 128), (129, 256), (257, 512), (513, 10000)):
        for model in ('prefix_pretrained', 'cross_pretrained'):
            left = [r for r in selected if r['model'] == model and low <= r['antigen_length'] <= high]
            right = [r for r in selected if r['model'] == 'iglm_pretrained' and low <= r['antigen_length'] <= high]
            effect = paired_difference(left, right, 'all', study['bootstrap_repeats'], study['seed'])
            if effect:
                lengths.append(dict(model=model, low=low, high=high, **effect))
    result = dict(source_checkpoints=source_audit, region_annotation_status=numbered['status_counts'],
                  confidence_interval='paired ab_ag_cluster bootstrap; 2000 resamples; one training seed; exploratory, no multiple-comparison correction',
                  wrong_antigen_control='within split and different antigen/group; two donor draws; length-matched prefix by interpolation',
                  scores=scores, comparisons=comparisons, antigen_length_associations=lengths)
    atomic_json(root / 'summary.json', result)
    for filename, rows in [('scores.csv', scores), ('paired_comparisons.csv', comparisons), ('length_associations.csv', lengths)]:
        with (root / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for offset, scope, color, label in [(-.15, 'train_probe', '#4C78A8', 'Training inspection subset'),
                                        (.15, 'test_primary', '#F58518', 'Primary test')]:
        effects = [next(c for c in comparisons if c['scope'] == scope and c['region'] == 'all' and
                        c['left_model'] == name and c['left_variant'] == 'wrong_0')
                   for name in ('prefix_scratch', 'cross_scratch', 'prefix_pretrained', 'cross_pretrained')]
        y = [e['delta_nll'] for e in effects]
        axes[0].errorbar(np.arange(4) + offset, y,
            yerr=[[v - e['cluster_bootstrap_95ci'][0] for v, e in zip(y, effects)],
                  [e['cluster_bootstrap_95ci'][1] - v for v, e in zip(y, effects)]],
            fmt='o', capsize=3, color=color, label=label)
    axes[0].set_xticks(np.arange(4), ['Prefix\nnative', 'Cross\nnative', 'Prefix\nASD', 'Cross\nASD'])
    axes[0].set_ylabel('NLL(wrong antigen) − NLL(correct)')
    axes[0].set_title('Does the correct antigen help?')
    axes[0].legend(fontsize=8)
    for offset, name, color in [(-.15, 'prefix_pretrained', '#4C78A8'), (.15, 'cross_pretrained', '#F58518')]:
        effects = [next(c for c in comparisons if c['scope'] == 'test_primary' and c['region'] == region and
                        c['left_model'] == name and c['left_variant'] == 'correct' and c['right_model'] == 'iglm_pretrained')
                   for region in ('all', 'framework', 'cdr', 'cdr3')]
        y = [e['delta_nll'] for e in effects]
        axes[1].errorbar(np.arange(4) + offset, y,
            yerr=[[v - e['cluster_bootstrap_95ci'][0] for v, e in zip(y, effects)],
                  [e['cluster_bootstrap_95ci'][1] - v for v, e in zip(y, effects)]],
            fmt='o', capsize=3, color=color, label=name.replace('_pretrained', ''))
    axes[1].set_xticks(np.arange(4), ['All + EOS', 'Framework', 'All CDRs', 'CDR3'])
    axes[1].set_ylabel('NLL(MANGO) − NLL(IgLM)')
    axes[1].set_title('Where is the gap? Primary test')
    axes[1].legend(fontsize=8)
    for name, color in [('prefix_pretrained', '#4C78A8'), ('cross_pretrained', '#F58518')]:
        items = [r for r in lengths if r['model'] == name]
        axes[2].plot([f'{r["low"]}–{min(r["high"], 999)}' if r['high'] < 10000 else '513+' for r in items],
                     [r['delta_nll'] for r in items], marker='o', color=color, label=name.replace('_pretrained', ''))
    axes[2].set_ylabel('NLL(MANGO) − NLL(IgLM)')
    axes[2].set_xlabel('Antigen sequence length; observational bins')
    axes[2].set_title('Gap versus antigen length')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color='gray', linewidth=1)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Antigen-conditioning diagnostics: original one-epoch pilot checkpoints')
    fig.tight_layout()
    fig.savefig(root / 'diagnostics.png', dpi=180)
    plt.close(fig)
    print('Wrote paired antigen diagnostics and region/length summaries', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/antigen_diagnostics.yaml')
    summarize(parser.parse_args().config)
