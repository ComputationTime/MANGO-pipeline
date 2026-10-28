"""Verify ASD scaling budgets and estimate paired antigen-specific scale effects."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antigen_diagnostics import atomic_json, read_csv, sha256
from chain_examples import example_chains
from verify_asd_loss_comparison import fingerprint
from prepare_asd_scaling import paths
from summarize_antigen_diagnostics import REGIONS, paired_difference, scored_examples


def verify_run(study, run, spec):
    folder = Path(study['experiment_root']) / 'runs' / run
    location = paths(study, spec['size'])
    value = json.loads((folder / 'eval.json').read_text())
    ready = json.loads((folder / 'train_complete.json').read_text())
    if (value['spec'] != spec or ready['spec'] != spec or value['training'] != ready or
            value['heldout_sha256'] != sha256(Path(study['experiment_root']) / 'evaluation_records.csv') or
            ready['cohort_sha256'] != fingerprint(read_csv(location['records'])) or
            ready['records_sha256'] != sha256(location['records']) or
            ready['checkpoint_sha256'] != sha256(ready['checkpoint']) or
            ready['evaluated_epoch'] != spec['epochs'] - 1 or
            value['token_scores_sha256'] != sha256(folder / 'token_scores.jsonl') or
            value['evaluation_loss'] != 'cross_entropy' or
            json.loads((folder / 'training_runtime.json').read_text())['status'] != 'complete'):
        raise ValueError('Scaling cohort/checkpoint/scoring provenance mismatch')
    records = read_csv(location['records'])
    expected = Counter((r['id'], chain) for r in records if r['split'] == 'train' for chain in example_chains(r))
    curve = read_csv(folder / 'training_curve.csv')
    for epoch in range(spec['epochs']):
        actual = Counter((r['record_id'], r['chain_type']) for r in curve if r['phase'] == 'train' and int(r['epoch']) == epoch)
        if actual != expected:
            raise ValueError('Scaling epoch did not train each requested masked target once')
    history = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines() if x]
    updates = len(expected) * spec['epochs']
    if len(history) != spec['epochs'] or history[-1]['iteration'] != updates:
        raise ValueError('Scaling optimization budget differs')
    tokens_per_epoch = sum(len(r['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']) + 1
                           for r in records if r['split'] == 'train' for chain in example_chains(r))
    return dict(run=run, passed=True, training_records=spec['size'], epochs=spec['epochs'],
                training_examples_per_epoch=sum(expected.values()), optimizer_updates=updates,
                training_target_tokens=tokens_per_epoch * spec['epochs'], evaluated_epoch=ready['evaluated_epoch'])


def average_rows(left, right, variant):
    rhs = {(r['id'], r['chain']): r for r in right}
    result = []
    if len(left) != len(right):
        raise ValueError('Wrong-antigen repetitions have different membership')
    for row in left:
        other = rhs[row['id'], row['chain']]
        value = dict(row, variant=variant)
        for region in REGIONS:
            if row[f'{region}_tokens'] != other[f'{region}_tokens']:
                raise ValueError('Wrong-antigen repetitions have different target counts')
            value[f'{region}_sum'] = (row[f'{region}_sum'] + other[f'{region}_sum']) / 2
        result.append(value)
    return result


def difference_rows(left, right):
    rhs = {(r['id'], r['chain']): r for r in right}
    if len(left) != len(right):
        raise ValueError('Scale contrast has different membership')
    result = []
    for row in left:
        other = rhs[row['id'], row['chain']]
        value = dict(row)
        for region in REGIONS:
            if row[f'{region}_tokens'] != other[f'{region}_tokens']:
                raise ValueError('Scale contrast has different target counts')
            value[f'{region}_sum'] -= other[f'{region}_sum']
        result.append(value)
    return result


def report(config, require_complete=False, status_only=False):
    study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root'])
    stages = []
    for run, spec in study['runs'].items():
        if not (root / 'runs' / run / 'eval.json').is_file():
            stages.append(dict(run=run, passed=False, missing='eval.json'))
            continue
        try:
            stages.append(verify_run(study, run, spec))
        except (ValueError, KeyError, FileNotFoundError) as error:
            stages.append(dict(run=run, passed=False, error=str(error)))
    complete = all(s['passed'] for s in stages)
    status = dict(complete=complete, completed_models=sum(s['passed'] for s in stages),
                  expected_models=len(stages), runs=stages)
    if status_only:
        status['models_complete'] = complete
        status['analysis_complete'] = (root / 'summary.json').is_file() and (root / 'asd_scaling.png').is_file()
        status['complete'] = complete and status['analysis_complete']
        atomic_json(root / 'results_report.json', status)
        print(json.dumps(status, indent=2))
        return status
    atomic_json(root / 'results_report.json', status)
    if not complete or not (root / 'regions.json').is_file():
        if not (root / 'regions.json').is_file():
            status['complete'] = False
            status['missing_analysis'] = 'regions.json'
            atomic_json(root / 'results_report.json', status)
        print(json.dumps(status, indent=2))
        if require_complete:
            raise SystemExit(1)
        return status
    numbered = json.loads((root / 'regions.json').read_text())
    if numbered['records_sha256'] != sha256(root / 'evaluation_records.csv'):
        raise ValueError('ASD region annotations differ from fixed held-out records')
    heldout = read_csv(root / 'evaluation_records.csv')
    heldout_by_id = {r['id']: r for r in heldout}
    test_ids = {r['id'] for r in heldout if r['split'] == 'test'}
    expected = {(r['id'], chain): (r['split'], r['ab_ag_cluster'], len(r['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']) + 1)
                for r in heldout for chain in example_chains(r)}
    values = []
    for run, spec in study['runs'].items():
        rows = scored_examples(root / 'runs' / run / 'token_scores.jsonl', numbered, test_ids)
        variants = ['correct'] if spec['family'] == 'iglm' else ['correct', 'wrong_0', 'wrong_1', 'no_context']
        for variant in variants:
            members = [r for r in rows if r['variant'] == variant]
            keys = [(r['id'], r['chain']) for r in members]
            if len(keys) != len(set(keys)) or set(keys) != set(expected):
                raise ValueError('Scaling model scored a different held-out target set')
            for row in members:
                if (row['split'], row['cluster'], row['all_tokens']) != expected[row['id'], row['chain']]:
                    raise ValueError('Scaling model scored different tokens/groups')
        values.extend(rows)
        if spec['family'] != 'iglm':
            values.extend(average_rows([r for r in rows if r['variant'] == 'wrong_0'],
                                      [r for r in rows if r['variant'] == 'wrong_1'], 'wrong_mean'))
            far_marker = json.loads((root / 'runs' / run / 'far_eval.json').read_text())
            ready = json.loads((root / 'runs' / run / 'train_complete.json').read_text())
            far_path = root / 'runs' / run / 'far_token_scores.jsonl'
            if (far_marker['checkpoint_sha256'] != ready['checkpoint_sha256'] or
                    far_marker['heldout_sha256'] != sha256(root / 'evaluation_records.csv') or
                    far_marker['token_scores_sha256'] != sha256(far_path)):
                raise ValueError('Different-family probe provenance differs')
            far_rows = scored_examples(far_path, numbered, test_ids)
            for variant in ('far_wrong_0', 'far_wrong_1'):
                members = [r for r in far_rows if r['variant'] == variant]
                keys = [(r['id'], r['chain']) for r in members]
                if len(keys) != len(set(keys)) or set(keys) != set(expected):
                    raise ValueError('Different-family probe target membership differs')
                for row in members:
                    target, donor = heldout_by_id[row['id']], heldout_by_id[row['donor_id']]
                    if (target['split'] != donor['split'] or target['ab_ag_cluster'] == donor['ab_ag_cluster'] or
                            set(target['cath_families'].split(';')) & set(donor['cath_families'].split(';')) or
                            row['all_tokens'] != expected[row['id'], row['chain']][2]):
                        raise ValueError('Different-family donor or target token count differs')
            values.extend(far_rows)
            values.extend(average_rows([r for r in far_rows if r['variant'] == 'far_wrong_0'],
                                      [r for r in far_rows if r['variant'] == 'far_wrong_1'], 'far_wrong_mean'))
    scores, contrasts = [], []
    for split in ('val', 'test'):
        selected = [r for r in values if r['split'] == split]
        get = lambda model, variant='correct': [r for r in selected if r['model'] == model and r['variant'] == variant]
        runs = sorted({(r['model'], r['variant']) for r in selected})
        for model, variant in runs:
            rows = get(model, variant)
            for chain in ('both', 'heavy', 'light'):
                members = rows if chain == 'both' else [r for r in rows if r['chain'] == chain]
                for region in REGIONS:
                    count = sum(r[f'{region}_tokens'] for r in members)
                    if count:
                        scores.append(dict(model=model, variant=variant, split=split, chain=chain, region=region,
                            nll=sum(r[f'{region}_sum'] for r in members) / count, n_tokens=count,
                            n_examples=len(members), n_clusters=len({r['cluster'] for r in members})))
        def compare(label, left, right):
            for region in ('all', 'framework', 'cdr', 'cdr3'):
                estimate = paired_difference(left, right, region, study['bootstrap_repeats'], study['seed'])
                if estimate:
                    contrasts.append(dict(contrast=label, split=split, region=region, **estimate))
        for size in study['sizes']:
            compare(f'cross_minus_iglm_n{size}', get(f'n{size}_cross'), get(f'n{size}_iglm'))
            compare(f'paired_minus_shuffled_training_n{size}', get(f'n{size}_cross'), get(f'n{size}_shuffled'))
            for suffix in ('cross', 'shuffled'):
                model = f'n{size}_{suffix}'
                compare(f'wrong_minus_correct_{model}', get(model, 'wrong_mean'), get(model))
                compare(f'far_wrong_minus_correct_{model}', get(model, 'far_wrong_mean'), get(model))
        for suffix in ('iglm', 'cross', 'shuffled'):
            compare(f'scale_4k_minus_2k_{suffix}', get(f'n4000_{suffix}'), get(f'n2000_{suffix}'))
        compare('change_in_cross_minus_iglm_gap', difference_rows(get('n4000_cross'), get('n4000_iglm')),
                difference_rows(get('n2000_cross'), get('n2000_iglm')))
        compare('change_in_antigen_benefit', difference_rows(get('n4000_cross', 'wrong_mean'), get('n4000_cross')),
                difference_rows(get('n2000_cross', 'wrong_mean'), get('n2000_cross')))
        compare('change_in_far_antigen_benefit', difference_rows(get('n4000_cross', 'far_wrong_mean'), get('n4000_cross')),
                difference_rows(get('n2000_cross', 'far_wrong_mean'), get('n2000_cross')))
        for suffix in ('iglm', 'cross'):
            compare(f'4k_once_minus_2k_twice_{suffix}', get(f'n4000_{suffix}'), get(f'n2000_{suffix}_2ep'))
            compare(f'2k_twice_minus_once_{suffix}', get(f'n2000_{suffix}_2ep'), get(f'n2000_{suffix}'))
        compare('cross_minus_iglm_2k_twice', get('n2000_cross_2ep'), get('n2000_iglm_2ep'))
    for name, rows in [('scores.csv', scores), ('paired_comparisons.csv', contrasts)]:
        with (root / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cohort = json.loads((root / 'cohort_audit.json').read_text())
    status.update(complete=True, fixed_heldout_verified=True, target_counts_verified=True,
                  analysis_complete=True, test_groups=len({r['ab_ag_cluster'] for r in heldout if r['split'] == 'test'}))
    summary = dict(status=status, cohorts=cohort, scores=scores, contrasts=contrasts,
                   region_status=numbered['status_counts'],
                   interpretation='ASD-only, nested training; shared 768-residue eligibility; correct/shuffled pairing controls; single seed; paired group bootstrap. Two-epoch controls use final checkpoint and have approximately the same updates as 4k once.')
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'results_report.json', status)
    draw(root, summary)
    write_findings(root, summary)
    print(json.dumps(status, indent=2))
    return summary


def write_findings(root, summary):
    scores, contrasts = summary['scores'], summary['contrasts']
    value = lambda model, region='all': next(r['nll'] for r in scores if r['model'] == model and r['variant'] == 'correct'
                                            and r['split'] == 'test' and r['chain'] == 'both' and r['region'] == region)
    contrast = lambda name: next(r for r in contrasts if r['contrast'] == name and r['split'] == 'test' and r['region'] == 'all')
    lines = ['# ASD-only data-size experiment', '',
             'Eight models completed. All trained from random weights using ordinary CE; antigen models use frozen ESM2 650M.', '',
             '| Training records | Epochs | Model | ASD test NLL | ASD CDR3 NLL |',
             '|---:|---:|---|---:|---:|']
    for size in (2000, 4000):
        for suffix, label in [('iglm', 'IgLM'), ('cross', 'Paired cross-attention'), ('shuffled', 'Shuffled-training cross-attention')]:
            model = f'n{size}_{suffix}'
            lines.append(f'| {size:,} | 1 | {label} | {value(model):.6f} | {value(model,"cdr3"):.6f} |')
    for suffix, label in [('iglm', 'IgLM'), ('cross', 'Paired cross-attention')]:
        model = f'n2000_{suffix}_2ep'
        lines.append(f'| 2,000 | 2 | {label} | {value(model):.6f} | {value(model,"cdr3"):.6f} |')
    lines += ['', '## Antigen-dependent and scale effects', '',
              'Contrasts below are paired over identical test targets. Intervals resample entire `ab_ag_cluster` groups, not individual residues.', '',
              '| Contrast | NLL difference | 95% paired group interval |', '|---|---:|---|']
    for key, label in [('cross_minus_iglm_n2000', 'MANGO − IgLM, 2k once'),
                       ('cross_minus_iglm_n4000', 'MANGO − IgLM, 4k once'),
                       ('change_in_cross_minus_iglm_gap', 'Change in MANGO − IgLM gap, 4k versus 2k'),
                       ('far_wrong_minus_correct_n2000_cross', 'Different-family wrong antigen − correct, 2k paired model'),
                       ('far_wrong_minus_correct_n4000_cross', 'Different-family wrong antigen − correct, 4k paired model'),
                       ('change_in_far_antigen_benefit', 'Change in different-family antigen benefit, 4k versus 2k'),
                       ('paired_minus_shuffled_training_n2000', 'Paired − shuffled training, 2k'),
                       ('paired_minus_shuffled_training_n4000', 'Paired − shuffled training, 4k'),
                       ('4k_once_minus_2k_twice_iglm', 'IgLM: 4k once − 2k twice'),
                       ('4k_once_minus_2k_twice_cross', 'MANGO: 4k once − 2k twice')]:
        effect = contrast(key)
        low, high = effect['cluster_bootstrap_95ci']
        lines.append(f'| {label} | {effect["delta_nll"]:+.6f} | [{low:+.6f}, {high:+.6f}] |')
    lines += ['', 'Positive wrong-minus-correct differences mean the correct antigen helps. Negative MANGO-minus-IgLM differences favor MANGO.', '',
              '## Controls and interpretation limits', '',
              '- The 4,000-record training set contains the entire 2,000-record set with unchanged target masks. Both contain the same 1,666 antigens, 600 CATH families and 79 split-graph components. Added data supplies more antibody examples per antigen.',
              '- Validation/test remain the original 95/200 ASD records, with 185/393 scored chain targets. Test scoring includes 44,934 residue/EOS tokens and only ten split-graph components.',
              '- The old 512-residue eligibility rule could not supply 4,000 capped records. Both new baselines were therefore trained under a shared 768-residue limit and ten-record-per-antigen cap. HER2/BUZZ exclusions and protected split assignments remain intact.',
              '- Two-epoch controls evaluate their final checkpoint to retain the intended optimization budget. They perform 7,976 updates, versus 7,876 for 4k once; the exact token budgets are in `results_report.json`.',
              '- Training shuffles preserve antigen record marginals and stay within train. Random and different-family inference donors stay within the scored split; the latter share no CATH family or split component with the target antigen.',
              '- No SAbDab2 fine-tuning or evaluation was performed. Existing native comparison artifacts remain separate. All ESM2 tensors were reused; no encoder forwards were recomputed.',
              '- These are exploratory, single-seed results. Test-cluster intervals do not include training-seed uncertainty or adjustments for multiple comparisons. Two training sizes do not establish an asymptotic scaling law.', '',
              'See `asd_scaling.png`, `summary.json`, `scores.csv`, `paired_comparisons.csv`, and `docs/asd_scaling.md` for the full results and protocol.', '']
    (root / 'findings.md').write_text('\n'.join(lines))


def draw(root, summary, sizes=(2000, 4000)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores = summary['scores']
    lookup = lambda model, region='all': next(r['nll'] for r in scores if r['model'] == model and r['variant'] == 'correct'
                                             and r['split'] == 'test' and r['chain'] == 'both' and r['region'] == region)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = dict(iglm='#4C78A8', mango='#54A24B', fixed='#B279A2', cross='#F58518', shuffled='#777777')
    labels = dict(iglm='IgLM', mango='Regular MANGO (prefix)', fixed='MANGO, fixed antibody positions', cross='Cross-attention, paired',
                  shuffled='Cross-attention, shuffled training')
    available = {r['model'] for r in scores}
    for ax, region, title in [(axes[0], 'all', 'ASD test: all residues + EOS'),
                              (axes[1], 'cdr3', 'ASD test: CDR3 residues')]:
        for family in colors:
            selected_sizes = [n for n in sizes if f'n{n}_{family}' in available]
            if not selected_sizes:
                continue
            ax.plot(selected_sizes, [lookup(f'n{n}_{family}', region) for n in selected_sizes],
                    marker='o', color=colors[family], label=labels[family])
        ax.set_xticks(sizes, [f'{n:,}' for n in sizes])
        ax.set_xlabel('ASD training records; one epoch')
        ax.set_ylabel('NLL (nats/target token); lower is better')
        ax.set_title(title)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('HER2-free ASD-only scaling; fixed 95-record validation / 200-record test sets')
    if 16000 in sizes:
        availability = summary['availability']
        fig.text(.5, .01,
                 f'16k: top two antigens supply {availability["selected_top_two_antigen_fraction"]:.1%}; '
                 f'maximum shuffle retains {availability["shuffle_correct_antigen_fraction"]:.1%} correct antigens. Held-outs remain fixed.',
                 ha='center', fontsize=9)
    elif 8000 in sizes:
        fig.text(.5, .005, '8k relaxes the antigen cap: two antigens supply 33% of records; the 4k prefix and held-outs are unchanged.',
                 ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(root / 'asd_scaling.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_scaling.yaml')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    report(args.config, args.require_complete, args.status_only)
