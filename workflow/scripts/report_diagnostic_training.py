"""Verify and summarize the small controlled antigen-conditioning experiments."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antigen_diagnostics import atomic_json, balanced_records, read_csv, sha256
from chain_examples import example_chains
from model_common import comparison_cohort
from summarize_antigen_diagnostics import REGIONS, paired_difference, scored_examples


def csv_table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for r in rows for key in r}))
        writer.writeheader()
        writer.writerows(rows)


def verify_stage(study, arm, phase):
    folder = Path(study['experiment_root']) / arm / phase
    metadata = json.loads((folder / 'metadata.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    spec = study['arms'][arm]
    records = Path(study['cohort_dir']) / ('asd_records.csv' if phase == 'asd' else 'sabdab2_records.csv')
    cohort = comparison_cohort(records)
    if (metadata['spec'] != spec or metadata['cohort_sha256'] != cohort or metadata['records_sha256'] != sha256(records) or
            metadata['epochs'] != 1 or metrics['epoch'] != 0 or metadata['seed'] != study['seed'] or
            metadata['optimizer_loaded'] or metadata['training_loss'] != 'cross_entropy' or
            metadata['lr'] != study['lr'] or metadata['loss_reduction'] != 'target_token_mean_including_eos' or
            metadata['target_chains'] != ['heavy', 'light']):
        raise ValueError('Diagnostic training settings/provenance differ from predeclared configuration')
    expected = Counter((r['id'], chain) for r in read_csv(records) if r['split'] == 'train'
                       for chain in example_chains(r))
    actual = Counter((r['record_id'], r['chain_type']) for r in read_csv(folder / 'training_curve.csv') if r['phase'] == 'train')
    if actual != expected or metrics['iterations'] != sum(expected.values()):
        raise ValueError('Diagnostic arm did not train every allowed chain target exactly once')
    if json.loads((folder / 'training_runtime.json').read_text())['status'] != 'complete':
        raise ValueError('Diagnostic training did not complete')
    if phase == 'asd':
        if metadata['source_checkpoint'] is not None:
            raise ValueError('ASD diagnostic pretraining must start fresh')
    else:
        if spec['source'] == 'prototype_asd':
            expected_source = Path(study['experiment_root']) / arm / 'asd/best.pt'
        elif spec['source'] == 'cross_finetuned':
            expected_source = Path(study['mango_root']) / 'sabdab2_finetuned/cross_entropy/cross_attention/checkpoints/best.pt'
        else:
            expected_source = Path(study['iglm_root']) / 'sabdab2_finetuned/checkpoints/best.pt'
        if (Path(metadata['source_checkpoint']).resolve() != expected_source.resolve() or
                metadata['source_checkpoint_sha256'] != sha256(expected_source)):
            raise ValueError('Diagnostic arm loaded the wrong source checkpoint')
    if spec.get('shuffle_training_antigens'):
        donors = json.loads((folder / 'training_antigen_donors.json').read_text())
        by_id = {r['id']: r for r in read_csv(records)}
        for target_id, donor_id in donors.items():
            target, donor = by_id[target_id], by_id[donor_id]
            if (target['split'] != 'train' or donor['split'] != 'train' or
                    target['resolved_ag_seq'] == donor['resolved_ag_seq'] or
                    target['ab_ag_cluster'] == donor['ab_ag_cluster']):
                raise ValueError('Shuffled training donor does not change antigen/group within train')
    return dict(arm=arm, phase=phase, passed=True, **metrics)


def report(config, require_complete=False):
    study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root'])
    root.mkdir(parents=True, exist_ok=True)
    parent = root.parent
    stages = []
    completed = []
    for arm, spec in study['arms'].items():
        phases = ['asd', 'native'] if spec['source'] == 'prototype_asd' else ['native']
        for phase in phases:
            folder = root / arm / phase
            required = ['best.pt', 'metadata.json', 'metrics.json', 'training_curve.csv', 'training_curve.png', 'training_runtime.json']
            if phase == 'native':
                required += ['complete.json', 'token_scores.jsonl']
            missing = [name for name in required if not (folder / name).is_file()]
            if missing:
                stages.append(dict(arm=arm, phase=phase, passed=False, missing=missing))
                continue
            try:
                stages.append(verify_stage(study, arm, phase))
                if phase == 'native':
                    completed.append(arm)
            except (KeyError, ValueError) as error:
                stages.append(dict(arm=arm, phase=phase, passed=False, error=str(error)))
    status = dict(complete=len(completed) == len(study['arms']) and all(s['passed'] for s in stages),
                  completed_final_arms=len(completed), expected_final_arms=len(study['arms']), stages=stages)
    atomic_json(root / 'results_report.json', status)
    if not status['complete']:
        print(json.dumps(status, indent=2))
        if require_complete:
            raise SystemExit(1)
        return status
    numbered = json.loads((parent / 'regions.json').read_text())
    native_path = Path(study['cohort_dir']) / 'sabdab2_records.csv'
    if numbered['records_sha256'] != sha256(native_path):
        raise ValueError('Region annotations do not match native scoring cohort')
    native_rows = read_csv(native_path)
    selected_train = balanced_records([r for r in native_rows if r['split'] == 'train'],
                                     study['train_probe_records'], study['seed'])
    expected_members = selected_train + [r for r in native_rows if r['split'] != 'train']
    expected_targets = {(r['id'], chain): (r, len(r['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']) + 1)
                        for r in expected_members for chain in example_chains(r)}
    primary = {r['id'] for r in read_csv(Path(study['cohort_dir']) / 'sabdab2_test_coverage.csv')
               if r['cath_assigned_all_antigen_chains'] == 'True'}
    values = []
    for name in ('prefix_pretrained', 'cross_pretrained', 'iglm_pretrained'):
        folder = parent / 'probes' / name
        marker = json.loads((folder / 'complete.json').read_text())
        if sha256(folder / 'token_scores.jsonl') != marker['token_scores_sha256']:
            raise ValueError('Original per-token diagnostic outputs changed')
        values.extend(scored_examples(folder / 'token_scores.jsonl', numbered, primary))
    for arm in completed:
        folder = root / arm / 'native'
        marker = json.loads((folder / 'complete.json').read_text())
        if (marker['scores_sha256'] != sha256(folder / 'token_scores.jsonl') or
                marker['checkpoint_sha256'] != sha256(folder / 'best.pt')):
            raise ValueError('Trained diagnostic weights or scores changed')
        scored = scored_examples(folder / 'token_scores.jsonl', numbered, primary)
        variants = ['correct'] if study['arms'][arm]['kind'] == 'iglm' else ['correct', 'wrong_0', 'no_context']
        for variant in variants:
            selected = [r for r in scored if r['variant'] == variant]
            actual_keys = [(r['id'], r['chain']) for r in selected]
            if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(expected_targets):
                raise ValueError('Experimental evaluation target membership differs')
            for row in selected:
                expected_row, tokens = expected_targets[row['id'], row['chain']]
                if row['all_tokens'] != tokens or row['cluster'] != expected_row['ab_ag_cluster'] or row['split'] != expected_row['split']:
                    raise ValueError('Experimental evaluation token count or group differs')
        validation = [r for r in scored if r['variant'] == 'correct' and r['split'] == 'val']
        val_nll = sum(r['all_sum'] for r in validation) / sum(r['all_tokens'] for r in validation)
        metrics = json.loads((folder / 'metrics.json').read_text())
        if abs(val_nll - metrics['val_nll']) > 2e-6:
            raise ValueError('Scored checkpoint does not reproduce selected validation NLL')
        values.extend(scored)
        if study['arms'][arm]['kind'] == 'iglm_cross':
            original_eval = json.loads((Path(study['iglm_root']) / 'sabdab2_finetuned/eval.json').read_text())
            stage_info = next(s for s in stages if s['arm'] == arm and s['phase'] == 'native')
            if abs(stage_info['initial_val_nll'] - original_eval['splits']['val']['nll']) > 2e-6:
                raise ValueError('Zero-output IgLM antigen branch did not start at exactly baseline NLL')
        if study['arms'][arm].get('freeze_decoder'):
            source = torch.load(Path(study['iglm_root']) / 'sabdab2_finetuned/checkpoints/best.pt',
                                map_location='cpu', weights_only=False)
            saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            if any(not torch.equal(saved['weights']['lm'][key], tensor) for key, tensor in source['lm'].items()):
                raise ValueError('Frozen IgLM decoder changed during adaptation')
            status.setdefault('frozen_decoder_verified', []).append(arm)
    csv_table(root / 'per_example.csv', values)
    scopes = dict(train_probe=lambda r: r['split'] == 'train', val=lambda r: r['split'] == 'val',
                  test_primary=lambda r: r['split'] == 'test' and r['primary'], test_all=lambda r: r['split'] == 'test')
    scores, effects = [], []
    for scope, includes in scopes.items():
        rows = [r for r in values if includes(r)]
        runs = sorted({(r['model'], r['variant']) for r in rows})
        for model, variant in runs:
            members = [r for r in rows if (r['model'], r['variant']) == (model, variant)]
            for region in REGIONS:
                tokens = sum(r[f'{region}_tokens'] for r in members)
                if tokens:
                    scores.append(dict(model=model, variant=variant, scope=scope, region=region,
                        nll=sum(r[f'{region}_sum'] for r in members) / tokens, n_tokens=tokens, n_examples=len(members)))
        comparisons = []
        for arm in completed:
            reference = 'iglm_pretrained' if study['arms'][arm]['source'] == 'prototype_asd' else 'iglm_continue'
            if arm != reference:
                comparisons.append((arm, 'correct', reference, 'correct'))
            if study['arms'][arm]['kind'] != 'iglm':
                comparisons += [(arm, variant, arm, 'correct') for variant in ('wrong_0', 'no_context')]
        comparisons += [('prefix_rebased', 'correct', 'prefix_pretrained', 'correct'),
                        ('cross_pooled', 'correct', 'cross_pretrained', 'correct'),
                        ('cross_continue', 'correct', 'cross_pretrained', 'correct'),
                        ('iglm_continue', 'correct', 'iglm_pretrained', 'correct'),
                        ('iglm_cross_frozen', 'correct', 'iglm_cross_frozen_shuffled', 'correct'),
                        ('prefix_rebased', 'correct', 'prefix_causal_control', 'correct'),
                        ('prefix_causal_control', 'correct', 'prefix_pretrained', 'correct')]
        for model, variant, reference, ref_variant in comparisons:
            left = [r for r in rows if (r['model'], r['variant']) == (model, variant)]
            right = [r for r in rows if (r['model'], r['variant']) == (reference, ref_variant)]
            for region in ('all', 'framework', 'cdr', 'cdr3'):
                estimate = paired_difference(left, right, region, 2000, study['seed'])
                if estimate:
                    effects.append(dict(scope=scope, region=region, model=model, variant=variant,
                                        reference=reference, reference_variant=ref_variant, **estimate))
    csv_table(root / 'scores.csv', scores)
    csv_table(root / 'paired_comparisons.csv', effects)
    status.update(scoring_verified=True, cohort_sha256=comparison_cohort(Path(study['cohort_dir']) / 'sabdab2_records.csv'))
    atomic_json(root / 'results_report.json', status)
    atomic_json(root / 'summary.json', dict(status=status, scores=scores, comparisons=effects,
                caution='Exploratory single-seed experiments; test cluster bootstrap does not measure training-seed uncertainty. Additional-epoch arms are compared to the additional-epoch IgLM control.'))
    draw(root, scores)
    print(json.dumps(status, indent=2))
    return status


def draw(root, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = ['iglm_pretrained', 'cross_pretrained', 'cross_pooled', 'prefix_pretrained', 'prefix_causal_control', 'prefix_rebased',
              'iglm_continue', 'cross_continue', 'iglm_cross_frozen', 'iglm_cross_frozen_shuffled', 'iglm_cross_joint']
    labels = ['IgLM', 'Cross', 'Cross pooled', 'Prefix', 'Prefix explicit mask', 'Prefix rebased',
              'IgLM +1 epoch', 'Cross +1 epoch', 'IgLM frozen + antigen', 'IgLM frozen + shuffled', 'IgLM joint + antigen']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, region, title in zip(axes, ['all', 'cdr3'], ['All residues + EOS', 'CDR3 residues']):
        y = [next(r['nll'] for r in scores if r['scope'] == 'test_primary' and r['region'] == region and
                  r['model'] == model and r['variant'] == 'correct') for model in models]
        colors = ['#4C78A8'] * 6 + ['#F58518'] * 5
        bars = ax.barh(labels, y, color=colors)
        for bar, value in zip(bars, y):
            ax.text(value + .01, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', va='center', fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Primary-test NLL (nats/token); lower is better')
        ax.set_xlim(0, max(y) * 1.15)
        ax.grid(axis='x', alpha=.2)
        ax.axhline(5.5, color='gray', linewidth=1)
    fig.suptitle('Small controlled follow-ups: blue = 1 ASD + 1 native epoch; orange = 1 additional native epoch')
    fig.tight_layout()
    fig.savefig(root / 'training_diagnostics.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/antigen_diagnostic_training.yaml')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report(args.config, args.require_complete)
