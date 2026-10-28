"""Report progress and independently verify the matched IgLM pilot controls."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from chain_examples import example_chains
from verify_asd_loss_comparison import fingerprint

PHASES = ('asd', 'sabdab2_scratch', 'sabdab2_finetuned')


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_stage(directory, phase, study, cohorts, primary):
    model = json.loads((directory / 'model_config.json').read_text())
    value = json.loads((directory / 'eval.json').read_text())
    history = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line]
    rows = cohorts['asd' if phase == 'asd' else 'sabdab2']
    digest = fingerprint(rows)
    if (len(history) != 1 or history[0]['epoch'] != 0 or value['checkpoint_epoch'] != 0 or
            abs(value['checkpoint_val_loss'] - history[0]['val_loss']) > 1e-10 or
            json.loads((directory / 'training_runtime.json').read_text())['status'] != 'complete'):
        raise ValueError('One completed epoch and validation-NLL checkpoint selection required')
    for item in (model, value):
        if (item['cohort_sha256'] != digest or item['training_settings'] != study['model'] or
                item['training_loss'] != {'name': 'cross_entropy'} or
                item['checkpoint_selection'] != 'validation_nll' or
                item['loss_reduction'] != 'target_token_mean_including_eos' or
                item['model_family'] != 'IgLM' or item['antigen_conditioning'] or
                item['antibody_prompt_used'] or item['public_pretrained_weights_loaded'] or
                item['target_chains'] != ['heavy', 'light']):
            raise ValueError('IgLM cohort, optimization or scoring protocol differs')
        initial = item.get('transfer_initialization')
        if phase == 'sabdab2_finetuned':
            source = Path(study['experiment_root']) / 'asd/checkpoints/best.pt'
            if (not initial or Path(initial['path']).resolve() != source.resolve() or
                    initial['sha256'] != sha256(source) or initial['source_epoch'] != 0 or
                    initial['source_cohort_sha256'] != fingerprint(cohorts['asd']) or
                    initial['optimizer_loaded'] or item['initialization'] != 'asd_checkpoint' or
                    item['pretrained_weights_loaded'] is not True):
                raise ValueError('IgLM initialization is not the matching prototype ASD checkpoint')
        elif (initial or item['initialization'] != 'random_from_scratch' or
              item['pretrained_weights_loaded']):
            raise ValueError('IgLM scratch/pretraining unexpectedly loaded weights')
    expected_training = Counter((r['id'], chain) for r in rows if r['split'] == 'train'
                                for chain in example_chains(r))
    actual_training = Counter((r['record_id'], r['chain_type']) for r in read_csv(directory / 'training_curve.csv')
                              if r['phase'] == 'train')
    if actual_training != expected_training or history[0]['iteration'] != sum(expected_training.values()):
        raise ValueError('IgLM did not train on exactly the same masked chain examples once')
    subsets = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val', 'test')}
    if phase != 'asd':
        subsets['test_cath_assigned'] = [r for r in subsets['test'] if r['id'] in primary]
    for split, members in subsets.items():
        score = value['splits'][split]
        total_tokens = total_examples = 0
        weighted = 0
        for chain, column in [('heavy', 'resolved_H_seq'), ('light', 'resolved_L_seq')]:
            selected = [r for r in members if chain in example_chains(r)]
            tokens = sum(len(r[column]) + 1 for r in selected)
            chain_score = score['by_chain'][chain]
            if chain_score['n_examples'] != len(selected) or chain_score['n_tokens'] != tokens:
                raise ValueError('IgLM per-chain evaluation membership differs')
            total_tokens += tokens
            total_examples += len(selected)
            weighted += chain_score['nll'] * tokens
        if (score['n_tokens'] != total_tokens or score['n_examples'] != total_examples or
                score['n_skipped_missing_embeddings'] or
                abs(weighted / total_tokens - score['nll']) > 1e-10 or
                abs(math.exp(score['nll']) - score['perplexity']) > 1e-8):
            raise ValueError('IgLM pooled NLL or membership differs')
    return value


def report(root, mango_root, cohort):
    root, mango_root, cohort = Path(root), Path(mango_root), Path(cohort)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'experiment_manifest.json'
    study = json.loads(manifest.read_text())['configuration'] if manifest.is_file() else None
    cohorts = {dataset: read_csv(cohort / f'{dataset}_records.csv') for dataset in ('asd', 'sabdab2')}
    primary = {r['id'] for r in read_csv(cohort / 'sabdab2_test_coverage.csv')
               if r['cath_assigned_all_antigen_chains'] == 'True'}
    if study:
        if study['mango_manifest_sha256'] != sha256(mango_root / 'experiment_manifest.json'):
            raise ValueError('MANGO experiment manifest changed')
        audit = json.loads((cohort / 'split_audit.json').read_text())
        for dataset in cohorts:
            path = cohort / f'{dataset}_records.csv'
            if sha256(path) != audit['sha256'][str(path)]:
                raise ValueError('Audited cohort changed')
    stages, table, evaluations = [], [], {}
    for phase in PHASES:
        directory = root / phase
        required = ('checkpoints/best.pt', 'checkpoints/latest.pt', 'model_config.json', 'config.json',
                    'metrics.jsonl', 'training_curve.csv', 'training_curve.png', 'training_runtime.json', 'eval.json')
        missing = [str(directory / name) for name in required if not (directory / name).is_file()]
        stage = dict(phase=phase, complete=False, missing=missing)
        if not missing:
            try:
                value = verify_stage(directory, phase, study, cohorts, primary)
                stage['complete'] = True
                evaluations[phase] = value
                for split, metric in value['splits'].items():
                    table.append(dict(phase=phase, split=split, nll=metric['nll'],
                                      perplexity=metric['perplexity'], n_examples=metric['n_examples'],
                                      n_tokens=metric['n_tokens']))
            except (ValueError, KeyError, TypeError) as error:
                stage['error'] = str(error)
        stages.append(stage)
    figure = mango_root / 'plots/sabdab2_loss_comparison_with_iglm.png'
    plot_table = figure.with_suffix('.csv')
    plot_verified = False
    plot_error = None
    if len(evaluations) == 3 and figure.is_file() and plot_table.is_file():
        try:
            plotted = read_csv(plot_table)
            expected_keys = {(phase, objective, architecture, split)
                             for phase in PHASES[1:] for objective, architecture in
                             [('cross_entropy', 'mango'), ('cross_entropy', 'cross_attention'),
                              ('focal', 'mango'), ('focal', 'cross_attention'), ('cross_entropy', 'iglm')]
                             for split in ('train', 'val', 'test_cath_assigned', 'test')}
            keys = [(r['phase'], r['objective'], r['architecture'], r['split']) for r in plotted]
            if len(keys) != 40 or set(keys) != expected_keys:
                raise ValueError('Combined plot must contain all ten models and four splits')
            for row in plotted:
                if row['architecture'] == 'iglm':
                    value = evaluations[row['phase']]
                else:
                    path = mango_root / row['phase'] / row['objective'] / row['architecture'] / 'eval.json'
                    value = json.loads(path.read_text())
                score = value['splits'][row['split']]
                if (abs(float(row['nll']) - score['nll']) > 1e-10 or
                        row['cohort_sha256'] != fingerprint(cohorts['sabdab2']) or
                        int(row['n_tokens']) != score['n_tokens'] or
                        int(row['n_examples']) != score['n_examples']):
                    raise ValueError('Combined plot differs from evaluation')
            plot_verified = True
        except (ValueError, KeyError) as error:
            plot_error = str(error)
    complete = all(stage['complete'] for stage in stages) and plot_verified
    result = dict(complete=complete, training_stages=stages,
                  completed_stages=sum(stage['complete'] for stage in stages),
                  completed_final_models=sum(stage['complete'] for stage in stages[1:]),
                  combined_plot=str(figure), combined_plot_verified=plot_verified,
                  combined_plot_error=plot_error, original_test_records=len([r for r in cohorts['sabdab2'] if r['split'] == 'test']),
                  primary_test_records=len(primary))
    (root / 'results_report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    with (root / 'results_report.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['phase', 'split', 'nll', 'perplexity', 'n_examples', 'n_tokens'])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--mango-root', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = report(args.root, args.mango_root, args.cohort)
    if args.require_complete and not result['complete']:
        raise SystemExit(1)
