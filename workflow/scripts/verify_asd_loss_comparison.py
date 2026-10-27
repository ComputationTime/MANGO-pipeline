"""Independently check loss-arm provenance, evaluation counts and plotted NLL."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from chain_examples import example_chains
from prepare_abrank_transfer import assert_disjoint
from prepare_asd_transfer import read_csv, sha256


def fingerprint(rows):
    fields = ('id', 'split', 'ab_ag_cluster', 'resolved_H_seq', 'resolved_L_seq',
              'resolved_ag_seq', 'antigen_chains', 'target_contract')
    if rows and 'example_target_chains' in rows[0]:
        fields += ('example_target_chains',)
    values = [[r.get(k, '') for k in fields] for r in rows]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def verify(root):
    root = Path(root)
    study = json.loads((root / 'experiment_manifest.json').read_text())['configuration']
    cohort = Path(study['cohort_dir'])
    audit = json.loads((cohort / 'split_audit.json').read_text())
    for path, expected in audit['sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Pinned input/output changed: {path}')
    snapshot = json.loads((cohort / 'input_manifest.json').read_text())
    for path, expected in snapshot['protected_figures'].items():
        if sha256(path) != expected:
            raise ValueError(f'Protected original figure changed: {path}')
    data = {dataset: read_csv(cohort / f'{dataset}_records.csv') for dataset in ('asd', 'sabdab2')}
    primary = {r['id'] for r in read_csv(cohort / 'sabdab2_test_coverage.csv')
               if r['cath_assigned_all_antigen_chains'] == 'True'}
    protected = read_csv(cohort / 'protected_sabdab2_test.csv')
    if [{key: r[key] for key in protected[0]} for r in data['sabdab2'] if r['split'] == 'test'] != protected:
        raise ValueError('Protected native test membership changed')
    assert_disjoint(data['asd'], [r for r in data['sabdab2'] if r['split'] in {'val', 'test'}])
    for rows in data.values():
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert_disjoint([r for r in rows if r['split'] == left], [r for r in rows if r['split'] == right])
    if audit.get('prototype'):
        if {r['antigen_id'] for r in data['asd']} & set(audit['excluded_her2_antigen_ids']):
            raise ValueError('HER2 antigen present in prototype')
        if any(r['has_buzz'] == 'True' for r in data['asd']):
            raise ValueError('BUZZ record present in prototype')
        settings = audit['prototype_policy']
        for split in ('train', 'val', 'test'):
            selected = [r for r in data['asd'] if r['split'] == split]
            if len(selected) > settings[f'max_{split}_records']:
                raise ValueError('Prototype exceeds its split size limit')
            if any(len(r['resolved_ag_seq']) > settings['max_antigen_tokens'] for r in selected):
                raise ValueError('Prototype exceeds its antigen length limit')
            from collections import Counter
            if max(Counter(r['antigen_id'] for r in selected).values()) > settings['max_records_per_antigen']:
                raise ValueError('Prototype exceeds its per-antigen record cap')
        source_native = Path(settings['source_cohort']) / 'sabdab2_records.csv'
        if sha256(source_native) != sha256(cohort / 'sabdab2_records.csv'):
            raise ValueError('Prototype native comparison cohort changed')
    seen = set()
    for row in data['asd']:
        for chain in example_chains(row):
            key = (row['resolved_ag_seq'], chain, row['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq'])
            if key in seen:
                raise ValueError('Duplicate ASD individual-chain example')
            seen.add(key)
    findings = []
    for phase in ('asd', 'sabdab2_scratch', 'sabdab2_finetuned'):
        rows = data['asd' if phase == 'asd' else 'sabdab2']
        digest = fingerprint(rows)
        by_split = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val', 'test')}
        if phase != 'asd':
            by_split['test_cath_assigned'] = [r for r in by_split['test'] if r['id'] in primary]
        for loss, loss_spec in study['objectives'].items():
            for architecture, expected_architecture in study['architectures'].items():
                directory = root / phase / loss / architecture
                value = json.loads((directory / 'eval.json').read_text())
                model = json.loads((directory / 'model_config.json').read_text())
                history = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line]
                if json.loads((directory / 'training_runtime.json').read_text())['status'] != 'complete':
                    raise ValueError(f'Training did not complete: {directory}')
                if study['model']['epochs'] == 1 and (len(history) != 1 or history[0]['epoch'] != 0):
                    raise ValueError(f'One-epoch budget violated: {directory}')
                if value['training_loss'] != loss_spec or model['training_loss'] != loss_spec:
                    raise ValueError('Wrong training objective')
                if value['architecture'] != expected_architecture or value['embedding_tag'] != 'esm2':
                    raise ValueError('Wrong model/conditioning representation')
                if value['evaluation_loss'] != 'cross_entropy' or value['checkpoint_selection'] != 'validation_nll':
                    raise ValueError('Evaluation/selection must use ordinary NLL')
                if value['cohort_sha256'] != digest or model['cohort_sha256'] != digest:
                    raise ValueError('Wrong comparison cohort fingerprint')
                best = min(history, key=lambda r: r['val_nll'])
                if value['checkpoint_epoch'] != best['epoch'] or abs(value['checkpoint_val_loss'] - best['val_nll']) > 1e-10:
                    raise ValueError('Checkpoint selection differs from ordinary validation NLL')
                for split, members in by_split.items():
                    result = value['splits'][split]
                    total_tokens = total_examples = 0
                    weighted_nll = 0.0
                    for chain, column in [('heavy', 'resolved_H_seq'), ('light', 'resolved_L_seq')]:
                        selected = [r for r in members if chain in example_chains(r)]
                        tokens = sum(len(r[column]) + 1 for r in selected)
                        chain_result = result['by_chain'][chain]
                        if chain_result['n_examples'] != len(selected) or chain_result['n_tokens'] != tokens:
                            raise ValueError('Chain evaluation membership/token counts differ')
                        total_tokens += tokens
                        total_examples += len(selected)
                        weighted_nll += chain_result['nll'] * tokens
                    if result['n_tokens'] != total_tokens or result['n_examples'] != total_examples or result['n_skipped_missing_embeddings']:
                        raise ValueError('Pooled evaluation membership/token counts differ')
                    if abs(weighted_nll / total_tokens - result['nll']) > 1e-10 or abs(math.exp(result['nll']) - result['perplexity']) > 1e-8:
                        raise ValueError('NLL/perplexity aggregation differs')
                if phase == 'sabdab2_finetuned':
                    source = root / 'asd' / loss / architecture / 'checkpoints/best.pt'
                    initial = value['initialization']
                    if Path(initial['path']).resolve() != source.resolve() or initial['sha256'] != sha256(source) or initial['optimizer_loaded']:
                        raise ValueError('Fine-tuning initialization differs from matching prototype checkpoint')
                    if initial['source_cohort_sha256'] != fingerprint(data['asd']):
                        raise ValueError('Fine-tuning used a different ASD cohort')
                elif value['initialization'] is not None:
                    raise ValueError('Scratch/pretraining model unexpectedly initialized from a checkpoint')
                findings.append(dict(phase=phase, objective=loss, architecture=architecture, passed=True))
    plot_rows = read_csv(root / 'plots/sabdab2_loss_comparison.csv')
    if len(plot_rows) != 32:
        raise ValueError('Comparison plot must contain four splits for eight final models')
    for row in plot_rows:
        evaluation = root / row['phase'] / row['objective'] / row['architecture'] / 'eval.json'
        expected = json.loads(evaluation.read_text())['splits'][row['split']]['nll']
        if abs(float(row['nll']) - expected) > 1e-10:
            raise ValueError('Plotted NLL differs from evaluation')
    result = dict(status='passed', training_stages_verified=len(findings), final_models_verified=8,
                  one_epoch_per_stage=study['model']['epochs'] == 1,
                  original_test_records=len(protected), primary_test_records=len(primary),
                  her2_free_asd=bool(audit.get('prototype')), ordinary_nll_evaluation=True, checks=findings)
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    verify(parser.parse_args().root)
