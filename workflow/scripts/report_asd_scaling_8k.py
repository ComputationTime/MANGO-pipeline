"""Join verified frozen ASD scaling references with a new size/architecture study."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import yaml

from antigen_diagnostics import atomic_json, read_csv, sha256
from asd_scaling_studies import studies_in_order
from chain_examples import example_chains
from prepare_asd_scaling import paths
from prepare_asd_scaling_8k import assert_nested
from report_asd_scaling import average_rows, difference_rows, draw, verify_run
from summarize_antigen_diagnostics import REGIONS, paired_difference, scored_examples


def verify_training_shuffle(study, size):
    root = Path(study['experiment_root']) / f'n{size}'
    rows = read_csv(paths(study, size)['records'])
    train = [r for r in rows if r['split'] == 'train']
    by_id = {r['id']: r for r in train}
    marker = json.loads((root / 'embeddings_complete.json').read_text())
    mapping_path = root / 'shuffled/antigen/esm2_shuffled/mapping.csv'
    if marker['shuffled_mapping_sha256'] != sha256(mapping_path):
        raise ValueError('Training shuffle map changed')
    mapping = read_csv(mapping_path)
    if Counter(r['id'] for r in mapping) != Counter(r['id'] for r in rows):
        raise ValueError('Training shuffle target membership changed')
    lookup = {r['id']: r for r in mapping}
    donors = [by_id[lookup[r['id']]['donor_id']] for r in train]
    if Counter(r['id'] for r in donors) != Counter(r['id'] for r in train):
        raise ValueError('Training shuffle did not preserve record marginals')
    if any(lookup[r['id']]['donor_id'] != r['id'] for r in rows if r['split'] != 'train'):
        raise ValueError('Training shuffle changed held-out context')
    largest = max(Counter(r['resolved_ag_seq'] for r in train).values())
    same = sum(r['resolved_ag_seq'] == d['resolved_ag_seq'] for r, d in zip(train, donors))
    if same != max(0, 2 * largest - len(train)) or same != marker.get('shuffled_correct_antigen_records', 0):
        raise ValueError('Training shuffle retained more correct antigens than claimed or mathematically necessary')
    if same and not study.get('allow_partial_training_shuffle'):
        raise ValueError('A strict training shuffle retained correct antigens')
    return dict(size=size, correct_antigen_records=same, correct_antigen_fraction=same / len(train),
                preserved_record_marginals=True, maximum_identity_changes_verified=True)


def load_scores(study, numbered, heldout):
    root = Path(study['experiment_root'])
    by_id = {r['id']: r for r in heldout}
    expected = {(r['id'], c): (r['split'], r['ab_ag_cluster'],
                len(r['resolved_H_seq' if c == 'heavy' else 'resolved_L_seq']) + 1)
                for r in heldout for c in example_chains(r)}
    test_ids = {r['id'] for r in heldout if r['split'] == 'test'}
    values = []
    for run, spec in study['runs'].items():
        folder = root / 'runs' / run
        rows = scored_examples(folder / 'token_scores.jsonl', numbered, test_ids)
        variants = ['correct'] if spec['family'] == 'iglm' else ['correct', 'wrong_0', 'wrong_1', 'no_context']
        def validate(members, variant):
            keys = [(r['id'], r['chain']) for r in members]
            if len(keys) != len(set(keys)) or set(keys) != set(expected):
                raise ValueError(f'{run}/{variant}: held-out target membership differs')
            for row in members:
                if (row['model'] != run or
                        (row['split'], row['cluster'], row['all_tokens']) != expected[row['id'], row['chain']]):
                    raise ValueError(f'{run}/{variant}: model, tokens or split groups differ')
        for variant in variants:
            validate([r for r in rows if r['variant'] == variant], variant)
        if set(r['variant'] for r in rows) != set(variants):
            raise ValueError('Unexpected scoring variants')
        values.extend(rows)
        if spec['family'] == 'iglm':
            continue
        values.extend(average_rows([r for r in rows if r['variant'] == 'wrong_0'],
                                  [r for r in rows if r['variant'] == 'wrong_1'], 'wrong_mean'))
        far_path = folder / 'far_token_scores.jsonl'
        marker = json.loads((folder / 'far_eval.json').read_text())
        ready = json.loads((folder / 'train_complete.json').read_text())
        if (marker['checkpoint_sha256'] != ready['checkpoint_sha256'] or
                marker['heldout_sha256'] != sha256(root / 'evaluation_records.csv') or
                marker['token_scores_sha256'] != sha256(far_path)):
            raise ValueError('Different-family probe provenance changed')
        far = scored_examples(far_path, numbered, test_ids)
        if set(r['variant'] for r in far) != {'far_wrong_0', 'far_wrong_1'}:
            raise ValueError('Unexpected different-family probe variants')
        for variant in ('far_wrong_0', 'far_wrong_1'):
            members = [r for r in far if r['variant'] == variant]
            validate(members, variant)
            for row in members:
                target, donor = by_id[row['id']], by_id[row['donor_id']]
                if (target['split'] != donor['split'] or target['ab_ag_cluster'] == donor['ab_ag_cluster'] or
                        set(target['cath_families'].split(';')) & set(donor['cath_families'].split(';'))):
                    raise ValueError('Different-family donor shares a family/group or crosses splits')
        values.extend(far)
        values.extend(average_rows([r for r in far if r['variant'] == 'far_wrong_0'],
                                  [r for r in far if r['variant'] == 'far_wrong_1'], 'far_wrong_mean'))
    return values


def report(config, require_complete=False, status_only=False):
    study = yaml.safe_load(Path(config).read_text())
    reference = yaml.safe_load(Path(study['reference_config']).read_text())
    root = Path(study['experiment_root'])
    studies = studies_in_order(study)
    stages = []
    for current in studies:
        for run, spec in current['runs'].items():
            try:
                stages.append(dict(verify_run(current, run, spec), source_root=current['experiment_root']))
            except (ValueError, KeyError, FileNotFoundError) as error:
                stages.append(dict(run=run, passed=False, error=str(error)))
    complete = all(s['passed'] for s in stages)
    status = dict(complete=False, models_complete=complete, completed_models=sum(s['passed'] for s in stages),
                  expected_models=len(stages), runs=stages,
                  completed_new_models=sum(s['passed'] for s in stages if s['run'] in study['runs']),
                  expected_new_models=len(study['runs']),
                  analysis_complete=(root / 'summary.json').is_file() and (root / 'asd_scaling.png').is_file())
    if status_only or not complete or not (root / 'regions.json').is_file():
        status['complete'] = complete and status['analysis_complete']
        atomic_json(root / 'results_report.json', status)
        print(json.dumps(status, indent=2))
        if require_complete and not status['complete']:
            raise SystemExit(1)
        return status
    heldout_path = root / 'evaluation_records.csv'
    heldout = read_csv(heldout_path)
    numbered = json.loads((root / 'regions.json').read_text())
    if numbered['records_sha256'] != sha256(heldout_path):
        raise ValueError('Region annotations differ from held-out cohort')
    larger = [r for r in read_csv(paths(study, max(study['sizes']))['records']) if r['split'] == 'train']
    values, cohorts = [], {}
    for current in studies:
        current_root = Path(current['experiment_root'])
        manifest = json.loads((current_root / 'experiment_manifest.json').read_text())
        if manifest['configuration'] != current or sha256(current_root / 'evaluation_records.csv') != sha256(heldout_path):
            raise ValueError('Reference configuration or fixed held-out membership changed')
        audit = json.loads((current_root / 'cohort_audit.json').read_text())
        for size in current['sizes']:
            assert_nested([r for r in read_csv(paths(current, size)['records']) if r['split'] == 'train'], larger)
        cohorts.update(audit['training'])
        values.extend(load_scores(current, numbered, heldout))
    sizes = sorted(map(int, cohorts))
    scores, contrasts = [], []
    for split in ('val', 'test'):
        selected = [r for r in values if r['split'] == split]
        get = lambda model, variant='correct': [r for r in selected if r['model'] == model and r['variant'] == variant]
        for model, variant in sorted({(r['model'], r['variant']) for r in selected}):
            for chain in ('both', 'heavy', 'light'):
                members = [r for r in get(model, variant) if chain == 'both' or r['chain'] == chain]
                for region in REGIONS:
                    count = sum(r[f'{region}_tokens'] for r in members)
                    if count:
                        scores.append(dict(model=model, variant=variant, split=split, chain=chain, region=region,
                            nll=sum(r[f'{region}_sum'] for r in members) / count, n_tokens=count,
                            n_examples=len(members), n_clusters=len({r['cluster'] for r in members})))
        def compare(label, left, right):
            for region in ('all', 'framework', 'cdr', 'cdr3'):
                result = paired_difference(left, right, region, study['bootstrap_repeats'], study['seed'])
                if result:
                    contrasts.append(dict(contrast=label, split=split, region=region, **result))
        for size in sizes:
            compare(f'cross_minus_iglm_n{size}', get(f'n{size}_cross'), get(f'n{size}_iglm'))
            compare(f'paired_minus_shuffled_training_n{size}', get(f'n{size}_cross'), get(f'n{size}_shuffled'))
            if get(f'n{size}_mango'):
                compare(f'mango_minus_iglm_n{size}', get(f'n{size}_mango'), get(f'n{size}_iglm'))
                compare(f'cross_minus_mango_n{size}', get(f'n{size}_cross'), get(f'n{size}_mango'))
            if get(f'n{size}_fixed'):
                compare(f'fixed_minus_iglm_n{size}', get(f'n{size}_fixed'), get(f'n{size}_iglm'))
                compare(f'fixed_minus_mango_n{size}', get(f'n{size}_fixed'), get(f'n{size}_mango'))
            for family in ('mango', 'fixed', 'cross', 'shuffled'):
                if not get(f'n{size}_{family}'):
                    continue
                for variant in ('wrong_mean', 'far_wrong_mean', 'no_context'):
                    compare(f'{variant}_minus_correct_n{size}_{family}',
                            get(f'n{size}_{family}', variant), get(f'n{size}_{family}'))
        for small, large in zip(sizes, sizes[1:]):
            for family in ('iglm', 'mango', 'fixed', 'cross', 'shuffled'):
                if get(f'n{large}_{family}') and get(f'n{small}_{family}'):
                    compare(f'scale_n{large}_minus_n{small}_{family}', get(f'n{large}_{family}'), get(f'n{small}_{family}'))
            compare(f'change_in_cross_iglm_gap_n{large}_vs_n{small}',
                    difference_rows(get(f'n{large}_cross'), get(f'n{large}_iglm')),
                    difference_rows(get(f'n{small}_cross'), get(f'n{small}_iglm')))
            compare(f'change_in_far_antigen_benefit_n{large}_vs_n{small}',
                    difference_rows(get(f'n{large}_cross', 'far_wrong_mean'), get(f'n{large}_cross')),
                    difference_rows(get(f'n{small}_cross', 'far_wrong_mean'), get(f'n{small}_cross')))
    # Keep the established plot's contrast names for the same within-size probes.
    for effect in contrasts:
        effect['contrast'] = effect['contrast'].replace('far_wrong_mean_minus_correct_', 'far_wrong_minus_correct_')
    availability = json.loads((root / 'availability.json').read_text())
    reference_hashes = availability.get('reference_sha256', {})
    if not reference_hashes:
        reference_hashes = {str(Path(reference['experiment_root']) / 'experiment_manifest.json'):
                            availability['reference_manifest_sha256']}
    for path, expected_hash in reference_hashes.items():
        if sha256(path) != expected_hash:
            raise ValueError('Reference input changed after scaling preparation')
    shuffle_audits = [verify_training_shuffle(study, size) for size in study['sizes']]
    largest_shuffle = next(a for a in shuffle_audits if a['size'] == max(study['sizes']))
    if largest_shuffle['correct_antigen_fraction'] != availability.get('shuffle_correct_antigen_fraction', 0):
        raise ValueError('Availability audit differs from actual training shuffle')
    status.update(complete=True, analysis_complete=True, fixed_heldout_verified=True,
                  target_counts_verified=True, nested_training_verified=True, training_shuffles_verified=True,
                  test_groups=len({r['ab_ag_cluster'] for r in heldout if r['split'] == 'test'}))
    summary = dict(status=status, cohorts=cohorts, scores=scores, contrasts=contrasts, availability=availability,
                   training_shuffles=shuffle_audits,
                   interpretation=availability['note'], region_status=numbered['status_counts'])
    for name, rows in [('scores.csv', scores), ('paired_comparisons.csv', contrasts)]:
        with (root / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    draw(root, summary, sizes=sizes)
    write_findings(root, summary)
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'results_report.json', status)
    print(json.dumps(status, indent=2))
    return summary


def write_findings(root, summary):
    scores, effects = summary['scores'], summary['contrasts']
    sizes = sorted(map(int, summary['cohorts']))
    largest, previous = sizes[-1], sizes[-2]
    lines = ['# ASD-only scaling comparison', '',
             f'All {summary["status"]["completed_models"]} models completed, including {summary["status"]["completed_new_models"]} new runs.',
             'Random initialization, ordinary cross-entropy, frozen ESM2 650M for antigen models; no SAbDab2 fine-tuning.', '',
             '| Training records | Model | Test NLL | CDR3 NLL |', '|---:|---|---:|---:|']
    for size in sizes:
        for family, label in [('iglm', 'IgLM'), ('mango', 'Regular MANGO (prefix)'),
                              ('fixed', 'MANGO with fixed antibody positions'),
                              ('cross', 'Paired cross-attention'), ('shuffled', 'Shuffled-training cross-attention')]:
            rows = [r for r in scores if r['model'] == f'n{size}_{family}' and r['variant'] == 'correct'
                    and r['split'] == 'test' and r['chain'] == 'both']
            if not rows:
                continue
            nll = lambda region: next(r['nll'] for r in rows if r['region'] == region)
            lines.append(f'| {size:,} | {label} | {nll("all"):.6f} | {nll("cdr3"):.6f} |')
    lines += ['', '| Test contrast | NLL difference | 95% cluster bootstrap interval |', '|---|---:|---|']
    names = {f'cross_minus_iglm_n{largest}', f'mango_minus_iglm_n{largest}', f'cross_minus_mango_n{largest}',
             f'fixed_minus_iglm_n{largest}', f'fixed_minus_mango_n{largest}',
             f'paired_minus_shuffled_training_n{largest}', f'far_wrong_minus_correct_n{largest}_cross',
             f'change_in_cross_iglm_gap_n{largest}_vs_n{previous}',
             f'change_in_far_antigen_benefit_n{largest}_vs_n{previous}'}
    for row in effects:
        if row['contrast'] in names and row['split'] == 'test' and row['region'] == 'all':
            lo, hi = row['cluster_bootstrap_95ci']
            lines.append(f'| {row["contrast"]} | {row["delta_nll"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    a = summary['availability']
    lines += ['', 'Positive wrong-minus-correct differences indicate antigen benefit; negative cross-minus-IgLM differences favor MANGO.', '',
              'The held-out cohort is unchanged: 95 validation records and 200 test records (393 test chain targets, ten test groups).',
              'Every earlier training prefix and all individual chain masks are preserved. HER2/BUZZ and family protection remain enforced.',
              f'The 768-residue pool contains {a["length_eligible_train_records"]:,} records, '
              f'{a["length_eligible_unique_train_records"]:,} after individual-chain deduplication. '
              f'The old ten-record cap permits only {a["ten_record_cap_unique_train_records"]:,}.',
              f'The selected maximum is {a["selected_max_records_per_antigen"]:,} records per antigen. '
              f'Two antigens now supply {a["selected_top_two_antigen_fraction"]:.2%} of the {largest:,}-record cohort. '
              'The antigen set is unchanged. This expansion increases both examples and antigen imbalance.',
              'All runs use one initialization seed. Intervals resample entire test split groups; they do not include training-seed uncertainty.',
              'One epoch at larger sizes also increases optimizer updates. Existing 2k/two-epoch controls remain in the data tables. The figure shows only overall and CDR3 NLL.',
              f'ESM2 uses the shared tensor cache; see n{largest}/embeddings_complete.json for encoder forward counts.', '']
    if a.get('shuffle_correct_antigen_fraction', 0):
        lines += [f'The marginal-preserving {largest:,}-record shuffle retains '
                  f'{a["shuffle_minimum_correct_antigen_records"]:,} correct-antigen records '
                  f'({a["shuffle_correct_antigen_fraction"]:.2%}), the mathematical minimum imposed by the dominant antigen. '
                  'Earlier sizes have zero retained correct identities. Interpret this as a partially shuffled control.', '']
    (root / 'findings.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_scaling_8k.yaml')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    report(args.config, args.require_complete, args.status_only)
