"""Extend the frozen ASD scale study, auditing available data and concentration."""
import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_asd_scaling import paths, prepare_scaling
from prepare_asd_prototype import diverse_subset, her2_antigens
from prepare_asd_transfer import read_csv, select_examples, sha256


def assert_nested(reference, larger):
    if larger[:len(reference)] != reference:
        raise ValueError('8k extension changed frozen training records or target masks')


def prepare(config):
    import yaml
    study = yaml.safe_load(Path(config).read_text())
    reference = yaml.safe_load(Path(study['reference_config']).read_text())
    reference_root = Path(reference['experiment_root'])
    frozen = json.loads((reference_root / 'experiment_manifest.json').read_text())
    if frozen['configuration'] != reference:
        raise ValueError('Reference scaling configuration changed')
    for key in ('max_antigen_tokens', 'seed', 'prototype_config', 'prototype_root'):
        if study[key] != reference[key]:
            raise ValueError(f'8k extension must preserve reference {key}')
    result = prepare_scaling(config)
    root = Path(study['experiment_root'])
    if sha256(root / 'evaluation_records.csv') != sha256(reference_root / 'evaluation_records.csv'):
        raise ValueError('8k extension changed held-out records')
    larger = [r for r in read_csv(paths(study, max(study['sizes']))['records']) if r['split'] == 'train']
    for size in reference['sizes']:
        location = paths(reference, size)
        audit = json.loads(Path(location['audit']).read_text())
        if sha256(location['records']) != audit['sha256'][location['records']]:
            raise ValueError('Frozen reference cohort changed')
        assert_nested([r for r in read_csv(location['records']) if r['split'] == 'train'], larger)
    base = yaml.safe_load(Path(study['prototype_config']).read_text())
    source = Path(base['prototype']['source_cohort']) / 'asd_records.csv'
    rows = read_csv(source)
    blocked = her2_antigens(base['prototype']['audit_database'])
    clean = [r for r in rows if r['antigen_id'] not in blocked and r.get('has_buzz') != 'True']
    train = [r for r in clean if r['split'] == 'train']
    eligible = [r for r in train if len(r['resolved_ag_seq']) <= study['max_antigen_tokens']]
    capacity = lambda members, cap: len(select_examples(diverse_subset(members, len(members), cap, study['seed'])))
    counts = Counter(r['antigen_id'] for r in larger)
    top = counts.most_common(10)
    availability = dict(
        audited_source_records_by_split=dict(Counter(r['split'] for r in rows)),
        her2_free_records_by_split=dict(Counter(r['split'] for r in clean)),
        her2_free_unique_train_records=capacity(train, len(train)),
        length_eligible_train_records=len(eligible),
        length_eligible_unique_train_records=capacity(eligible, len(eligible)),
        eligible_train_antigens=len({r['antigen_id'] for r in eligible}),
        ten_record_cap_unique_train_records=capacity(eligible, 10),
        selected_train_records=len(larger), selected_antigens=len(counts),
        selected_max_records_per_antigen=max(counts.values()),
        selected_top_two_antigen_fraction=sum(c for _, c in top[:2]) / len(larger),
        selected_top_antigens=[dict(antigen_id=a, records=c) for a, c in top],
        reference_sizes=reference['sizes'], reference_prefixes_verified=True,
        heldout_sha256=result['heldout_sha256'],
        source_sha256=sha256(source), reference_manifest_sha256=sha256(reference_root / 'experiment_manifest.json'),
        preparation_sha256=sha256(__file__),
        note='8k retains the exact 4k prefix and target masks. Raising the per-antigen cap changes antigen frequencies; this is not a balanced scale-only intervention.')
    path = root / 'availability.json'
    if path.is_file() and json.loads(path.read_text()) != availability:
        raise ValueError('Frozen 8k availability audit changed')
    path.write_text(json.dumps(availability, indent=2, sort_keys=True) + '\n')
    print(json.dumps(availability, indent=2), flush=True)
    return availability


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_scaling_8k.yaml')
    prepare(parser.parse_args().config)
