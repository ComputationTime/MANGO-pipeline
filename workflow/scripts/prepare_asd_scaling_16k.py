"""Prepare nested 2k/4k/8k/16k cohorts and pin all frozen reference cohorts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from asd_scaling_studies import studies_in_order
from prepare_asd_scaling import paths, prepare_scaling
from prepare_asd_scaling_8k import assert_nested
from prepare_asd_transfer import read_csv, sha256


def prepare(config):
    import yaml
    study = yaml.safe_load(Path(config).read_text())
    studies = studies_in_order(study)
    for current in studies[:-1]:
        for key in ('max_antigen_tokens', 'seed', 'prototype_config', 'prototype_root'):
            if current[key] != study[key]:
                raise ValueError(f'Scaling extension must preserve {key}')
    result = prepare_scaling(config)
    root = Path(study['experiment_root'])
    larger = [r for r in read_csv(paths(study, max(study['sizes']))['records']) if r['split'] == 'train']
    references = {}
    for current in studies[:-1]:
        refroot = Path(current['experiment_root'])
        manifest_path = refroot / 'experiment_manifest.json'
        if json.loads(manifest_path.read_text())['configuration'] != current:
            raise ValueError('Frozen reference configuration changed')
        if sha256(refroot / 'evaluation_records.csv') != result['heldout_sha256']:
            raise ValueError('Frozen held-out records changed')
        references[str(manifest_path)] = sha256(manifest_path)
        for size in current['sizes']:
            location = paths(current, size)
            audit = json.loads(Path(location['audit']).read_text())
            if sha256(location['records']) != audit['sha256'][location['records']]:
                raise ValueError('Frozen reference cohort changed')
            assert_nested([r for r in read_csv(location['records']) if r['split'] == 'train'], larger)
            # All reused sizes must also match the newly prepared cohorts.
            assert read_csv(location['records']) == read_csv(paths(study, size)['records'])
            references[location['records']] = sha256(location['records'])
    previous_root = Path(studies[-2]['experiment_root'])
    old = json.loads((previous_root / 'availability.json').read_text())
    base = yaml.safe_load(Path(study['prototype_config']).read_text())
    source = Path(base['prototype']['source_cohort']) / 'asd_records.csv'
    if sha256(source) != old['source_sha256']:
        raise ValueError('Reference availability uses a different source')
    counts = Counter(r['resolved_ag_seq'] for r in larger)
    ids = Counter(r['antigen_id'] for r in larger)
    unchanged = max(0, 2 * max(counts.values()) - len(larger))
    availability = dict(old, selected_train_records=len(larger), selected_antigens=len(counts),
        selected_max_records_per_antigen=max(counts.values()),
        selected_top_two_antigen_fraction=sum(c for _, c in counts.most_common(2)) / len(larger),
        selected_top_antigens=[dict(antigen_id=a, records=c) for a, c in ids.most_common(10)],
        selected_per_antigen_cap=study['max_records_per_antigen'],
        reference_sizes=sorted({s for c in studies[:-1] for s in c['sizes']}),
        reference_sha256=references, preparation_sha256=sha256(__file__),
        shuffle_minimum_correct_antigen_records=unchanged,
        shuffle_correct_antigen_fraction=unchanged / len(larger),
        note='16k preserves all earlier training prefixes and target masks. Antigen frequencies become more concentrated; the maximum shuffle retains the mathematically unavoidable fraction of correct antigens.')
    path = root / 'availability.json'
    if path.is_file() and json.loads(path.read_text()) != availability:
        raise ValueError('Frozen scaling availability audit changed')
    path.write_text(json.dumps(availability, indent=2, sort_keys=True) + '\n')
    print(json.dumps(availability, indent=2), flush=True)
    return availability


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_scaling_16k.yaml')
    prepare(parser.parse_args().config)
