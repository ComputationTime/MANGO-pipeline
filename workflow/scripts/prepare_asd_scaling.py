"""Double the audited ASD prototype while freezing held-out data and exclusions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from chain_examples import example_chains
from prepare_asd_prototype import diverse_subset, her2_antigens
from prepare_asd_transfer import read_csv, select_examples, sha256
from prepare_abrank_transfer import assert_disjoint, family_set
from prepare_cath_sequences import write_csv


def record_derangement(rows, seed=0):
    """Preserve donor record marginals within split, changing antigen identity.

    A cyclic shift by the largest antigen block deranges sorted sequence blocks
    when no antigen occupies more than half the split. Different split-graph
    components cannot be required: one ASD training component contains >90% of
    records. This control changes pairings, never group/split membership.
    """
    result = {}
    splits = defaultdict(list)
    for row in rows:
        splits[row['split']].append(row)
    for split, members in splits.items():
        key = lambda r: (hashlib.sha256(f'{seed}:{r["resolved_ag_seq"]}'.encode()).digest(), r['id'])
        ordered = sorted(members, key=key)
        largest = max(Counter(r['resolved_ag_seq'] for r in ordered).values())
        if largest * 2 > len(ordered):
            raise ValueError(f'Cannot preserve antigen marginals while deranging {split}')
        donors = ordered[largest:] + ordered[:largest]
        for target, donor in zip(ordered, donors):
            if target['resolved_ag_seq'] == donor['resolved_ag_seq']:
                raise ValueError('Antigen derangement retained an original identity')
            result[target['id']] = donor
    return result


def paths(study, size):
    root = Path(study['experiment_root'])
    cohort = root / f'n{size}/cohort'
    return dict(records=str(cohort / 'asd_records.csv'), audit=str(cohort / 'split_audit.json'),
                embeddings=str(root / f'n{size}/asd/embeddings'))


def prepare_scaling(config):
    import yaml
    study = yaml.safe_load(Path(config).read_text())
    root = Path(study['experiment_root'])
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(study['prototype_config']).read_text())
    original_path = Path(study['prototype_root']) / 'cohort/asd_records.csv'
    original_audit_path = original_path.with_name('split_audit.json')
    original = read_csv(original_path)
    original_audit = json.loads(original_audit_path.read_text())
    if sha256(original_path) != original_audit['sha256'][str(original_path)]:
        raise ValueError('Original ASD prototype changed')
    source = Path(base['prototype']['source_cohort'])
    database = Path(base['prototype']['audit_database'])
    source_audit = json.loads((source / 'split_audit.json').read_text())
    for path in (source / 'asd_records.csv', source / 'sabdab2_records.csv', database):
        if source_audit['sha256'].get(str(path)) != sha256(path):
            raise ValueError(f'Audited full ASD source changed: {path}')
    manifest = dict(configuration=study, prototype_configuration=base,
                    prototype_records_sha256=sha256(original_path),
                    prototype_audit_sha256=sha256(original_audit_path),
                    source_audit_sha256=sha256(source / 'split_audit.json'),
                    preparation_sha256=sha256(__file__))
    manifest_path = root / 'experiment_manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Scaling inputs changed; use a new experiment_root')
    if not manifest_path.is_file():
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    overview_path = root / 'cohort_audit.json'
    if overview_path.is_file():
        frozen = json.loads(overview_path.read_text())
        for entry in frozen['training'].values():
            if sha256(entry['records']) != entry['records_sha256']:
                raise ValueError('Frozen scaling cohort changed')
        if sha256(root / 'evaluation_records.csv') != frozen['heldout_sha256']:
            raise ValueError('Frozen scaling held-out data changed')
        print(json.dumps(frozen, indent=2))
        return frozen
    overview = {}
    heldout = [r for r in original if r['split'] != 'train']
    blocked = her2_antigens(database)
    eligible = [r for r in read_csv(source / 'asd_records.csv')
                if r['split'] == 'train' and r['antigen_id'] not in blocked and r.get('has_buzz') != 'True'
                and len(r['resolved_ag_seq']) <= study['max_antigen_tokens']]
    # Deduplicate before taking size prefixes so skipped complete duplicates do
    # not prevent an exact 4,000-record cohort or change earlier target masks.
    stream = diverse_subset(eligible, len(eligible), study['max_records_per_antigen'], study['seed'])
    stream = select_examples(stream)
    if len(stream) < max(study['sizes']):
        raise ValueError(f'Only {len(stream)} distinct capped records available for scaling')
    native_reserved = [r for r in read_csv(source / 'sabdab2_records.csv') if r['split'] in {'val', 'test'}]
    smaller = None
    for size in study['sizes']:
        locations = paths(study, size)
        train = stream[:size]
        if smaller is not None and train[:len(smaller)] != smaller:
            raise ValueError('Larger training set does not preserve the smaller target masks')
        smaller = train
        rows = train + heldout
        checks = dict(train_vs_heldout=assert_disjoint(train, heldout),
                      train_vs_native_reserved=assert_disjoint(train, native_reserved))
        write_csv(locations['records'], rows)
        audit = dict(status='split_complete', pretraining_dataset='asd', her2_excluded=True,
                     excluded_her2_antigen_ids=sorted(blocked), overlap_checks=checks,
                     sha256={locations['records']: sha256(locations['records'])})
        Path(locations['audit']).write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
        counts = Counter(chain for r in train for chain in example_chains(r))
        overview[str(size)] = dict(n_records=size, chain_examples=sum(counts.values()),
            examples_by_chain=dict(counts), antigens=len({r['resolved_ag_seq'] for r in train}),
            families=len(family_set(train)), groups=len({r['ab_ag_cluster'] for r in train}),
            unique_heavy=len({r['resolved_H_seq'] for r in train}), unique_light=len({r['resolved_L_seq'] for r in train}),
            records_sha256=sha256(locations['records']), **locations)
    write_csv(root / 'evaluation_records.csv', heldout)
    result = dict(status='complete', training=overview, heldout_records=dict(Counter(r['split'] for r in heldout)),
                  heldout_examples=dict(Counter(r['split'] for r in heldout for _ in example_chains(r))),
                  heldout_sha256=sha256(root / 'evaluation_records.csv'), nested_training=True,
                  her2_excluded=True, native_finetuning=False)
    overview_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_scaling.yaml')
    prepare_scaling(parser.parse_args().config)
