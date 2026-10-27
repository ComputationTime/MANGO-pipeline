"""Small, antigen-diverse HER2-free ASD pilot using already audited splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

from chain_examples import example_chains
from prepare_abrank_transfer import assert_disjoint, family_set
from prepare_asd_transfer import read_csv, select_examples, sha256
from prepare_cath_sequences import write_csv

HER2 = re.compile(r'(?<![a-z0-9])(?:her[\s_-]*2|erbb[\s_-]*2|erb[\s_-]*b[\s_-]*2|'
                  r'epidermal\s+growth\s+factor\s+receptor\s+2)(?![a-z0-9])', re.I)


def her2_antigens(database):
    """Exclude named HER2/ERBB2 sequences across every source, including BUZZ."""
    with sqlite3.connect(f'file:{Path(database).resolve()}?mode=ro', uri=True) as db:
        blocked = {ag for ag, name in db.execute('SELECT DISTINCT r.antigen_ids,t.target_name FROM rows r JOIN targets t USING(row_id)')
                   if name and HER2.search(name)}
        blocked.update(ag for ag, in db.execute("SELECT DISTINCT antigen_ids FROM rows WHERE source='buzz'"))
    return blocked


def diverse_subset(rows, limit, per_antigen=10, seed=0):
    """Round-robin over antigens within a fixed split; never reassign groups."""
    if limit < 1 or per_antigen < 1:
        raise ValueError('Prototype record limits must be positive')
    key = lambda identifier: hashlib.sha256(f'{seed}:{identifier}'.encode()).digest()
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['antigen_id']].append(row)
    ordered = sorted(buckets, key=key)
    for antigen in ordered:
        buckets[antigen].sort(key=lambda r: key(r['id']))
    result = []
    for index in range(per_antigen):
        for antigen in ordered:
            if index < len(buckets[antigen]):
                result.append(dict(buckets[antigen][index]))
                if len(result) == limit:
                    return result
    return result


def prepare(study):
    settings = dict(study['prototype'])
    source = Path(settings['source_cohort'])
    output = Path(study['cohort_dir'])
    database = Path(settings['audit_database'])
    source_audit_path = source / 'split_audit.json'
    source_audit = json.loads(source_audit_path.read_text())
    if source_audit.get('status') != 'split_complete' or source_audit.get('pretraining_dataset') != 'asd':
        raise ValueError('A completed audited ASD source cohort is required')
    if study['model']['epochs'] != 1:
        raise ValueError('The rapid prototype requires exactly one epoch per stage')
    for path, expected in source_audit['sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Audited source changed: {path}')
    pinned_database = source_audit['sha256'].get(str(database))
    if not pinned_database or sha256(database) != pinned_database:
        raise ValueError('The source audit does not pin this HER2 annotation database')
    inputs = [source_audit_path, database, source / 'asd_records.csv', source / 'sabdab2_records.csv',
              source / 'sabdab2_test_coverage.csv', source / 'protected_sabdab2_test.csv', source / 'input_manifest.json']
    source_manifest = json.loads((source / 'input_manifest.json').read_text())
    manifest = dict(pretraining_dataset='asd', preparation='her2_free_antigen_diverse_prototype_v1',
                    settings=settings, protected_figures=source_manifest['protected_figures'],
                    inputs={str(p): sha256(p) for p in inputs},
                    preparation_sha256=sha256(Path(__file__)))
    for path, expected in manifest['protected_figures'].items():
        if sha256(path) != expected:
            raise ValueError(f'Protected figure changed: {path}')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'input_manifest.json'
    audit_path = output / 'split_audit.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Prototype inputs/settings changed; choose a new experiment directory')
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            for path, expected in audit['sha256'].items():
                if sha256(path) != expected:
                    raise ValueError(f'Frozen prototype changed: {path}')
            print(f'Reusing unchanged prototype cohort: {output}', flush=True)
            return audit
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    blocked = her2_antigens(database)
    buckets = {split: [] for split in ('train', 'val', 'test')}
    exclusions = Counter()
    with (source / 'asd_records.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['antigen_id'] in blocked or row.get('has_buzz') == 'True':
                exclusions['her2_or_buzz'] += 1
                continue
            if len(row['resolved_ag_seq']) > settings['max_antigen_tokens']:
                exclusions['antigen_over_prototype_length_limit'] += 1
                continue
            buckets[row['split']].append(row)
    rows, summaries = [], {}
    for split in ('train', 'val', 'test'):
        limit = settings[f'max_{split}_records']
        selected = diverse_subset(buckets[split], limit, settings['max_records_per_antigen'], settings['seed'])
        selected = select_examples(selected)  # Recompute target masks after sampling.
        if len(selected) < 2:
            raise ValueError(f'Prototype split {split} has too few eligible records')
        rows.extend(selected)
        chains = Counter(chain for r in selected for chain in example_chains(r))
        summaries[split] = dict(records=len(selected), examples=sum(chains.values()), examples_by_chain=dict(chains),
            eligible_nonher2_short_records=len(buckets[split]), antigen_sequences=len({r['antigen_id'] for r in selected}),
            families=len(family_set(selected)), groups=len({r['ab_ag_cluster'] for r in selected}),
            antigen_tokens_max=max(len(r['resolved_ag_seq']) for r in selected),
            antigen_tokens_mean=sum(len(r['resolved_ag_seq']) for r in selected) / len(selected))
    by_split = {s: [r for r in rows if r['split'] == s] for s in buckets}
    checks = {f'{a}_vs_{b}': assert_disjoint(by_split[a], by_split[b])
              for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    native = read_csv(source / 'sabdab2_records.csv')
    reserved_check = assert_disjoint(rows, [r for r in native if r['split'] in {'val', 'test'}])
    assert not {r['antigen_id'] for r in rows} & blocked
    assert all(r.get('has_buzz') != 'True' for r in rows)
    for name in ('sabdab2_records.csv', 'sabdab2_test_coverage.csv', 'protected_sabdab2_test.csv'):
        shutil.copyfile(source / name, output / name)
    write_csv(output / 'asd_records.csv', rows)
    audit = dict(source_audit)
    audit.update(status='split_complete', prototype=True, include_buzz=False,
        asd=summaries, source_cohort=str(source),
        prototype_policy=settings, excluded_her2_antigen_ids=sorted(blocked), exclusions=dict(exclusions),
        split_policy='Inherit audited source splits/groups; seeded antigen round-robin selects a bounded subset within each split',
        overlap_checks=dict(asd=checks, all_asd_vs_native_validation_and_test=reserved_check),
        deduplication='Recompute exact (antigen, requested chain, target) deduplication after subset selection',
        sha256={**manifest['inputs'], **{str(output / name): sha256(output / name)
               for name in ('asd_records.csv', 'sabdab2_records.csv', 'sabdab2_test_coverage.csv',
                            'protected_sabdab2_test.csv', 'input_manifest.json')}})
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(asd=summaries, exclusions=dict(exclusions), her2_antigen_ids_excluded=len(blocked)), indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/asd_prototype.yaml')
    args = parser.parse_args()
    import yaml
    with open(args.config) as handle:
        prepare(yaml.safe_load(handle))
