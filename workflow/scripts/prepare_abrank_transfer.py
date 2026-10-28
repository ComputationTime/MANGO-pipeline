"""Build audited AbRank pretraining and CATH-disjoint SAbDab2 transfer cohorts.

Uses pinned CATH sequence-search results and IMGT-numbered antibody domains.
Unknown antigen families are quarantined from development. The original 426
SAbDab2 test rows remain unchanged; a coverage table defines the primary,
CATH-assigned evaluation subset without rewriting that test set.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from prepare_cath_sequences import clean_sequence, sequence_id, write_csv


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def components(rows, preserve_source_groups=False):
    """Keep every family, exact target chain, and source group in one split."""
    uf = UnionFind()
    for row in rows:
        key = 'row:' + row['id']
        uf.find(key)
        for family in row['cath_families'].split(';'):
            if family:
                uf.union(key, 'cath:' + family)
        for chain in ['H', 'L']:
            uf.union(key, chain + ':' + row[f'resolved_{chain}_seq'])
        if preserve_source_groups:
            uf.union(key, 'group:' + row['source_ab_ag_cluster'])
        elif row.get('source_antigen_cluster'):
            uf.union(key, 'ag_cluster:' + row['source_antigen_cluster'])
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[uf.find('row:' + row['id'])].append(row)
    return sorted(grouped.values(), key=lambda g: min(r['id'] for r in g))


def select_holdout(groups, target, seed, max_groups=None):
    """Deterministic subset-sum selection uses counts, never model outcomes."""
    ordered = sorted(groups, key=lambda g: hashlib.sha256(
        f"{seed}:{min(r['id'] for r in g)}".encode()).hexdigest())
    reachable = {(0, 0): ()}
    for i, group in enumerate(ordered):
        for (total, count), indices in list(reachable.items()):
            if max_groups is not None and count >= max_groups:
                continue
            key = (total + len(group), count + 1)
            if key not in reachable:
                reachable[key] = indices + (i,)
    positive = [key for key in reachable if key[0] > 0]
    if not positive:
        raise ValueError('No independent component available for held-out split')
    size = min(positive, key=lambda key: (abs(key[0] - target), key[0], key[1]))
    selected = set(reachable[size])
    return ([g for i, g in enumerate(ordered) if i in selected],
            [g for i, g in enumerate(ordered) if i not in selected])


def assign_splits(rows, seed=0):
    groups = components(rows)
    if len(groups) < 3:
        raise ValueError('At least three independent CATH/antibody components required')
    largest = max(groups, key=len)
    remaining = [g for g in groups if g is not largest]
    # Split independent components, not rows: a dominant HER2 component must
    # not force every other family into validation/test to reach row fractions.
    n_holdout = max(1, round(len(groups) * .1))
    test, remaining = select_holdout(remaining, max(1, round(len(rows) * .1)), seed, n_holdout)
    val, remaining = select_holdout(remaining, max(1, round(len(rows) * .1)), seed + 1, n_holdout)
    for split, selected in [('train', [largest] + remaining), ('val', val), ('test', test)]:
        for group in selected:
            gid = 'abrank_cath_' + hashlib.sha256(';'.join(sorted(r['id'] for r in group)).encode()).hexdigest()[:16]
            for row in group:
                row.update(split=split, ab_ag_cluster=gid)
    return rows


def family_set(rows):
    return set().union(*(set(r['cath_families'].split(';')) - {''} for r in rows)) if rows else set()


def audit_pair(left, right):
    result = {'cath_families': sorted(family_set(left) & family_set(right))}
    for key in ['id', 'ab_ag_cluster', 'resolved_H_seq', 'resolved_L_seq']:
        overlap = {r[key] for r in left} & {r[key] for r in right}
        # Avoid reproducing sequence strings in the compact audit.
        result[key] = len(overlap)
    return result


def assert_disjoint(left, right):
    result = audit_pair(left, right)
    if any(result.values()):
        raise ValueError(f'Contamination between cohorts: {result}')
    return result


def cath_annotations(cohort, reference):
    domain_families = {}
    for line in (reference / 'cath-domain-list-S100.txt').open():
        fields = line.split()
        if fields and not fields[0].startswith('#'):
            domain_families[fields[0]] = '.'.join(fields[1:5])
    all_families = collections.defaultdict(set)
    confident = collections.defaultdict(set)
    for hit in csv.reader((cohort / 'cath_sequence_hits.tsv').open(), delimiter='\t'):
        domain = hit[1].split('|')[-1].split('/')[0]
        if domain not in domain_families:
            raise ValueError(f'CATH hit lacks classification: {domain}')
        family = domain_families[domain]
        # Recheck thresholds, rather than trusting a possibly different search.
        if float(hit[2]) >= 20 and float(hit[13]) >= .6 and float(hit[10]) <= .001:
            all_families[hit[0]].add(family)
        if float(hit[2]) >= 35 and float(hit[13]) >= .8 and float(hit[10]) <= 1e-5:
            confident[hit[0]].add(family)
    native = collections.defaultdict(set)
    with gzip.open(reference / 'cath-b-newest-all.gz', 'rt') as fh:
        for line in fh:
            fields = line.split()
            if len(fields) >= 3:
                native[fields[0][:5]].add(fields[2])
    row_families = collections.defaultdict(set)
    row_confident_chains = collections.defaultdict(set)
    row_sids = collections.defaultdict(set)
    for member in csv.DictReader((cohort / 'antigen_members.csv').open()):
        key = (member['dataset'], member['record_id'])
        sid = member['sequence_id']
        row_families[key].update(all_families[sid])
        row_sids[key].add(sid)
        if confident[sid]:
            row_confident_chains[key].add(member['chain'])
    return row_families, row_confident_chains, row_sids, native


def pdb_codes(value):
    return set(re.findall(r'(?<![A-Za-z0-9])[0-9][A-Za-z0-9]{3}(?![A-Za-z0-9])', value.lower()))


def prepare(abrank, sabdab, cohort, reference, numbered, seed=0):
    cohort, reference = Path(cohort), Path(reference)
    input_manifest = json.loads((cohort / 'input_manifest.json').read_text())
    for dataset, path in [('abrank', abrank), ('sabdab2', sabdab)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != input_manifest[dataset]['sha256']:
            raise ValueError(f'{dataset} input changed after protected snapshot')
    native_rows = list(csv.DictReader(open(sabdab)))
    protected = [r for r in native_rows if r['split'] == 'test']
    if protected != list(csv.DictReader((cohort / 'protected_sabdab2_test.csv').open())):
        raise ValueError('Protected SAbDab2 test changed')
    annotations, confident_chains, row_sids, native_cath = cath_annotations(cohort, reference)
    for row in native_rows:
        key = ('sabdab2', row['id'])
        for chain in row['antigen_chains'].split(','):
            if len(chain) == 1:
                family = native_cath[row['pdb_id'][-4:].lower() + chain]
                annotations[key].update(family)
                if family:
                    confident_chains[key].add(chain)
    blocked_families = set().union(*(annotations['sabdab2', r['id']] for r in protected))
    protected_chains = {c: {r[f'resolved_{c}_seq'] for r in protected} for c in ['H', 'L']}
    protected_pdb = {r['pdb_id'][-4:].lower() for r in protected}
    homologous_sids = set()
    for hit in csv.reader((cohort / 'protected_antigen_hits.tsv').open(), delimiter='\t'):
        if float(hit[2]) >= 20 and max(float(hit[12]), float(hit[13])) >= .6 and float(hit[10]) <= .001:
            homologous_sids.add(hit[0])
    coverage = []
    primary_ids = set()
    for row in protected:
        assigned = set(row['antigen_chains'].split(',')) <= confident_chains['sabdab2', row['id']]
        if assigned:
            primary_ids.add(row['id'])
        coverage.append(dict(id=row['id'], ab_ag_cluster=row['ab_ag_cluster'],
                             cath_assigned_all_antigen_chains=assigned,
                             cath_families=';'.join(sorted(annotations['sabdab2', row['id']])),
                             evaluation='primary_and_secondary' if assigned else 'secondary_only'))
    if not primary_ids:
        raise ValueError('No CATH-assigned protected test records')
    numbered_data = json.loads(Path(numbered).read_text())
    rejected, candidates = [], []
    def exclude(dataset, rid, reason, source=''):
        rejected.append(dict(dataset=dataset, id=rid, source=source, reason=reason))
    seen_triples = set()
    for index, raw in enumerate(csv.DictReader(open(abrank), delimiter='\t')):
        rid = f'abrank_{index:07d}'
        source = raw['Source']
        def reject(reason):
            exclude('abrank', rid, reason, source)
        try:
            antigen = clean_sequence(raw['Ag_seq'])
        except ValueError:
            reject('invalid_antigen_sequence'); continue
        key = ('abrank', rid)
        if annotations[key] & blocked_families:
            reject('protected_test_cath_family'); continue
        if 'AG' not in confident_chains[key]:
            reject('unassigned_antigen_cath'); continue
        if row_sids[key] & homologous_sids:
            reject('protected_test_antigen_homology'); continue
        pdbname = ' '.join(raw[c] for c in ['Ab_PDB_ID', 'Ag_PDB_ID', 'bound_AbAg_PDB_ID'])
        if pdb_codes(pdbname) & protected_pdb:
            reject('protected_test_pdb'); continue
        affinity = []
        for column in ['Affinity_Kd [nM]', 'IC50 [ug/mL]']:
            try:
                value = float(raw[column])
                if math.isfinite(value) and value > 0: affinity.append(value)
            except ValueError:
                pass
        if not affinity or raw['Aff_op'].strip() not in {'=', '<', '<='}:
            reject('no_uncensored_or_upper_bound_binding_evidence'); continue
        targets = {}
        for c, column, chain in [('H', 'Ab_heavy_chain_seq', 'heavy'), ('L', 'Ab_light_chain_seq', 'light')]:
            try:
                seq = clean_sequence(raw[column])
                numbered_entry = numbered_data[sequence_id(seq)]
                if numbered_entry['input_sequence'] != seq or numbered_entry['expected_chain'] != chain:
                    raise ValueError('Incompatible antibody numbering cache')
                if numbered_entry['status'] == 'valid' and len(numbered_entry['domains']) == 1:
                    targets[c] = numbered_entry['domains'][0]['sequence']
            except (ValueError, KeyError):
                pass
        if len(targets) != 2:
            reject('missing_or_ambiguous_imgt_variable_target'); continue
        if any(targets[c] in protected_chains[c] for c in ['H', 'L']):
            reject('protected_test_exact_antibody_chain'); continue
        triple = (targets['H'], targets['L'], antigen)
        if triple in seen_triples:
            reject('duplicate_antibody_antigen_triple'); continue
        seen_triples.add(triple)
        candidates.append(dict(id=rid, pdb_path='', antigen_chains='AG', chains='H,L,AG',
            expected_heavy_seq=targets['H'], expected_light_seq=targets['L'],
            expected_ag_seq=antigen, resolved_H_seq=targets['H'], resolved_L_seq=targets['L'],
            resolved_ag_seq=antigen, split='', ab_ag_cluster='',
            target_contract='imgt_variable_vh_vl_v1', cath_families=';'.join(sorted(annotations[key])),
            source=source, source_row_index=index, source_antigen_name=raw['Ag_name'],
            source_antigen_cluster=raw['Ag25_cluster'],
            source_full_heavy_seq=raw['Ab_heavy_chain_seq'], source_full_light_seq=raw['Ab_light_chain_seq']))
    abrank_rows = assign_splits(candidates, seed)
    abrank_by_split = {s: [r for r in abrank_rows if r['split'] == s] for s in ['train', 'val', 'test']}
    abrank_checks = {}
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        abrank_checks[first + '_vs_' + second] = assert_disjoint(abrank_by_split[first], abrank_by_split[second])
    # Drop whole SAbDab2 source groups when any member fails a protection check.
    unsafe = {}
    for row in native_rows:
        if row['split'] == 'test': continue
        key = ('sabdab2', row['id'])
        reasons = []
        if annotations[key] & blocked_families: reasons.append('protected_test_cath_family')
        if not set(row['antigen_chains'].split(',')) <= confident_chains[key]: reasons.append('unassigned_antigen_cath')
        if row_sids[key] & homologous_sids: reasons.append('protected_test_antigen_homology')
        if row['pdb_id'][-4:].lower() in protected_pdb: reasons.append('protected_test_pdb')
        if any(row[f'resolved_{c}_seq'] in protected_chains[c] for c in ['H', 'L']): reasons.append('protected_test_exact_antibody_chain')
        if reasons:
            unsafe.setdefault(row['ab_ag_cluster'], set()).update(reasons)
    development = []
    for row in native_rows:
        if row['split'] == 'test': continue
        if row['ab_ag_cluster'] in unsafe:
            exclude('sabdab2', row['id'], ';'.join(sorted(unsafe[row['ab_ag_cluster']])), 'sabdab2')
            continue
        row = dict(row, source_ab_ag_cluster=row['ab_ag_cluster'],
                   original_split=row['split'], cath_families=';'.join(sorted(annotations['sabdab2', row['id']])))
        development.append(row)
    groups = components(development, preserve_source_groups=True)
    pretraining_development = abrank_by_split['train'] + abrank_by_split['val']
    pretraining_families = family_set(pretraining_development)
    pretraining_chains = {c: {r[f'resolved_{c}_seq'] for r in pretraining_development} for c in ['H', 'L']}
    eligible_val = [g for g in groups if not family_set(g) & pretraining_families and not any(
        r[f'resolved_{c}_seq'] in pretraining_chains[c] for r in g for c in ['H', 'L'])]
    val_groups, _ = select_holdout(eligible_val, max(1, round(len(development) * .1)), seed + 2)
    val_ids = {r['id'] for g in val_groups for r in g}
    for row in development:
        row['split'] = 'val' if row['id'] in val_ids else 'train'
        # Source ab_ag_cluster remains unchanged for SAbDab2.
    native_out = development + [dict(r, cath_families=';'.join(sorted(annotations['sabdab2', r['id']])),
                                    original_split='test', source_ab_ag_cluster=r['ab_ag_cluster']) for r in protected]
    native_by_split = {s: [r for r in native_out if r['split'] == s] for s in ['train', 'val', 'test']}
    native_checks = {}
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        native_checks[first + '_vs_' + second] = assert_disjoint(native_by_split[first], native_by_split[second])
    # Pretraining and its checkpoint selection must also protect downstream tests.
    pretraining_test_check = assert_disjoint(abrank_rows, native_by_split['test'])
    validation_check = audit_pair(pretraining_development, native_by_split['val'])
    if validation_check['cath_families'] or validation_check['resolved_H_seq'] or validation_check['resolved_L_seq']:
        raise ValueError('Fine-tuning validation was seen during AbRank model development')
    native_fields = list(native_rows[0]) + ['cath_families', 'original_split', 'source_ab_ag_cluster']
    write_csv(cohort / 'abrank_records.csv', abrank_rows)
    write_csv(cohort / 'sabdab2_records.csv', native_out, native_fields)
    write_csv(cohort / 'sabdab2_test_coverage.csv', coverage)
    write_csv(cohort / 'exclusions.csv', rejected)
    annotation_rows = [dict(dataset=dataset, id=rid, cath_families=';'.join(sorted(families)),
                            confident_chains=';'.join(sorted(confident_chains[dataset, rid])))
                       for (dataset, rid), families in sorted(annotations.items())]
    write_csv(cohort / 'cath_annotations.csv', annotation_rows)
    hashes = {}
    for path in [Path(abrank), Path(sabdab), Path(numbered),
                 cohort/'cath_sequence_hits.tsv', cohort/'protected_antigen_hits.tsv',
                 reference/'cath-domain-list-S100.txt', reference/'cath-domain-seqs-S100.fa',
                 reference/'cath-b-newest-all.gz', cohort/'abrank_records.csv', cohort/'sabdab2_records.csv']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(status='split_complete', seed=seed,
        family_definition='CATH homologous superfamily C.A.T.H (four numeric levels); union of all antigen-chain/domain hits',
        search=dict(confident='identity >=35%, CATH domain coverage >=80%, E <=1e-5',
                    exclusion='identity >=20%, CATH domain coverage >=60%, E <=0.001',
                    direct_test_homology='identity >=20%, coverage of either antigen >=60%, E <=0.001'),
        protected_test_records=len(protected), primary_cath_assigned_test_records=len(primary_ids),
        secondary_only_test_records=len(protected)-len(primary_ids),
        protected_cath_families=sorted(blocked_families),
        abrank_split_policy='Reserve largest connected component for training. Deterministic subset-sum targets 10% of rows for test then validation, capped at 10% of independent components per holdout to preserve training diversity. Ties use seed-hashed component order.',
        abrank={s: dict(records=len(rs), examples=2*len(rs), families=len(family_set(rs)),
                       groups=len({r['ab_ag_cluster'] for r in rs}), sources=dict(collections.Counter(r['source'] for r in rs)))
                for s, rs in abrank_by_split.items()},
        sabdab2={s: dict(records=len(rs), examples=2*len(rs), families=len(family_set(rs)),
                        groups=len({r['ab_ag_cluster'] for r in rs})) for s, rs in native_by_split.items()},
        exclusions={d: dict(collections.Counter(r['reason'] for r in rejected if r['dataset']==d)) for d in ['abrank','sabdab2']},
        overlap_checks=dict(abrank=abrank_checks,sabdab2=native_checks,
                            all_abrank_vs_sabdab2_test=pretraining_test_check,
                            abrank_development_vs_finetuning_validation=validation_check),
        limitation='Family disjointness is verified against the pinned CATH annotations and sequence-homology screens, not a guarantee about unannotated remote homology. All 426 original tests are retained; only the CATH-assigned subset is the primary family-disjoint evaluation.',
        sha256=hashes)
    (cohort / 'split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k:v for k,v in audit.items() if k in ['status','abrank','sabdab2','primary_cath_assigned_test_records','secondary_only_test_records','exclusions']},indent=2),flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abrank', required=True)
    parser.add_argument('--sabdab', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--numbered', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    prepare(**vars(args))
