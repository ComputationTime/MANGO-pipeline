"""Freeze the audited ASD 2.0 positive-binding pool for transfer pretraining.

Consumes the read-only ASD family/RIOT audit, not AbRank training examples.
Reuses the fixed SAbDab2 transfer cohort and additionally protects its validation
families, exact targets, antigen sequences and PDBs from all ASD pretraining.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

from chain_examples import example_chains
from prepare_abrank_transfer import assign_splits, assert_disjoint, family_set
from prepare_cath_sequences import sequence_id, write_csv


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def fasta(path):
    sequences = {}
    key = None
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):
                key = line[1:].split()[0]
                sequences[key] = ''
            elif line.strip():
                sequences[key] += line.strip()
    for key, sequence in sequences.items():
        if sequence_id(sequence) != key:
            raise ValueError(f'Antigen hash mismatch: {key}')
    return sequences


def pdb_ids(value):
    return set(re.findall(r'(?<![a-z0-9])[0-9][a-z0-9]{3}(?![a-z0-9])',
                         str(value).lower().replace('pdb_0000', '')))


def select_examples(rows):
    """Deduplicate each (antigen, requested chain, target) after group splitting."""
    seen = {}
    selected = []
    for row in rows:
        chains = []
        for chain, column in [('heavy', 'resolved_H_seq'), ('light', 'resolved_L_seq')]:
            key = (row['resolved_ag_seq'], chain, row[column])
            if key in seen:
                if seen[key] != row['split']:
                    raise ValueError('A duplicate individual-chain target crosses splits')
            else:
                seen[key] = row['split']
                chains.append(chain)
        if chains:
            row['example_target_chains'] = ';'.join(chains)
            selected.append(row)
    return selected


def candidate_query(include_buzz=True):
    query = '''SELECT r.antigen_ids,t.heavy,t.light,a.cath_families,
        GROUP_CONCAT(DISTINCT r.source),GROUP_CONCAT(DISTINCT t.target_pdb),
        COUNT(*),MIN(r.row_id),MAX(r.source='buzz'),MAX(r.source<>'buzz')
        FROM rows r JOIN targets t USING(row_id) JOIN eligibility e USING(row_id)
        JOIN antigens a ON a.antigen_id=r.antigen_ids
        WHERE e.family_safe=1 AND e.homology_safe=1 AND e.pdb_safe=1
          AND e.heavy_safe=1 AND e.light_safe=1
          AND (e.binding_positive=1 OR (r.source='buzz' AND r.affinity IN ('h','m','l')))
    '''
    if not include_buzz:
        query += " AND r.source<>'buzz'"
    return query + ' GROUP BY r.antigen_ids,t.heavy,t.light ORDER BY r.antigen_ids,t.heavy,t.light'


def prepare(audit_dir, binding_counts, native_cohort, output, include_buzz=True, seed=0, decoder_positions=2048):
    audit_dir, native_cohort, output = map(Path, (audit_dir, native_cohort, output))
    counts_path = Path(binding_counts)
    database = audit_dir / 'antigen_rows.sqlite'
    source_audit = audit_dir / 'family_audit.json'
    sequences_path = audit_dir / 'antigen_queries.fasta'
    old_audit = json.loads((native_cohort / 'split_audit.json').read_text())
    snapshot = json.loads((native_cohort / 'input_manifest.json').read_text())
    counts = json.loads(counts_path.read_text())
    # Refuse stale/edited eligibility decisions or a different native cohort.
    for path in (database, source_audit):
        expected = counts['input_sha256'].get(str(path.resolve()), counts['input_sha256'].get(str(path)))
        if not expected or sha256(path) != expected:
            raise ValueError(f'ASD binding-policy audit does not pin this input: {path}')
    native_path = native_cohort / 'sabdab2_records.csv'
    if sha256(native_path) != old_audit['sha256'][str(native_path)]:
        raise ValueError('The fixed SAbDab2 fine-tuning cohort changed')
    inputs = [database, source_audit, sequences_path, counts_path,
              native_cohort / 'split_audit.json', native_cohort / 'input_manifest.json', native_path,
              native_cohort / 'protected_sabdab2_test.csv', native_cohort / 'sabdab2_test_coverage.csv']
    manifest = dict(pretraining_dataset='asd', include_buzz=include_buzz, seed=seed,
                    decoder_positions=decoder_positions,
                    record_id_policy='minimum_qualifying_frozen_ASD_row_id_v1',
                    protected_figures={path: sha256(path) for path in snapshot['protected_figures']},
                    inputs={str(p): sha256(p) for p in inputs})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'input_manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Preparation inputs changed; choose a new cohort directory')
    completed = output / 'split_audit.json'
    if manifest_path.is_file() and completed.is_file():
        existing = json.loads(completed.read_text())
        if existing.get('status') != 'split_complete':
            raise ValueError('Existing ASD split audit is incomplete')
        for path, expected in existing['sha256'].items():
            if sha256(path) != expected:
                raise ValueError(f'Prepared ASD input/output changed: {path}')
        print(f'Reusing unchanged audited ASD cohort: {output}', flush=True)
        return existing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    native = read_csv(native_path)
    reserved = [r for r in native if r['split'] in {'val', 'test'}]
    families = family_set(reserved)
    heavy = {r['resolved_H_seq'] for r in reserved}
    light = {r['resolved_L_seq'] for r in reserved}
    antigen = {s for r in reserved for column in ('resolved_ag_seq', 'expected_ag_seq') for s in r[column].split(',')}
    pdbs = set().union(*(pdb_ids(r.get('pdb_id', r['id'])) for r in reserved))
    sequences = fasta(sequences_path)
    db = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=MEMORY')
    query = candidate_query(include_buzz)
    candidates, exclusions = [], []
    before = Counter()
    for ag, h, l, row_families, sources, target_pdb, observations, source_row, buzz, other in db.execute(query):
        before['unique_triples'] += 1
        before['positive_observations'] += observations
        before['triples_with_buzz'] += buzz
        before['triples_with_non_buzz_source'] += other
        reasons = []
        if set(row_families.split(';')) & families:
            reasons.append('reserved_native_cath_family')
        if h in heavy or l in light:
            reasons.append('reserved_native_target_chain')
        if sequences[ag] in antigen:
            reasons.append('reserved_native_antigen')
        if pdb_ids(target_pdb) & pdbs:
            reasons.append('reserved_native_pdb')
        # Random controls seed on record identity. Use the frozen source row,
        # never an antibody-sequence hash, to keep targets out of conditioning.
        identifier = f'asd_{source_row:09d}'
        if reasons:
            exclusions.append(dict(id=identifier, antigen_id=ag, reasons=';'.join(reasons)))
            continue
        candidates.append(dict(id=identifier, split='', ab_ag_cluster='',
            resolved_ag_seq=sequences[ag], antigen_chains='AG', resolved_H_seq=h, resolved_L_seq=l,
            target_contract='imgt_variable_vh_vl_v1', cath_families=row_families,
            antigen_id=ag, source=sources.replace(',', ';'), source_row_id=source_row,
            positive_observations=observations, has_buzz=bool(buzz), has_non_buzz_source=bool(other),
            target_annotation='ASD_RIOT_IMGT_primary_sequence_aa'))
    db.close()
    expected = counts['with_buzz' if include_buzz else 'without_buzz']['unique_antigen_vh_vl_triples']
    if before['unique_triples'] != expected:
        raise ValueError(f'ASD policy recount differs: {before["unique_triples"]} != {expected}')
    # A shuffled donor may have a different length from the recipient antigen.
    # One conservative bound admits every donor/target pairing in every arm.
    max_target = max(max(len(r['resolved_H_seq']), len(r['resolved_L_seq'])) for r in candidates)
    max_antigen = decoder_positions - max_target - 3  # selector, BOS, EOS
    retained = []
    for row in candidates:
        if len(row['resolved_ag_seq']) > max_antigen:
            exclusions.append(dict(id=row['id'], antigen_id=row['antigen_id'],
                                   reasons='full_context_exceeds_shared_decoder_capacity'))
        else:
            retained.append(row)
    candidates = retained
    print(f'ASD: {len(candidates):,} eligible triples; {len(exclusions):,} reserved-cohort/capacity exclusions', flush=True)
    assign_splits(candidates, seed)
    for row in candidates:
        row['ab_ag_cluster'] = row['ab_ag_cluster'].replace('abrank_cath_', 'asd_cath_', 1)
    triple_counts = {s: len([r for r in candidates if r['split'] == s]) for s in ('train', 'val', 'test')}
    records = select_examples(candidates)
    split_rows = {s: [r for r in records if r['split'] == s] for s in ('train', 'val', 'test')}
    overlaps = {f'{a}_vs_{b}': assert_disjoint(split_rows[a], split_rows[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    reserved_checks = assert_disjoint(records, reserved)
    for name in ('sabdab2_records.csv', 'protected_sabdab2_test.csv', 'sabdab2_test_coverage.csv'):
        shutil.copyfile(native_cohort / name, output / name)
    write_csv(output / 'asd_records.csv', records)
    write_csv(output / 'exclusions.csv', exclusions, ['id', 'antigen_id', 'reasons'])
    summaries = {}
    for split, rows in split_rows.items():
        by_chain = Counter(chain for r in rows for chain in example_chains(r))
        summaries[split] = dict(records=len(rows), unique_paired_triples=triple_counts[split],
            examples=sum(by_chain.values()), examples_by_chain=dict(by_chain),
            antigen_sequences=len({r['antigen_id'] for r in rows}), families=len(family_set(rows)),
            groups=len({r['ab_ag_cluster'] for r in rows}),
            triples_with_buzz=sum(r['has_buzz'] for r in candidates if r['split'] == split),
            triples_with_non_buzz_source=sum(r['has_non_buzz_source'] for r in candidates if r['split'] == split))
    audit = dict(status='split_complete', pretraining_dataset='asd', seed=seed,
        binding_policy=counts['policy'], include_buzz=include_buzz, candidate_pool=dict(before),
        record_id_policy=manifest['record_id_policy'],
        deduplication='Exact antigen/VH/VL triples, then exact (antigen, requested chain, target) examples after splitting',
        target_annotation='ASD RIOT IMGT primary_sequence_aa; no independent ANARCI rerun',
        decoder_capacity=dict(positions=decoder_positions, max_target_residues=max_target,
                              max_antigen_tokens=max_antigen,
                              policy='Common whole-sequence filter for all arms, including shuffled donors; no truncation'),
        family_definition=old_audit['family_definition'], search=old_audit['search'],
        protected_test_records=old_audit['protected_test_records'],
        primary_cath_assigned_test_records=old_audit['primary_cath_assigned_test_records'],
        secondary_only_test_records=old_audit['secondary_only_test_records'],
        protected_cath_families=old_audit['protected_cath_families'],
        split_policy=old_audit['abrank_split_policy'],
        asd=summaries, sabdab2=old_audit['sabdab2'],
        exclusions=dict(Counter(reason for r in exclusions for reason in r['reasons'].split(';'))),
        overlap_checks=dict(asd=overlaps, all_asd_vs_native_validation_and_test=reserved_checks),
        limitation=old_audit['limitation'],
        sha256={**manifest['inputs'], **{str(output / name): sha256(output / name)
                for name in ['asd_records.csv', 'sabdab2_records.csv', 'protected_sabdab2_test.csv',
                             'sabdab2_test_coverage.csv', 'exclusions.csv', 'input_manifest.json']}})
    (output / 'split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', default='artifacts/data/asd/sabdab2_family_audit_v1')
    parser.add_argument('--binding-counts', default='artifacts/data/asd/sabdab2_binding_any_affinity_v1/counts.json')
    parser.add_argument('--native-cohort', default='artifacts/experiments/abrank_cath_transfer_v1/cohort')
    parser.add_argument('--output', default='artifacts/experiments/asd_cath_transfer_v1/cohort')
    parser.add_argument('--without-buzz', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--decoder-positions', type=int, default=2048)
    args = parser.parse_args()
    prepare(args.audit_dir, args.binding_counts, args.native_cohort, args.output,
            include_buzz=not args.without_buzz, seed=args.seed, decoder_positions=args.decoder_positions)
