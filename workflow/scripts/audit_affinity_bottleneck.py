"""Typed protection-graph audit and conservative, source-linked affinity curation.

This is a census of the local data, not a training experiment. Ablations diagnose
which graph relations merge records. They never approve a revised split. No
unknown assay, uncertainty, replicate count or biological parent is imputed.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import json
import math
from pathlib import Path
import re

from affinity_study_common import digest, fingerprint, freeze, read_csv, read_json, write_csv, write_json
from prepare_native_pilot import verified_reference
from prepare_abrank_transfer import UnionFind

EDGE_TYPES = {'source_cluster', 'exact_antigen', 'cath_family',
              'heavy_sequence', 'light_sequence', 'near_antibody'}


def pdb_code(row):
    return row['pdb_id'][-4:].lower()


def graph_edges(rows, hits):
    edges = set()
    for row in rows:
        node = 'row:' + row['id']
        edges.add((node, 'source:' + row['ab_ag_cluster'], 'source_cluster'))
        edges.add((node, 'antigen:' + fingerprint(row['resolved_ag_seq']), 'exact_antigen'))
        for family in row['cath_families'].split(';'):
            if family:
                edges.add((node, 'cath:' + family, 'cath_family'))
        for chain, kind in [('H', 'heavy_sequence'), ('L', 'light_sequence')]:
            if row[f'resolved_{chain}_seq']:
                edges.add((node, 'sequence:' + fingerprint(row[f'resolved_{chain}_seq']), kind))
    for a, b, identity, qcov, tcov, evalue in hits:
        if float(identity) < .9 or min(float(qcov), float(tcov)) < .8 or float(evalue) > .001:
            raise ValueError('Cached antibody alignment violates the frozen threshold')
        if a != b:
            edges.add(('sequence:' + min(a, b), 'sequence:' + max(a, b), 'near_antibody'))
    return sorted(edges)


def partition(rows, edges, kinds=EDGE_TYPES):
    uf = UnionFind()
    for row in rows:
        uf.find('row:' + row['id'])
    for a, b, kind in edges:
        if kind in kinds:
            uf.union(a, b)
    groups = defaultdict(list)
    for row in rows:
        groups[uf.find('row:' + row['id'])].append(row['id'])
    return sorted((tuple(sorted(g)) for g in groups.values()), key=lambda g: (-len(g), g))


def exclusion_paths(edges, direct_ids):
    """Deterministic shortest witnesses, with every factor/edge type retained."""
    adjacent = defaultdict(list)
    for a, b, kind in edges:
        adjacent[a].append((b, kind)); adjacent[b].append((a, kind))
    queue = deque('row:' + rid for rid in sorted(direct_ids))
    previous = {node: None for node in queue}
    for node in queue:
        adjacent[node].sort()
    while queue:
        node = queue.popleft()
        for nxt, kind in sorted(adjacent[node]):
            if nxt not in previous:
                previous[nxt] = (node, kind); queue.append(nxt)
    result = []
    for node in sorted(n for n in previous if n.startswith('row:')):
        cursor = node; nodes = [cursor]; kinds = []
        while previous[cursor] is not None:
            cursor, kind = previous[cursor]; nodes.append(cursor); kinds.append(kind)
        result.append(dict(id=node[4:], direct=node[4:] in direct_ids,
                           witness_id=cursor[4:], steps=len(kinds),
                           edge_types=';'.join(reversed(kinds)),
                           path_nodes=json.dumps(list(reversed(nodes)))))
    return result


def edit_distance(a, b):
    previous=list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        current=[i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]


def annotate_near_edges(cfg, rows, edges, root):
    """CDR3 differences are diagnostics, not an inferred clonal assignment."""
    regions=read_json(cfg['region_cache'])['sequences']
    roles=defaultdict(set);cdr3={}
    for r in rows:
        for c in ('H','L'):
            seq=r[f'resolved_{c}_seq']
            if not seq:continue
            key=fingerprint(seq);roles[key].add(c)
            item=regions.get(key)
            if item and item['sequence']==seq and len(item['labels'])==len(seq):
                value=''.join(aa for aa,label in zip(seq,item['labels']) if label=='cdr3')
                if value:cdr3[key]=value
    annotated=[]
    for a,b,kind in edges:
        if kind!='near_antibody':continue
        a=a.split(':')[1];b=b.split(':')[1]
        av=cdr3.get(a,'');bv=cdr3.get(b,'')
        annotated.append(dict(a=a,b=b,roles_a=';'.join(sorted(roles[a])),roles_b=';'.join(sorted(roles[b])),
                              both_cdr3_numbered=bool(av and bv),
                              cdr3_a=av,cdr3_b=bv,
                              cdr3_edit_distance=edit_distance(av,bv) if av and bv else '',
                              cdr3_identical=(av==bv) if av and bv else ''))
    write_csv(root/'near_antibody_cdr3_audit.csv',annotated)
    return dict(near_edges=len(annotated),both_cdr3_numbered=sum(r['both_cdr3_numbered'] for r in annotated),
                numbered_edges_with_different_cdr3=sum(r['cdr3_identical'] is False for r in annotated),
                interpretation='Coverage is incomplete. Different CDR3s do not alone establish independent lineages; V/J and experimental-parent evidence remain necessary.')


def audit_graph(cfg, reference, root):
    rows, groups, antigen_hits = verified_reference(reference, root)
    hits = list(csv.reader(open(Path(reference['reference_audit_root']) /
                               'searches/all_lineages/hits.tsv'), delimiter='\t'))
    edges = graph_edges(rows, hits)
    complete = partition(rows, edges)
    expected = sorted((tuple(sorted(r['id'] for r in group)) for group in groups), key=lambda g: (-len(g), g))
    if complete != expected:
        raise ValueError('Typed graph does not reconstruct the verified full partition')
    members = read_csv(Path(reference['inventory_root']) / 'membership.csv')
    original = defaultdict(list)
    for r in members:
        original[r['component']].append(r['id'])
    if partition(rows, edges, EDGE_TYPES - {'near_antibody'}) != sorted(
            (tuple(sorted(g)) for g in original.values()), key=lambda g: (-len(g), g)):
        raise ValueError('Typed graph does not reconstruct the inventory partition')
    direct = {r['id'] for r in rows if any(fingerprint(s) in antigen_hits for s in r['resolved_ag_seq'].split(','))}
    protected = {m['id'] for m in members if m['protected_group'] == 'True'}
    schemes = {
        'source_only': {'source_cluster'},
        'source_and_exact_antigen': {'source_cluster', 'exact_antigen'},
        'plus_CATH': {'source_cluster', 'exact_antigen', 'cath_family'},
        'plus_exact_heavy': EDGE_TYPES - {'light_sequence', 'near_antibody'},
        'inventory_exact_heavy_and_light': EDGE_TYPES - {'near_antibody'},
        'full_verified_graph': EDGE_TYPES,
        'diagnostic_without_CATH': EDGE_TYPES - {'cath_family'},
        'diagnostic_without_light_attachments': EDGE_TYPES - {'light_sequence'},
        'diagnostic_without_near_antibody': EDGE_TYPES - {'near_antibody'},
        'diagnostic_source_antigen_only': {'source_cluster', 'exact_antigen'},
    }
    summaries = []
    for name, kinds in schemes.items():
        parts = partition(rows, edges, kinds)
        quarantined = [g for g in parts if any(rid in direct for rid in g)]
        reserved = [g for g in parts if any(rid in protected for rid in g)]
        summaries.append(dict(scheme=name, edge_types=';'.join(sorted(kinds)),
                              records=len(rows), components=len(parts), largest_component=len(parts[0]),
                              direct_antigen_hits=len(direct), antigen_quarantined=sum(map(len, quarantined)),
                              protected_source_rows=len(protected), reserved_by_source=sum(map(len, reserved)),
                              applied_to_training=False))
    write_csv(root / 'graph_ablation.csv', summaries)
    write_csv(root / 'typed_graph_edges.csv', [dict(a=a, b=b, edge_type=k) for a,b,k in edges])
    witnesses = exclusion_paths(edges, direct)
    write_csv(root / 'antigen_exclusion_witnesses.csv', witnesses)
    hubs = defaultdict(set)
    for a, b, kind in edges:
        if a.startswith('row:'):
            hubs[kind, b].add(a[4:])
    by_id = {r['id']: r for r in rows}
    write_csv(root / 'graph_hubs.csv', sorted([
        dict(edge_type=kind, factor=node, records=len(ids),
             distinct_antigens=len({by_id[rid]['resolved_ag_seq'] for rid in ids}),
             distinct_heavy=len({by_id[rid]['resolved_H_seq'] for rid in ids}),
             distinct_source_groups=len({by_id[rid]['ab_ag_cluster'] for rid in ids}))
        for (kind, node), ids in hubs.items()], key=lambda r: (-r['records'], r['factor'])))
    result = dict(records=len(rows), components=len(complete), largest_component=len(complete[0]),
                  direct_antigen_hit_records=len(direct), antigen_quarantined=len(witnesses),
                  indirectly_quarantined=len(witnesses)-len(direct),
                  reproduced_inventory_partition=True, reproduced_full_partition=True,
                  edge_counts=dict(Counter(k for _,_,k in edges)), ablations=summaries,
                  interpretation='Diagnostic census only. Removing a relation is not evidence of safe data or a revised split.')
    result['near_edge_cdr3_diagnostics']=annotate_near_edges(cfg,rows,edges,root)
    write_json(root / 'graph_audit.json', result)
    return rows, result


def canonical_candidate(link, record, source):
    """Check the actual source row, then identify construct copies conservatively."""
    if (source['pdb'].lower() != pdb_code(record) or
        source['Hchain'] != record['heavy_chain'] or source['Lchain'] != record['light_chain']):
        raise ValueError('SAbDab source row does not identify the linked antibody: ' + record['id'])
    if not math.isclose(float(source['affinity']), float(link['affinity_molar']), rel_tol=1e-10):
        raise ValueError('SAbDab linked affinity changed')
    if source['pmid'] != link['pmid']:
        raise ValueError('SAbDab linked publication changed')
    if source['affinity_method'] != link['affinity_method'] or source['temperature'] != link['temperature']:
        raise ValueError('SAbDab linked method or temperature changed')
    seqs = [record[k] for k in ('expected_heavy_seq', 'expected_light_seq', 'expected_ag_seq')]
    # If the full construct is unavailable, do not collapse two different PDBs
    # by an identical unresolved/truncated structure sequence.
    sequence_key = seqs + ([] if seqs[0] and seqs[2] else [record['id']])
    key = sequence_key + [link['pmid'], link['affinity_method'], link['temperature'], str(link['affinity_molar'])]
    return fingerprint(json.dumps(key)), seqs


def audit_native_labels(cfg, source_records, root, safe_ids):
    links = read_json(cfg['candidate_links']); raw = read_csv_tsv(cfg['sabdab_affinity'])
    records = {r['id']:r for r in source_records}
    candidates = []; dedup = defaultdict(list)
    for link in links:
        row = records[link['record_id']]
        key, seqs = canonical_candidate(link, row, raw[int(link['source_row'])])
        item = dict(link, construct_measurement_key=key,
                    numeric_pmid=bool(re.fullmatch(r'[1-9][0-9]*', link['pmid'])),
                    protected_eligible=row['id'] in safe_ids,
                    full_heavy_available=bool(seqs[0]), full_antigen_available=bool(seqs[2]),
                    expected_heavy_sha256=fingerprint(seqs[0]), expected_light_sha256=fingerprint(seqs[1]),
                    expected_antigen_sha256=fingerprint(seqs[2]),
                    no_light_chain=not row['light_chain'],
                    parent_verified=False, assay_verified=False, uncertainty_verified=False)
        dedup[key].append(item); candidates.append(item)
    write_csv(root / 'native_affinity_candidates.csv', candidates)
    unique = []
    for key, values in sorted(dedup.items()):
        unique.append(dict(values[0], copy_count=len(values),
                           record_ids=';'.join(v['record_id'] for v in values)))
    write_csv(root / 'native_affinity_deduplicated.csv', unique)
    # These are only search bins for a human curator, never declared assays.
    bins = defaultdict(list)
    for r in unique:
        if r['numeric_pmid'] and r['full_heavy_available'] and r['full_antigen_available']:
            key = (r['pmid'], r['expected_antigen_sha256'], r['expected_light_sha256'],
                   r['affinity_method'], r['temperature'])
            bins[key].append(r)
    write_csv(root / 'native_comparison_candidates.csv', [
        dict(pmid=key[0], antigen_sha256=key[1], light_sha256=key[2], method=key[3],
             temperature_unverified=key[4], variants=len({r['expected_heavy_sha256'] for r in values}),
             distinct_kd=len({r['affinity_molar'] for r in values}),
             max_kd_ratio=max(r['affinity_molar'] for r in values)/min(r['affinity_molar'] for r in values),
             record_ids=';'.join(r['record_id'] for r in values), assay_verified=False)
        for key, values in sorted(bins.items()) if len({r['expected_heavy_sha256'] for r in values}) > 1])
    result = dict(candidate_links=len(links), numeric_publication_links=sum(r['numeric_pmid'] for r in candidates),
                  missing_publication_links=sum(not r['numeric_pmid'] for r in candidates),
                  numeric_publications=len({r['pmid'] for r in candidates if r['numeric_pmid']}),
                  deduplicated_construct_measurement_links=len(unique),
                  additional_crystal_copies_collapsed=len(links)-len(unique),
                  deduplication_is_not_biological_independence=True,
                  protected_eligible_candidate_links=sum(r['protected_eligible'] for r in candidates),
                  approved_training_labels=0,
                  reason='Database linkage does not verify parent, complete assay construct, or measurement uncertainty.')
    write_json(root / 'native_affinity_audit.json', result)
    return result


def read_csv_tsv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def parse_mutations(value):
    result = []
    for token in re.split('[,;]', value):
        match = re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])([A-Za-z0-9])(-?\d+[A-Za-z]?)([ACDEFGHIKLMNPQRSTVWY])', token.strip())
        if not match:
            raise ValueError('Unsupported mutation notation: ' + token)
        wt, chain, number, mut = match.groups()
        result.append((wt, chain, number, mut))
    return result


def audit_skempi(cfg, source_records, root, safe_ids):
    with open(cfg['skempi'], newline='') as handle:
        raw = list(csv.DictReader(handle, delimiter=';'))
    known = {r['#Pdb'] for r in read_csv(cfg['skempi_antibody_pairs'])}
    by_pdb = defaultdict(list)
    for r in source_records:
        by_pdb[pdb_code(r)].append(r)
    safe_pdb = {pdb_code(r) for r in source_records if r['id'] in safe_ids}
    results = []; roles = []
    for index, r in enumerate(raw):
        if r['#Pdb'] not in known:
            continue
        pdb, partner1, partner2 = r['#Pdb'].split('_')
        annotations = {(s['heavy_chain'],s['light_chain']) for s in by_pdb[pdb.lower()]
                       if s['heavy_chain'] in partner1 + partner2 and
                       (not s['light_chain'] or s['light_chain'] in partner1 + partner2)}
        antibody_partner = next((p for p in (partner1,partner2)
            if any(h in p and (not l or l in p) for h,l in annotations)), '')
        matches = {(h,l) for h,l in annotations if h in antibody_partner and (not l or l in antibody_partner)}
        role_ok = len(matches) == 1
        h, l = next(iter(matches)) if role_ok else ('','')
        try:
            mutations = parse_mutations(r['Mutation(s)_PDB'])
            heavy_only = role_ok and bool(mutations) and all(m[1] == h for m in mutations)
            parse_ok = True
        except ValueError:
            heavy_only = False; parse_ok = False
        results.append(dict(source_row=index, complex=r['#Pdb'], reference=r['Reference'],
                            heavy_chain=h, light_chain=l, chain_role_resolved=role_ok,
                            mutation=r['Mutation(s)_PDB'], mutation_parse_ok=parse_ok,
                            heavy_only_by_chain=heavy_only, protected_eligible=pdb.lower() in safe_pdb,
                            kd_mut_raw=r['Affinity_mut (M)'], kd_wt_raw=r['Affinity_wt (M)'],
                            method=r['Method'], temperature_raw=r['Temperature'],
                            source_notes=r['Notes'],
                            variable_domain_mutations_verified=False,
                            replicate_count='', uncertainty_sd='', approved_training_label=False))
        if role_ok:
            roles.append(dict(complex=r['#Pdb'], antibody_partner=antibody_partner,
                              native_heavy_chain=h, native_light_chain=l,
                              upstream_first_chain_is_heavy=antibody_partner[0] == h))
    roles = list({r['complex']:r for r in roles}.values())
    write_csv(root / 'skempi_candidate_audit.csv', results)
    write_csv(root / 'skempi_chain_roles.csv', sorted(roles,key=lambda r:r['complex']))
    # Verify the inversion against actual sequences, rather than a chain-order
    # heuristic. Full variable-domain prefix must match the opposite column.
    inversions = []
    for r in read_csv_tsv(cfg['skempi_processed']):
        name = r['Ab_name'].upper()
        pdb = r['Ab_PDB_ID']
        if pdb.lower() not in by_pdb:
            continue
        for s in by_pdb[pdb.lower()]:
            h=s['expected_heavy_seq']; l=s['expected_light_seq']
            if h and l and r['Ab_heavy_chain_seq'].startswith(l) and r['Ab_light_chain_seq'].startswith(h):
                inversions.append(dict(processed_antibody=name, native_record=s['id'],
                                       heavy_column_matches_native_light=True,
                                       light_column_matches_native_heavy=True))
                break
    write_csv(root / 'skempi_verified_column_inversions.csv', inversions)
    result = dict(total_rows=len(raw), locally_identified_antibody_rows=len(results),
                  identified_complexes=len({r['complex'] for r in results}),
                  chain_role_resolved_rows=sum(r['chain_role_resolved'] for r in results),
                  heavy_only_by_chain_rows=sum(r['heavy_only_by_chain'] for r in results),
                  protected_eligible_rows=sum(r['protected_eligible'] for r in results),
                  verified_processed_column_inversions=len(inversions),
                  approved_training_labels=0,
                  limitations=['Local antibody complex index is not a complete SKEMPI antibody annotation.',
                    'Heavy-chain membership does not establish variable-domain location or correct residue numbering.',
                    'No replicate count or uncertainty is inferred from repeated WT rows.',
                    'All identified complexes fail the unchanged protected native whitelist.'])
    write_json(root / 'skempi_audit.json', result)
    return result


def curate_primary(cfg, records, root, safe_ids):
    ledger = read_json(cfg['primary_ledger']); result = []
    for entry in ledger['measurements']:
        candidates = [r for r in records if pdb_code(r) in entry['pdb_codes'] and not r['light_chain']]
        sequences = {r['expected_heavy_seq'] for r in candidates} - {''}
        if len(sequences) > 1:
            raise ValueError('Curated variant maps to conflicting PDB sequences')
        r = dict(entry, native_record_ids=';'.join(sorted(r['id'] for r in candidates)),
                 variable_heavy_sequence=next(iter(sequences)) if sequences else '',
                 protected_native_ids=';'.join(sorted(r['id'] for r in candidates if r['id'] in safe_ids)),
                 construct_verified=False, uncertainty_verified=False, approved_training_label=False)
        r['pdb_codes']=';'.join(entry['pdb_codes'])
        result.append(r)
    write_csv(root / 'primary_curated_candidates.csv', result)
    write_json(root / 'primary_curation.json', dict(measurements=len(result),
        independent_documented_parents=len({r['parent_id'] for r in result}),
        mapped_measurements=sum(bool(r['variable_heavy_sequence']) for r in result),
        approved_training_labels=0, reason='Error type, replicate count and exact display construct remain unresolved.'))
    return result


def plot(root, graph, curated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in graph['ablations'] if r['scheme'] in (
        'source_only','plus_CATH','inventory_exact_heavy_and_light','full_verified_graph')]
    fig, axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].bar(range(len(selected)),[r['antigen_quarantined'] for r in selected],color='#366c95')
    axes[0].set_xticks(range(len(selected)),['Source\ngroups','+ exact Ag\n+ CATH','+ exact\nH and L','+ near\nantibody'])
    axes[0].set_ylabel('Records quarantined by antigen-hit propagation')
    axes[0].set_title('Deterministic census; no sampling error bars')
    for i,r in enumerate(selected): axes[0].text(i,r['antigen_quarantined']+35,str(r['antigen_quarantined']),ha='center')
    axes[0].set_ylim(0,graph['records']*1.13)
    for i,r in enumerate(curated):
        axes[1].errorbar(i,r['kd_nm'],yerr=r['reported_plus_minus_nm'],fmt='o',capsize=4,color='#a85832')
    axes[1].set_yscale('log');axes[1].set_ylabel('Reported display KD (nM)')
    axes[1].set_xticks(range(len(curated)),['2007\nWT','2007\na','2008\nWT','2008\n22','2008\n23','2008\n7'])
    axes[1].set_title('Published +/- values; error type unresolved')
    fig.suptitle('Data bottleneck audit — no model comparison or significance claim')
    fig.savefig(root/'bottleneck_audit.png',dpi=180);plt.close(fig)


def main():
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/affinity_bottleneck_audit.yaml')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg['apply_grouping_changes']:
        raise ValueError('This audit cannot modify biological grouping or authorize training')
    reference=yaml.safe_load(Path(cfg['reference_config']).read_text())
    root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    inputs=[args.config,cfg['reference_config'],cfg['candidate_links'],cfg['sabdab_affinity'],cfg['skempi'],cfg['region_cache'],
            cfg['skempi_antibody_pairs'],cfg['skempi_processed'],cfg['primary_ledger'],reference['source_records'],
            str(Path(reference['experiment_root'])/'cohort_audit.json'),
            str(Path(reference['experiment_root'])/'component_membership.csv')]
    helpers=[str(Path(__file__)),'workflow/scripts/prepare_native_pilot.py',
             'workflow/scripts/affinity_study_common.py','workflow/scripts/prepare_abrank_transfer.py']
    freeze(root/'audit_protocol.json',dict(settings=cfg,input_sha256={p:digest(p) for p in inputs},
           implementation_sha256={p:digest(p) for p in helpers},
           purpose='Diagnose grouping and curate labels; no retrospective split relaxation or scientific fitting.'))
    oldaudit=read_json(Path(reference['experiment_root'])/'cohort_audit.json')
    for name,value in oldaudit['files_sha256'].items():
        if digest(Path(reference['experiment_root'])/name)!=value:
            raise ValueError('Protected pilot cohort changed: '+name)
    safe_ids={r['id'] for r in read_csv(Path(reference['experiment_root'])/'component_membership.csv') if not r['excluded']}
    source_records=read_csv(reference['source_records'])
    rows,graph=audit_graph(cfg,reference,root)
    native=audit_native_labels(cfg,source_records,root,safe_ids)
    skempi=audit_skempi(cfg,source_records,root,safe_ids)
    curated=curate_primary(cfg,source_records,root,safe_ids)
    plot(root,graph,curated)
    arms=[dict(arm=arm,conditioning=conditioning,objective=objective)
          for arm,conditioning,objective in [('IgLM_CE','constant','ce'),('MANGO_CE','antigen_only','ce'),
          ('IgLM_CE_rank','constant','ce_plus_within_assay_ranking'),
          ('MANGO_CE_rank','antigen_only','ce_plus_within_assay_ranking')]]
    freeze(root/'four_arm_design.json',dict(arms=arms, status='blocked_before_training',
        comparisons=['MANGO_CE - IgLM_CE','MANGO_CE_rank - IgLM_CE_rank',
                     '(MANGO_CE_rank - MANGO_CE) - (IgLM_CE_rank - IgLM_CE)'],
        shared=['decoder size and initialization seed','CE batches and token budget','same ranking pairs and updates',
                'antigen-only biological conditioning and learned H/L selector','unchanged biological holdout protection'],
        primary_metric='Macro within-assay, within-parent Spearman correlation with negative log10 KD',
        uncertainty='Paired bootstrap of independent held-out parents within target; optimization seed SD reported separately. Insufficient parents means no biological CI.',
        architecture='Hold PLM and geometric representation fixed until data and objective comparison is feasible.',
        gate=dict(approved_affinity_labels=0,safe_native_records=len(safe_ids),
                  minimum_training_records=cfg['minimum_training_records'],
                  minimum_test_heavy_examples=cfg['minimum_test_heavy_examples'],
                  minimum_test_groups=cfg['minimum_test_groups'],
                  power_calculation_completed=False),
        reasons=['No approved assay-compatible affinity labels with verified uncertainty and constructs.',
                 f'{len(safe_ids)} protected native records cannot meet disjoint {cfg["minimum_training_records"]}-train + {cfg["minimum_test_heavy_examples"]}-heavy-test minimum.']))
    report=dict(status='audit_complete_training_blocked',graph=graph,native=native,skempi=skempi,
                safe_native_records=len(safe_ids),scientific_fits_launched=0,
                approved_affinity_training_labels=0,existing_protocols_changed=False,
                independent_primary_parents=len({r['parent_id'] for r in curated}),
                plots=['bottleneck_audit.png'])
    outputs=[p for p in root.iterdir() if p.is_file() and p.name not in ('results_report.json','findings.md')]
    report['output_sha256']={p.name:digest(p) for p in sorted(outputs)}
    write_json(root/'results_report.json',report)
    full=next(r for r in graph['ablations'] if r['scheme']=='full_verified_graph')
    without=next(r for r in graph['ablations'] if r['scheme']=='diagnostic_without_near_antibody')
    text=(f"Affinity bottleneck audit\n\n"
          f"The audit completed; scientific training remains blocked by data eligibility. No existing split was changed.\n\n"
          f"The typed graph exactly reproduces {graph['records']} records, {graph['components']} components and the {graph['largest_component']}-record largest component. "
          f"There are {graph['direct_antigen_hit_records']} direct antigen-hit records; propagation quarantines {graph['antigen_quarantined']}, including {graph['indirectly_quarantined']} indirect exclusions. "
          f"Without near-antibody edges, quarantine is {without['antigen_quarantined']} rather than {full['antigen_quarantined']}; this is diagnostic, not a safe revised split.\n\n"
          f"The {native['candidate_links']} affinity links contain {native['numeric_publication_links']} links to {native['numeric_publications']} numeric PubMed IDs; {native['missing_publication_links']} lack a usable ID. "
          f"Conservative construct/measurement deduplication leaves {native['deduplicated_construct_measurement_links']} links, not independent biological measurements.\n\n"
          f"The raw SKEMPI audit identifies {skempi['locally_identified_antibody_rows']} antibody rows; {skempi['heavy_only_by_chain_rows']} involve only the mapped heavy chain, before residue/domain validation. "
          f"None map to the unchanged protected whitelist. {skempi['verified_processed_column_inversions']} upstream processed rows have heavy/light columns reversed, verified against native sequences.\n\n"
          f"Primary-source curation captures {len(curated)} reported measurements from one RNase-A VHH parent. These are not independent parents. "
          f"The plotted error bars reproduce published +/- values; their definition and replicate counts remain unresolved, so they are not labeled SD, SEM or confidence intervals. "
          f"The 2007 clone-a narrative (0.18 nM) disagrees with Table 1/SAbDab (0.15 nM); construct tags/termini also need reconciliation. "
          f"Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC2148503/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC2442005/.\n\n"
          f"The four-arm factorial design is frozen in four_arm_design.json. Zero affinity labels currently pass all gates, and {len(safe_ids)} protected native records cannot meet the existing 150-train + 100-heavy-test minimum. "
          f"Counts are a deterministic census and do not need sampling error bars. No model performance, power or statistical confirmation claim follows.\n")
    (root/'findings.md').write_text(text)
    print(json.dumps(dict(status=report['status'],graph=graph,native=native,skempi=skempi),indent=2))


if __name__=='__main__':
    main()
