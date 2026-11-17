"""Complete native-cohort protection before a separately versioned mechanism pilot.

No model outcomes or affinity values participate in eligibility or allocation.
This does not relax or overwrite the ASD v2/v3 protocols.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3

from affinity_study_common import digest, fingerprint, freeze, read_csv, read_json, write_csv, write_json
from accelerated_affinity_search import search
from prepare_affinity_study import fasta
from prepare_abrank_transfer import UnionFind
from prepare_controlled_antigen import allocate
from prepare_asd_transfer import select_examples
from chain_examples import example_chains


def containment(cfg, root, queries, targets):
    """Both directions detect a protected domain contained in a longer chain."""
    import subprocess
    found = set()
    for coverage in (1, 2):
        directory = root / 'searches' / f'antigen_containment_{coverage}'
        directory.mkdir(parents=True, exist_ok=True)
        fasta(directory / 'query.fa', queries)
        fasta(directory / 'target.fa', targets)
        command = [cfg['mmseqs'], 'easy-search', str(directory / 'query.fa'),
                   str(directory / 'target.fa'), str(directory / 'hits.tsv'), str(directory / 'tmp'),
                   '--min-seq-id', '0.2', '-c', '0.6', '--cov-mode', str(coverage),
                   '-e', '0.001', '-s', '7.5', '--threads', '4', '--max-seqs', '100000',
                   '--max-accept', '1', '--alignment-mode', '3',
                   '--format-output', 'query,target,fident,qcov,tcov,evalue']
        freeze(directory / 'manifest.json', dict(command=command,
               query_sha256=digest(directory/'query.fa'), target_sha256=digest(directory/'target.fa')))
        with (directory / 'search.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        found.update(line.split('\t')[0] for line in (directory/'hits.tsv').read_text().splitlines())
    return found


def audit(root):
    root = Path(root)
    cfg = read_json(root / 'protocol.json')['settings']
    inventory = Path(cfg['inventory_root'])
    info = read_json(inventory / 'results_report.json')
    for path, expected in info['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Inventory input changed: ' + path)
    if digest(inventory/'membership.csv') != info['membership_sha256']:
        raise ValueError('Inventory membership changed')
    source = {r['id']: r for r in read_csv(cfg['source_records'])}
    membership = read_csv(inventory/'membership.csv')
    rows = []
    excluded = []
    for member in membership:
        r = source[member['id']]
        # The native IgLM positional limit includes chain/species tokens and EOS.
        if any(len(r[f'resolved_{c}_seq']) > 509 for c in ('H', 'L')):
            excluded.append(dict(id=r['id'], reason='decoder_capacity'))
            continue
        rows.append(dict(r, source_split='reserved' if member['component_reserved']=='True' else 'train',
                         cath_families=member['cath_families'], inventory_component=member['component']))
    external = Path(cfg['external_root'])
    inputs = [Path(cfg['source_records']), inventory/'membership.csv', inventory/'results_report.json',
              external/'domains.sqlite', external/'antigens.csv', Path(cfg['reserved_records'])]
    freeze(root/'data_manifest.json', dict(files_sha256={str(p):digest(p) for p in inputs}))
    with sqlite3.connect(f'file:{(external/"domains.sqlite").resolve()}?mode=ro', uri=True) as db:
        external_domains = {r[0] for r in db.execute("SELECT target FROM domains WHERE status='valid'")}
    antigens = {chain for r in read_csv(external/'antigens.csv') for chain in r['resolved_ag_seq'].split(',') if chain}
    ag_hits = containment(cfg, root, {s for r in rows for s in r['resolved_ag_seq'].split(',')}, antigens)
    sequences = {r[f'resolved_{c}_seq'] for r in rows for c in ('H','L')} - {''}
    # Keep all accepted lineage edges, rather than the existence-only shortcut.
    search(cfg, root, 'all_lineages', sequences, sequences, True)
    uf = UnionFind()
    for r in rows:
        node = 'row:' + r['id']
        uf.union(node, 'inventory:' + r['inventory_component'])
        for c in ('H','L'):
            if r[f'resolved_{c}_seq']:
                uf.union(node, 'sequence:' + fingerprint(r[f'resolved_{c}_seq']))
    for line in (root/'searches/all_lineages/hits.tsv').read_text().splitlines():
        a,b = line.split('\t')[:2]
        uf.union('sequence:'+a, 'sequence:'+b)
    groups = defaultdict(list)
    for r in rows:
        groups[uf.find('row:'+r['id'])].append(r)
    ab_hits = search(cfg, root, 'external_antibodies', sequences, external_domains, True)
    reserved = {r[f'resolved_{c}_seq'] for r in read_csv(cfg['reserved_records']) if r['split']!='train' for c in ('H','L')} - {''}
    reserved_hits = search(cfg, root, 'reserved_antibodies', sequences, reserved, True)
    safe = []
    components = []
    for group in groups.values():
        group = sorted(group, key=lambda r:r['id'])
        gid = 'native_' + fingerprint(';'.join(r['id'] for r in group))[:20]
        # Quarantine complete components, preserving the graph built BEFORE exclusion.
        reasons = []
        if any(any(fingerprint(s) in ag_hits for s in r['resolved_ag_seq'].split(',')) for r in group):
            reasons.append('external_antigen_containment')
        if any(any(fingerprint(r[f'resolved_{c}_seq']) in ab_hits for c in ('H','L')) for r in group):
            reasons.append('external_antibody_lineage')
        if any(any(fingerprint(r[f'resolved_{c}_seq']) in reserved_hits for c in ('H','L')) for r in group):
            reasons.append('reserved_ASD_antibody_lineage')
        for r in group:
            components.append(dict(id=r['id'], group=gid, source_split=r['source_split'],
                                   ab_ag_cluster=r['ab_ag_cluster'], excluded=';'.join(reasons)))
            excluded.extend(dict(id=r['id'], reason=reason) for reason in reasons)
            r['full_biological_group'] = gid
        if not reasons:
            safe.append(group)
    write_csv(root/'component_membership.csv', components)
    write_csv(root/'exclusions.csv', excluded)
    allocated = allocate(safe, cfg)
    feasible = allocated is not None
    if feasible:
        for r in allocated:
            r['biological_group'] = r['full_biological_group']
        allocated = select_examples(allocated)
        # Preserve complete components; no row cap may remove a panel's small groups.
        counts = {s:dict(records=sum(r['split']==s for r in allocated),
                         components=len({r['biological_group'] for r in allocated if r['split']==s}),
                         heavy_examples=sum(r['split']==s and 'heavy' in example_chains(r) for r in allocated))
                  for s in ('train','val','test')}
        feasible = all(counts[s]['components']>=cfg['minimum_groups'] and
                       counts[s]['heavy_examples']>=cfg['minimum_heavy_examples'] for s in ('val','test'))
        if feasible:
            write_csv(root/'records.csv', allocated)
    else:
        counts = {}
    files = ['component_membership.csv','exclusions.csv'] + (['records.csv'] if feasible else [])
    write_json(root/'cohort_audit.json', dict(complete=True, feasible=feasible,
        reason='requirements_satisfied' if feasible else 'insufficient_protected_independent_capacity',
        original_rows=len(rows), original_components=len(groups), safe_components=len(safe),
        safe_records=sum(map(len,safe)), safe_training_records=sum(r['source_split']=='train' for g in safe for r in g),
        exclusions=dict(Counter(r['reason'] for r in excluded)), counts=counts,
        component_sizes=sorted(map(len,safe),reverse=True),
        files_sha256={n:digest(root/n) for n in files},
        limitations=['Native source outcomes have previously been inspected; this is exploratory internal replication.',
                     'Homology-connected components are conservative dependence blocks, not verified pedigrees.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/config/native_mechanism_study.yaml')
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg['experiment_root'])
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path('experiments/scripts/accelerated_affinity_search.py'),
               Path('experiments/scripts/prepare_controlled_antigen.py'), Path(args.config)]
    freeze(root/'protocol.json', dict(settings=cfg, purpose='prospective native-cohort eligibility audit',
           implementation_sha256={str(p):digest(p) for p in sources}))
    audit(root)
