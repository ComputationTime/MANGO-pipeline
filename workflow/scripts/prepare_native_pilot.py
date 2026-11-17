"""Resume verified native searches; quarantine whole components before allocation.

The v1 study cannot support its two panels. This separately frozen fixed-budget
pilot uses a single exploratory panel, never a confirmation panel.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3

from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_csv,write_json
from accelerated_affinity_search import search
from prepare_abrank_transfer import UnionFind


def deduplicate(rows):
    seen=set();result=[]
    for original in sorted(rows,key=lambda r:r['id']):
        row=dict(original);chains=[]
        for chain,c in (('heavy','H'),('light','L')):
            seq=row[f'resolved_{c}_seq'];key=(row['resolved_ag_seq'],chain,seq)
            if seq and key not in seen:
                seen.add(key);chains.append(chain)
        if chains:
            row['example_target_chains']=';'.join(chains);result.append(row)
    return result


def allocate_single_panel(groups,cfg):
    """Deterministic whole-component subset allocation using data counts only."""
    groups=sorted((deduplicate(g) for g in groups),key=lambda g:g[0]['full_biological_group'])
    reserved=[g for g in groups if any(r['source_split']!='train' for r in g)]
    training=[g for g in groups if all(r['source_split']=='train' for r in g)]
    heavy=lambda g:sum('heavy' in r['example_target_chains'].split(';') for r in g)
    initial_h=sum(map(heavy,reserved));initial_g=len(reserved)
    capacity=sum(map(len,training))-cfg['minimum_training_records']
    if capacity<0:return None
    # State: capped holdout group/heavy counts and additional removed train rows.
    states={(min(initial_g,cfg['minimum_groups']),min(initial_h,cfg['minimum_heavy_examples']),0):()}
    for i,group in enumerate(training):
        for (ng,nh,nr),chosen in list(states.items()):
            if nr+len(group)>capacity:continue
            key=(min(ng+1,cfg['minimum_groups']),min(nh+heavy(group),cfg['minimum_heavy_examples']),nr+len(group))
            states.setdefault(key,chosen+(i,))
    candidates=sorted((nr,choice) for (ng,nh,nr),choice in states.items()
                      if ng>=cfg['minimum_groups'] and nh>=cfg['minimum_heavy_examples'])
    for _,choice in candidates:
        chosen=set(choice);train=[r for i,g in enumerate(training) if i not in chosen for r in g]
        if len({r['resolved_ag_seq'] for r in train})<cfg['minimum_training_antigens']:continue
        test=[r for g in reserved for r in g]+[r for i,g in enumerate(training) if i in chosen for r in g]
        result=[dict(r,split=split,biological_group=r['full_biological_group']) for split,rows in (('train',train),('test',test)) for r in rows]
        if any(r['source_split']!='train' for r in result if r['split']=='train'):raise ValueError('Prior holdout entered training')
        for field in ('ab_ag_cluster','biological_group'):
            assignment=defaultdict(set)
            for r in result:assignment[r[field]].add(r['split'])
            if any(len(v)>1 for v in assignment.values()):raise ValueError('Group crosses splits')
        return result
    return None


def verified_reference(cfg,root):
    reference=Path(cfg['reference_audit_root']);old=read_json(reference/'protocol.json')
    for path,value in old['implementation_sha256'].items():
        if digest(path)!=value:raise ValueError('Reference audit implementation changed: '+path)
    for path,value in read_json(reference/'data_manifest.json')['files_sha256'].items():
        if digest(path)!=value:raise ValueError('Reference source changed: '+path)
    inventory=Path(cfg['inventory_root']);info=read_json(inventory/'results_report.json')
    if digest(inventory/'membership.csv')!=info['membership_sha256']:raise ValueError('Inventory changed')
    for path,value in info['source_sha256'].items():
        if digest(path)!=value:raise ValueError('Inventory source changed: '+path)
    by_id={r['id']:r for r in read_csv(cfg['source_records'])};rows=[]
    for member in read_csv(inventory/'membership.csv'):
        r=by_id[member['id']]
        if any(len(r[f'resolved_{c}_seq'])>509 for c in ('H','L')):continue
        rows.append(dict(r,source_split='reserved' if member['component_reserved']=='True' else 'train',
                         cath_families=member['cath_families'],inventory_component=member['component']))
    graph=reference/'searches/all_lineages';complete=read_json(graph/'search_complete.json');manifest=read_json(graph/'search_manifest.json')
    for name in ('query','target'):
        if digest(graph/f'{name}.fa')!=manifest[f'{name}_sha256']:raise ValueError('Graph FASTA changed')
    sequences={r[f'resolved_{c}_seq'] for r in rows for c in ('H','L')} - {''}
    expected=''.join(f'>{fingerprint(s)}\n{s}\n' for s in sorted(sequences))
    if (graph/'query.fa').read_text()!=expected or (graph/'target.fa').read_text()!=expected:raise ValueError('Graph does not cover full candidate cohort')
    if digest(graph/'search_manifest.json')!=complete['manifest_sha256'] or digest(graph/'hits.tsv')!=complete['hits_sha256']:
        raise ValueError('Reference graph completion changed')
    hits=set();proof={str(graph/'search_complete.json'):digest(graph/'search_complete.json')}
    for mode in (1,2):
        directory=reference/f'searches/antigen_containment_{mode}';m=read_json(directory/'manifest.json')
        for name in ('query','target'):
            if digest(directory/f'{name}.fa')!=m[f'{name}_sha256']:raise ValueError('Containment input changed')
        # Both checked subprocess calls necessarily succeeded before the verified
        # all-lineages marker could have been written by the unchanged v1 script.
        for line in (directory/'hits.tsv').read_text().splitlines():
            q,t,identity,qcov,tcov,evalue=line.split('\t')
            if float(identity)<.2 or float(evalue)>.001 or float(tcov if mode==1 else qcov)<.6:
                raise ValueError('Containment hit violates recorded threshold')
            hits.add(q)
        proof[str(directory/'hits.tsv')]=digest(directory/'hits.tsv')
        proof[str(directory/'manifest.json')]=digest(directory/'manifest.json')
    uf=UnionFind()
    for r in rows:
        node='row:'+r['id'];uf.union(node,'inventory:'+r['inventory_component'])
        for c in ('H','L'):
            if r[f'resolved_{c}_seq']:uf.union(node,'sequence:'+fingerprint(r[f'resolved_{c}_seq']))
    for line in (graph/'hits.tsv').read_text().splitlines():
        a,b,*_=line.split('\t');uf.union('sequence:'+a,'sequence:'+b)
    groups=defaultdict(list)
    for r in rows:groups[uf.find('row:'+r['id'])].append(r)
    for group in groups.values():
        gid='native_'+fingerprint(';'.join(sorted(r['id'] for r in group)))[:20]
        for r in group:r['full_biological_group']=gid
    freeze(root/'reference_reuse.json',dict(files_sha256=proof,reference_protocol_sha256=digest(reference/'protocol.json'),
           interpretation='Verified completed stages only; interrupted external-antibody alignment is not reused as eligibility evidence.'))
    return rows,list(groups.values()),hits


def audit(root):
    root=Path(root);cfg=read_json(root/'audit_protocol.json')['settings']
    rows,groups,ag_hits=verified_reference(cfg,root)
    rejected=[];remaining=[]
    for g in groups:
        if any(fingerprint(s) in ag_hits for r in g for s in r['resolved_ag_seq'].split(',')):
            rejected.extend(dict(id=r['id'],reason='external_antigen_containment') for r in g)
        else:remaining.append(g)
    write_json(root/'capacity_before_antibody_screen.json',dict(original_records=len(rows),original_components=len(groups),
        retained_records=sum(map(len,remaining)),retained_components=len(remaining),
        original_two_panel_design_feasible=False,reason='298-record upper bound below original 350-record minimum'))
    external=Path(cfg['external_root'])
    with sqlite3.connect(f'file:{(external/"domains.sqlite").resolve()}?mode=ro',uri=True) as db:
        domains={r[0] for r in db.execute("SELECT target FROM domains WHERE status='valid'")}
    sequences={r[f'resolved_{c}_seq'] for g in remaining for r in g for c in ('H','L')} - {''}
    external_hits=search(cfg,root,'external_antibodies',sequences,domains,True)
    reserved={r[f'resolved_{c}_seq'] for r in read_csv(cfg['reserved_records']) if r['split']!='train' for c in ('H','L')} - {''}
    reserved_hits=search(cfg,root,'reserved_antibodies',sequences,reserved,True)
    safe=[];membership=[]
    for g in remaining:
        reasons=[]
        for name,hits in (('external_antibody_lineage',external_hits),('reserved_ASD_antibody_lineage',reserved_hits)):
            if any(fingerprint(r[f'resolved_{c}_seq']) in hits for r in g for c in ('H','L')):reasons.append(name)
        for r in g:
            membership.append(dict(id=r['id'],group=r['full_biological_group'],source_split=r['source_split'],excluded=';'.join(reasons)))
            rejected.extend(dict(id=r['id'],reason=reason) for reason in reasons)
        if not reasons:safe.append(g)
    write_csv(root/'component_membership.csv',membership);write_csv(root/'exclusions.csv',rejected)
    allocated=allocate_single_panel(safe,cfg)
    if allocated is not None:write_csv(root/'records.csv',allocated)
    counts={s:dict(records=sum(r['split']==s for r in allocated),groups=len({r['biological_group'] for r in allocated if r['split']==s}),
                 heavy_examples=sum(r['split']==s and 'heavy' in r['example_target_chains'].split(';') for r in allocated))
            for s in ('train','test')} if allocated else {}
    names=['component_membership.csv','exclusions.csv']+(['records.csv'] if allocated else [])
    write_json(root/'cohort_audit.json',dict(complete=True,feasible=allocated is not None,
        reason='single_panel_pilot_feasible' if allocated else 'insufficient_protected_independent_capacity',
        safe_records=sum(map(len,safe)),safe_components=len(safe),counts=counts,
        exclusions=dict(Counter(r['reason'] for r in rejected)),files_sha256={n:digest(root/n) for n in names},
        confirmation_eligible=False,scope='fixed-budget exploratory mechanism pilot; no confirmation or tuning on test'))


if __name__=='__main__':
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/native_mechanism_pilot_v2.yaml')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());root=Path(cfg['experiment_root'])
    root.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(args.config),Path('workflow/scripts/accelerated_affinity_search.py')]
    freeze(root/'audit_protocol.json',dict(settings=cfg,implementation_sha256={str(p):digest(p) for p in sources},
        purpose='single-panel exploratory feasibility audit; fixed model budget and no confirmation'))
    audit(root)
