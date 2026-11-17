"""Protected full-source audit and label-blind selection for the short screen."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sqlite3

from affinity_study_common import digest,freeze,read_csv,read_json,write_csv,write_json,fingerprint
from chain_examples import example_chains
from prepare_affinity_study import search
from prepare_controlled_antigen import components,allocate
from prepare_asd_transfer import select_examples
from short_affinity_common import balanced_select,schedule,affinity_pairs


def settings(root):return read_json(root/'protocol.json')['settings']


def diagnostic_selection(root):
    cfg=settings(root);binding=Path(cfg['binding_root']);queries={r['id']:r for r in read_csv(binding/'heldout_antibodies.csv')}
    with sqlite3.connect(f'file:{(binding/"measurements.sqlite").resolve()}?mode=ro',uri=True) as db:
        pairs=db.execute('SELECT query_id,antigen_id,reliable_sum/reliable_count FROM pairs WHERE reliable_count>0').fetchall()
    targets={r['id']:r['family'] for r in read_csv(binding/'antigens.csv')}
    with sqlite3.connect(f'file:{(binding/"domains.sqlite").resolve()}?mode=ro',uri=True) as db:
        cdr={seq:json.loads(indices) for seq,indices in db.execute("SELECT target,cdr3_indices FROM domains WHERE status='valid' AND chain='heavy'")}
    selected=[]
    for aid,family in targets.items():
        rows=sorted((p for p in pairs if p[1]==aid),key=lambda p:fingerprint(p[0]))[:cfg['diagnostic_pairs_per_target']]
        for q,a,value in rows:
            seq=queries[q]['heavy_sequence']
            if seq not in cdr or not cdr[seq]:raise ValueError('Diagnostic heavy chain lacks verified CDR3 labels')
            selected.append(dict(query_id=q,antigen_id=a,target=family,affinity=value,heavy_sequence=seq,cdr3_indices=json.dumps(cdr[seq])))
    write_csv(root/'diagnostic_selection.csv',selected)
    write_json(root/'diagnostic_selection.json',dict(complete=True,selection='sequence hash; no affinity thresholds',
        pairs=len(selected),files_sha256={n:digest(binding/n) for n in ('heldout_antibodies.csv','measurements.sqlite','antigens.csv','domains.sqlite')},
        selection_sha256=digest(root/'diagnostic_selection.csv')))


def ranking_audit(root,rows):
    cfg=settings(root);path=cfg.get('verified_affinity_records')
    if not path:
        write_json(root/'ranking_audit.json',dict(feasible=False,reason='no_independent_curator_verified_affinity_table'));return
    retained,pairs=affinity_pairs(read_csv(path),rows)
    write_csv(root/'affinities.csv',retained);write_csv(root/'ranking_pairs.csv',pairs)
    groups={s:len({r['biological_group'] for r in retained if r['split']==s}) for s in ('train','val','test')}
    feasible=all(any(p['split']==s for p in pairs) for s in groups) and all(groups[s]>=cfg['minimum_groups'] for s in ('val','test'))
    write_json(root/'ranking_audit.json',dict(feasible=feasible,reason='verified_pairs_available' if feasible else 'insufficient_independent_assay_groups_or_pairs',
        groups=groups,source_sha256=digest(path),pairs=dict(Counter(p['split'] for p in pairs)),
        files_sha256={n:digest(root/n) for n in ('affinities.csv','ranking_pairs.csv')},smoke_only=bool(cfg.get('smoke'))))


def smoke_rows(root):
    cfg=settings(root);old=Path('artifacts/experiments/affinity_improvement_smoke_v1')
    source=read_csv(old/'pool.csv');regions=read_json(old/'regions.json')['sequences'];rows=[];measurements=[]
    for split in ('train','val','test'):
        for i,base in enumerate([r for r in source if r['split']==split]):
            for weak in (False,True):
                row=dict(base,id=f'smoke_{split}_{i}_{int(weak)}',source_split=split,biological_group=f'smoke_{split}_{i}')
                if weak:
                    seq=row['resolved_H_seq'];index=next(j for j,label in enumerate(regions[fingerprint(seq)]['labels']) if label=='cdr3')
                    row['resolved_H_seq']=seq[:index]+('A' if seq[index]!='A' else 'G')+seq[index+1:]
                row['example_target_chains']='heavy;light';rows.append(row)
                measurements.append(dict(id=row['id'],target_id=f'{split}_{i}',parent_id=f'{split}_{i}',assay_id=f'{split}_{i}',
                    assay_provenance_url='https://example.invalid/synthetic-smoke',measurement_type='Kd',construct_format='smoke',
                    operator='=',log10_kd_nm=2 if weak else 0,log10_kd_sd=.01,replicate_count=3))
    path=root/'synthetic_affinities.csv';write_csv(path,measurements)
    if str(path)!=cfg['verified_affinity_records']:raise ValueError('Smoke affinity path differs')
    return select_examples(rows)


def finalize(root,rows,audit):
    cfg=settings(root);rows=select_examples(rows)
    for field in ('ab_ag_cluster','biological_group'):
        memberships=defaultdict(set)
        for r in rows:memberships[r[field]].add(r['split'])
        if any(len(s)>1 for s in memberships.values()):raise ValueError('Biological group crosses splits')
    if any(r['source_split']!='train' for r in rows if r['split']=='train'):raise ValueError('Old holdout entered training')
    write_csv(root/'records.csv',rows)
    try:
        for seed in cfg['seeds']:freeze(root/'schedules'/f'{seed}.json',dict(records_sha256=digest(root/'records.csv'),items=schedule(rows,cfg['updates'],seed)))
    except ValueError as error:
        audit.update(feasible=False,reason='training_schedule_infeasible',detail=str(error))
    ranking_audit(root,rows)
    audit.update(files_sha256={'records.csv':digest(root/'records.csv')},smoke_only=bool(cfg.get('smoke')))
    write_json(root/'cohort_audit.json',audit)


def audit(root):
    cfg=settings(root)
    if cfg.get('smoke'):
        return finalize(root,smoke_rows(root),dict(feasible=True,reason='synthetic_plumbing_only',candidates=[]))
    old=Path(cfg['reference_root']);prior=read_json(old/'input_manifest.json')
    for path,value in prior['source_sha256'].items():
        if digest(path)!=value:raise ValueError('Original audited source changed: '+path)
    freeze(root/'data_manifest.json',dict(source_sha256=prior['source_sha256'],reference_protocol_sha256=digest(old/'protocol.json'),settings=cfg))
    original=[];source_splits=defaultdict(set);quarantine=Counter();eligible=Counter()
    for r in csv.DictReader(Path(cfg['source_records']).open()):
        source_splits[r['ab_ag_cluster']].add(r['split'])
        if len(r['resolved_ag_seq'])>max(cfg['length_limits']):quarantine['antigen_length']+=1;continue
        if r.get('has_buzz')=='True' or 'alpha' in r.get('source','').lower() or any(s.startswith('flab_') for s in r['source'].lower().split(';')):quarantine['source_quarantine']+=1;continue
        original.append(dict(r,source_split=r['split']));eligible[r['split']]+=1
    if any(len(v)>1 for v in source_splits.values()):raise ValueError('Original source group crosses splits')
    with sqlite3.connect(f'file:{(Path(cfg["external_root"])/"domains.sqlite").resolve()}?mode=ro',uri=True) as db:
        protected_chains={r[0] for r in db.execute("SELECT target FROM domains WHERE status='valid'")}
    protected_antigens={r['resolved_ag_seq'] for r in read_csv(Path(cfg['external_root'])/'antigens.csv')}
    ag_hits=search(cfg,root,'external_antigens',[r['resolved_ag_seq'] for r in original],protected_antigens,False)
    ab_hits=search(cfg,root,'external_antibodies',[r[f'resolved_{c}_seq'] for r in original for c in ('H','L')],protected_chains,True)
    rejected=[];safe=[]
    for r in original:
        reasons=[]
        if r['resolved_ag_seq'] in protected_antigens or fingerprint(r['resolved_ag_seq']) in ag_hits:reasons.append('external_antigen_homology')
        if any(r[f'resolved_{c}_seq'] in protected_chains or fingerprint(r[f'resolved_{c}_seq']) in ab_hits for c in ('H','L')):reasons.append('external_antibody_lineage')
        if reasons:rejected.extend(dict(id=r['id'],reason=x) for x in reasons)
        else:safe.append(r)
    protected=[r for r in csv.DictReader(Path(cfg['source_records']).open()) if r['split']!='train']
    protected.extend(r for r in read_csv(cfg['native_records']) if r['split']!='train')
    hits=search(cfg,root,'reserved_lineages',[r[f'resolved_{c}_seq'] for r in safe if r['split']=='train' for c in ('H','L')],
                [r[f'resolved_{c}_seq'] for r in protected for c in ('H','L')],True)
    safe2=[]
    for r in safe:
        if r['split']=='train' and any(fingerprint(r[f'resolved_{c}_seq']) in hits for c in ('H','L')):rejected.append(dict(id=r['id'],reason='reserved_antibody_lineage'))
        else:safe2.append(r)
    near=search(cfg,root,'development_lineages',[r[f'resolved_{c}_seq'] for r in safe2 if r['split']=='val' for c in ('H','L')],
                [r[f'resolved_{c}_seq'] for r in safe2 if r['split']=='test' for c in ('H','L')],True)
    removed={r['ab_ag_cluster'] for r in safe2 if r['split']=='val' and any(fingerprint(r[f'resolved_{c}_seq']) in near for c in ('H','L'))}
    safe=[r for r in safe2 if r['ab_ag_cluster'] not in removed]
    for r in safe2:
        if r['ab_ag_cluster'] in removed:rejected.append(dict(id=r['id'],reason='development_lineage_group'))
    write_csv(root/'exclusions.csv',rejected);seqs={r[f'resolved_{c}_seq'] for r in safe for c in ('H','L')}
    search(cfg,root,'all_lineages',seqs,seqs,True)
    hits=[line.split('\t')[:2] for line in (root/'searches/all_lineages/hits.tsv').read_text().splitlines()]
    full=components(safe,hits)
    for group in full:
        gid='short_'+fingerprint(';'.join(r['id'] for r in group))[:20]
        for r in group:r['full_biological_group']=gid
    write_csv(root/'component_membership.csv',[dict(id=r['id'],source_split=r['source_split'],ab_ag_cluster=r['ab_ag_cluster'],
        biological_group=r['full_biological_group'],antigen_length=len(r['resolved_ag_seq'])) for g in full for r in g])
    candidates=[];selected=None
    for limit in cfg['length_limits']:
        subset=[[r for r in g if len(r['resolved_ag_seq'])<=limit] for g in full];subset=[g for g in subset if g]
        allocated=allocate(subset,cfg);entry=dict(length_limit=limit,records=sum(map(len,subset)),components=len(subset),feasible=False)
        if allocated:
            for r in allocated:r['biological_group']=r['full_biological_group']
            allocated=select_examples(allocated)
            bounded=[]
            for split in ('train','val','test'):bounded.extend(balanced_select([r for r in allocated if r['split']==split],cfg['training_records'] if split=='train' else cfg['holdout_records']))
            bounded=select_examples(bounded);part={s:[r for r in bounded if r['split']==s] for s in ('train','val','test')}
            entry['counts']={s:dict(records=len(v),groups=len({r['biological_group'] for r in v}),heavy_examples=sum('heavy' in example_chains(r) for r in v)) for s,v in part.items()}
            entry['feasible']=(len(part['train'])>=cfg['minimum_training_records'] and len({r['resolved_ag_seq'] for r in part['train']})>=cfg['minimum_training_antigens'] and
                all(entry['counts'][s]['groups']>=cfg['minimum_groups'] and entry['counts'][s]['heavy_examples']>=cfg['minimum_heavy_examples'] for s in ('val','test')))
            if entry['feasible'] and selected is None:selected=(limit,bounded)
        if not entry['feasible']:entry['reason']='independent_components_or_capacity_insufficient'
        candidates.append(entry)
    result=dict(feasible=selected is not None,reason='requirements_satisfied' if selected else 'no_length_limit_supports_protected_panels',
        candidates=candidates,eligible_counts=dict(eligible),quarantine=dict(quarantine),excluded=dict(Counter(r['reason'] for r in rejected)),
        full_component_sizes=sorted(map(len,full),reverse=True),source_manifest_sha256=digest(root/'data_manifest.json'))
    if selected:
        result['selected_length_limit']=selected[0];finalize(root,selected[1],result)
    else:
        ranking_audit(root,[]);result['files_sha256']={n:digest(root/n) for n in ('component_membership.csv','exclusions.csv')};write_json(root/'cohort_audit.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['selection','audit']);p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root)
    (diagnostic_selection if args.job=='selection' else audit)(root)
