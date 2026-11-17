"""Audit all eligible prior records and reserve whole independent components."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json

from affinity_study_common import digest,freeze,read_csv,read_json,write_csv,write_json,fingerprint,row_key
from prepare_affinity_study import search
from prepare_abrank_transfer import UnionFind
from prepare_asd_transfer import select_examples
from chain_examples import example_chains


def components(rows,hits):
    uf=UnionFind()
    for r in rows:
        key='row:'+r['id'];uf.union(key,'source:'+r['ab_ag_cluster'])
        uf.union(key,'antigen:'+fingerprint(r['resolved_ag_seq']))
        for f in r['cath_families'].split(';'):
            if f:uf.union(key,'family:'+f)
        for c in ('H','L'):
            if r[f'resolved_{c}_seq']:uf.union(key,'seq:'+fingerprint(r[f'resolved_{c}_seq']))
    for a,b in hits:uf.union('seq:'+a,'seq:'+b)
    groups=defaultdict(list)
    for r in rows:groups[uf.find('row:'+r['id'])].append(r)
    return [sorted(v,key=lambda r:r['id']) for k,v in sorted(groups.items())]


def allocate(groups,settings):
    required=settings['minimum_groups'];minimum=settings['minimum_heavy_examples']
    panels={'val':[],'test':[]};free=[]
    def examples(panel):return sum('heavy' in example_chains(r) for g in panel for r in g)
    def pick_panel():return min(panels,key=lambda s:(len(panels[s])>=required,examples(panels[s])>=minimum,len(panels[s]),examples(panels[s])))
    for g in sorted(groups,key=lambda g:(-len(g),row_key(g[0]))):
        if any(r['source_split']!='train' for r in g):panels[pick_panel()].append(g)
        else:free.append(g)
    free.sort(key=lambda g:(len(g),row_key(g[0])))
    while any(len(panels[s])<required or examples(panels[s])<minimum for s in panels):
        selected=None
        for i,g in enumerate(free):
            remaining=[r for j,group in enumerate(free) if j!=i for r in group]
            if len(remaining)>=settings['minimum_training_records'] and len({r['resolved_ag_seq'] for r in remaining})>=settings['minimum_training_antigens']:
                selected=i;break
        if selected is None:return None
        panels[pick_panel()].append(free.pop(selected))
    panels['train']=free
    result=[]
    for split,groups in panels.items():
        for group in groups:
            gid='controlled_'+fingerprint(';'.join(r['id'] for r in group))[:20]
            for r in group:result.append(dict(r,split=split,biological_group=gid))
    return result


def prepare(config):
    import yaml
    cfg=yaml.safe_load(Path(config).read_text());root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    reference=Path(cfg['reference_root']);old=read_json(reference/'cohort_audit.json')
    manifest=dict(settings=cfg,source_pool_sha256=digest(reference/'pool.csv'),reference_audit_sha256=digest(reference/'cohort_audit.json'))
    freeze(root/'data_manifest.json',manifest)
    if (root/'cohort_audit.json').exists():
        old_audit=read_json(root/'cohort_audit.json')
        for name,value in old_audit.get('files_sha256',{}).items():
            if digest(root/name)!=value:raise ValueError('Frozen controlled-study data changed')
        return
    if digest(reference/'pool.csv')!=old['files_sha256']['pool.csv']:raise ValueError('Reference pool changed')
    if cfg.get('smoke'):
        rows=read_csv('artifacts/experiments/affinity_improvement_smoke_v1/pool.csv')
        rows=[dict(r,source_split=r['split'],biological_group=r['ab_ag_cluster']) for r in rows]
    else:
        prior=read_json(reference/'input_manifest.json');original=prior['settings']
        for path,value in prior['source_sha256'].items():
            if digest(path)!=value:raise ValueError('Original audited source changed')
        counts=Counter()
        for r in read_csv(original['source_records']):
            if len(r['resolved_ag_seq'])<=original['max_antigen_tokens'] and r.get('has_buzz')!='True' and 'alpha' not in r.get('source','').lower() and not any(s.startswith('flab_') for s in r['source'].lower().split(';')):counts[r['split']]+=1
        # The old audit covered the entire eligible pool only when its bounds
        # were nonbinding. Never silently reuse a truncated discovery cohort.
        if counts['train']>original['candidate_records'] or any(counts[s]>400 for s in ('val','test')):
            write_json(root/'cohort_audit.json',dict(complete=True,feasible=False,reason='full_source_audit_required',eligible_counts=dict(counts)));return
        candidates=[dict(r,source_split=r['split']) for r in read_csv(reference/'pool.csv')]
        seqs={r[f'resolved_{c}_seq'] for r in candidates for c in ('H','L')}
        search(cfg,root,'all_antibody_lineages',seqs,seqs,True)
        path=root/'searches/all_antibody_lineages/hits.tsv'
        hits=[line.split('\t')[:2] for line in path.read_text().splitlines()]
        groups=components(candidates,hits)
        write_csv(root/'component_membership.csv',[dict(id=r['id'],source_split=r['source_split'],
            ab_ag_cluster=r['ab_ag_cluster'],cath_families=r['cath_families'],component=i)
            for i,g in enumerate(groups) for r in g])
        component_files={str(p.relative_to(root)):digest(p) for p in (root/'component_membership.csv',path)}
        rows=allocate(groups,cfg)
        if rows is None:
            write_json(root/'cohort_audit.json',dict(complete=True,feasible=False,reason='insufficient_independent_components',
                components=len(groups),required_holdout_components=2*cfg['minimum_groups'],
                component_sizes=sorted([len(g) for g in groups],reverse=True),eligible_counts=dict(counts),
                reference_exclusions=old['excluded'],source_pool_records=len(candidates),files_sha256=component_files)
            );return
    rows=select_examples(rows)
    splits={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    if any(r['source_split']!='train' for r in splits['train']):raise ValueError('Prior holdout entered training')
    for key in ('ab_ag_cluster','biological_group'):
        mapping=defaultdict(set)
        for r in rows:mapping[r[key]].add(r['split'])
        if any(len(v)>1 for v in mapping.values()):raise ValueError('Group crosses splits')
    if not cfg.get('smoke'):
        if any(len({r['biological_group'] for r in splits[s]})<cfg['minimum_groups'] or sum('heavy' in example_chains(r) for r in splits[s])<cfg['minimum_heavy_examples'] for s in ('val','test')):
            write_json(root/'cohort_audit.json',dict(complete=True,feasible=False,reason='post_deduplication_holdout_capacity'));return
    train=sorted(splits['train'],key=row_key)[:cfg['pilot_records']]
    rows=select_examples(train+splits['val']+splits['test'])
    write_csv(root/'records.csv',rows)
    feasible=cfg.get('smoke') or (len(train)>=cfg['minimum_training_records'] and len({r['resolved_ag_seq'] for r in train})>=cfg['minimum_training_antigens'])
    shuffle_feasible={}
    for chain in ('heavy','light'):
        members=[r for r in train if chain in example_chains(r)]
        shuffle_feasible[chain]=bool(members) and max(Counter(r['resolved_ag_seq'] for r in members).values())*2<=len(members)
    feasible=feasible and all(shuffle_feasible.values())
    write_json(root/'cohort_audit.json',dict(complete=True,feasible=bool(feasible),smoke_only=bool(cfg.get('smoke')),
        reason='requirements_satisfied' if feasible else 'training_capacity_or_chainwise_shuffle_infeasible',
        chainwise_shuffle_feasible=shuffle_feasible,
        counts={s:dict(records=len([r for r in rows if r['split']==s]),groups=len({r['biological_group'] for r in rows if r['split']==s}),
        heavy_examples=sum('heavy' in example_chains(r) for r in rows if r['split']==s)) for s in splits},
        files_sha256=dict({'records.csv':digest(root/'records.csv')},**({} if cfg.get('smoke') else component_files)),
        reference_pool_sha256=digest(reference/'pool.csv'),reference_exclusions=old['excluded'],
        interpretation='New development roles; original holdouts never trained. Earlier studies have already inspected related source data.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/controlled_antigen_study.yaml');prepare(**vars(p.parse_args()))
