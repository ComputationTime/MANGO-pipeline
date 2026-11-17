"""Region-specific affinity, antigen intervention and contact alignment summaries."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from affinity_study_common import digest,read_csv,read_json,write_csv,write_json
from epitope_diagnostics_common import REGIONS,pooling_weights,safe_rho,weighted_difference


def stats(values):
    values=[float(v) for v in values if v is not None and np.isfinite(float(v))]
    return dict(mean=float(np.mean(values)) if values else None,
                seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,seed_points=values)


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);p=read_json(root/'protocol.json');cfg=p['settings'];regions=read_json(root/'regions.json')['sequences']
    native_records={r['id']:r for r in read_csv(root/'native_records.csv')};contacts=read_json(root/'contacts.json')['records']
    correlations=[];effects=[];native_effects=[];pair_gaps=[];spatial=[];checks=[]
    pair=read_json(Path(cfg['reference_root'])/'ranking_pair.json')
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_s{seed}';marker=read_json(folder/'complete.json');checks.append(marker)
            for name,value in marker['files_sha256'].items():
                if digest(folder/name)!=value:raise ValueError('Diagnostic output changed')
            ext=read_csv(folder/'external.csv');bins=defaultdict(dict)
            for row in ext:bins[row['target'],row['condition']][row['query_id']]=row
            for (target,condition),values in bins.items():
                values=list(values.values());aff=[-float(v['affinity']) for v in values]
                for region in REGIONS:
                    if all(v[region]!='' for v in values):
                        correlations.append(dict(arm=arm,seed=seed,target=target,condition=condition,region=region,
                            rho=safe_rho([-float(v[region]) for v in values],aff),rows=len(values)))
            if arm.startswith('MANGO'):
                for target in ('HER2','PD1','TIGIT'):
                    correct=bins[target,'correct'];ids=sorted(correct);aff=[-float(correct[q]['affinity']) for q in ids]
                    for region in REGIONS:
                        if not all(correct[q][region]!='' for q in ids):continue
                        a=np.array([float(correct[q][region]) for q in ids]);none=np.array([float(bins[target,'none'][q][region]) for q in ids])
                        mean=np.array([float(bins[target,'mean'][q][region]) for q in ids])
                        wrong=np.mean([[float(bins[target,c][q][region]) for q in ids] for c in ('wrong_0','wrong_1')],axis=0)
                        residual=wrong-a
                        effects.append(dict(arm=arm,seed=seed,target=target,region=region,
                            rank_agreement_none=safe_rho(a,none),rank_agreement_mean=safe_rho(a,mean),
                            correct_rho=safe_rho(-a,aff),none_rho=safe_rho(-none,aff),mean_rho=safe_rho(-mean,aff),
                            specificity_rho=safe_rho(residual,aff),
                            score_change_sd=float(np.std(a-none,ddof=1)),score_change_mean=float(np.mean(a-none)),
                            wrong_minus_correct_mean=float(np.mean(wrong-a))))
            native=read_csv(folder/'native.csv');lookup={(r['id'],r['condition']):r for r in native}
            conditions=sorted({r['condition'] for r in native})
            for condition in conditions:
                if (pair['strong']['id'],condition) not in lookup:continue
                strong=lookup[pair['strong']['id'],condition];weak=lookup[pair['weak']['id'],condition]
                for region in (*REGIONS,'paratope'):
                    if strong.get(region,'')!='' and weak.get(region,'')!='':
                        pair_gaps.append(dict(arm=arm,seed=seed,condition=condition,region=region,
                                              gap=float(weak[region])-float(strong[region])))
            if arm.startswith('MANGO'):
                for row in native:
                    if row['split']!='test' or row['condition']=='correct':continue
                    correct=lookup[row['id'],'correct']
                    for region in (*REGIONS,'paratope'):
                        if row.get(region,'')!='' and correct.get(region,'')!='':
                            native_effects.append(dict(arm=arm,seed=seed,id=row['id'],group=row['group'],condition=row['condition'],region=region,
                                delta_nll=float(row[region])-float(correct[region])))
                occlusion=defaultdict(dict)
                for r in read_csv(folder/'occlusions.csv'):occlusion[r['id'],r['region']][int(r['token'])]=float(r['delta_nll'])
                for row in read_csv(folder/'attention.csv'):
                    geometry=contacts[row['id']];weights=pooling_weights(geometry['embedding_length'],32)
                    attention=np.array(json.loads(row['weights']));delta=np.array([occlusion[row['id'],row['region']][i] for i in range(32)])
                    for cutoff in cfg['contact_cutoffs']:
                        contact=geometry['contacts'][str(float(cutoff))]
                        mask=np.zeros(geometry['embedding_length']);mask[contact['epitope']]=1;fraction=weights@mask
                        if len(contact['epitope'])<3 or len(contact['paratope'])<3:continue
                        baseline=float(fraction.mean());mass=float(np.dot(attention,fraction))
                        spatial.append(dict(arm=arm,seed=seed,id=row['id'],split=row['split'],group=row['group'],region=row['region'],cutoff=cutoff,
                            attention_epitope_mass=mass,uniform_epitope_mass=baseline,attention_enrichment=mass/baseline,
                            attention_entropy=float(-np.dot(attention,np.log(np.maximum(attention,1e-30)))/np.log(32)),
                            occlusion_epitope_contrast=weighted_difference(delta,fraction),
                            occlusion_epitope_rho=safe_rho(delta,fraction),
                            occlusion_abs_epitope_rho=safe_rho(abs(delta),fraction),
                            mean_abs_occlusion=float(np.mean(abs(delta))),
                            peak_epitope_fraction=float(fraction.max()),
                            query_contact_cutoff=cfg['primary_contact_cutoff'] if row['region']=='paratope' else ''))
    write_csv(root/'external_correlations.csv',correlations);write_csv(root/'external_effects.csv',effects)
    write_csv(root/'native_effects.csv',native_effects);write_csv(root/'training_pair_context_gaps.csv',pair_gaps)
    write_csv(root/'structural_alignment.csv',spatial)
    def summarize_rows(rows,keys,fields):
        bins=defaultdict(list)
        for row in rows:bins[tuple(row[k] for k in keys)].append(row)
        return [dict(zip(keys,key),**{f:stats([r[f] for r in values]) for f in fields}) for key,values in sorted(bins.items())]
    ext_summary=summarize_rows(effects,['arm','target','region'],['rank_agreement_none','rank_agreement_mean','correct_rho','none_rho','mean_rho','specificity_rho','score_change_sd','score_change_mean'])
    corr_summary=summarize_rows(correlations,['arm','target','condition','region'],['rho'])
    pair_summary=summarize_rows(pair_gaps,['arm','condition','region'],['gap'])
    # First average records within biological group, then groups within seed.
    def group_then_seed(rows,keys,fields):
        bins=defaultdict(list)
        for r in rows:bins[tuple(r[k] for k in [*keys,'seed','group'])].append(r)
        grouped=[]
        for key,values in bins.items():
            grouped.append(dict(zip([*keys,'seed','group'],key),**{f:float(np.mean([r[f] for r in values if r[f] is not None])) if any(r[f] is not None for r in values) else None for f in fields}))
        bins=defaultdict(list)
        for r in grouped:bins[tuple(r[k] for k in [*keys,'seed'])].append(r)
        seeds=[]
        for key,values in bins.items():
            seeds.append(dict(zip([*keys,'seed'],key),groups=len(values),**{f:float(np.mean([r[f] for r in values if r[f] is not None])) if any(r[f] is not None for r in values) else None for f in fields}))
        return summarize_rows(seeds,keys,fields)
    spatial_summary=group_then_seed([r for r in spatial if r['split']=='test'],['arm','region','cutoff'],
        ['attention_enrichment','attention_entropy','occlusion_epitope_contrast','occlusion_epitope_rho','mean_abs_occlusion','peak_epitope_fraction'])
    native_summary=group_then_seed(native_effects,['arm','condition','region'],['delta_nll'])
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,target in zip(axes,('HER2','PD1','TIGIT')):
        selected=[]
        for arm,condition,region,label in [('MANGO_CE','correct','full','MANGO full / correct'),('MANGO_CE','none','full','MANGO full / absent'),
            ('MANGO_CE','mean','full','MANGO full / mean'),('MANGO_CE','correct','cdr3','MANGO CDR3 / correct'),
            ('MANGO_CE','correct','framework','MANGO framework / correct'),('IgLM_CE','none','full','IgLM full')]:
            r=next(x for x in corr_summary if (x['arm'],x['target'],x['condition'],x['region'])==(arm,target,condition,region))
            selected.append((label,r['rho']))
        r=next(x for x in ext_summary if (x['arm'],x['target'],x['region'])==('MANGO_CE',target,'full'));selected.append(('MANGO specificity',r['specificity_rho']))
        for i,(label,v) in enumerate(selected):
            if v['mean'] is not None:ax.errorbar(v['mean'],i,xerr=v['seed_sd'] or 0,fmt='o',capsize=3)
        ax.set(yticks=range(len(selected)),yticklabels=[x[0] for x in selected],xlabel='Spearman rho',title=target)
    fig.suptitle('Frozen-checkpoint diagnostics; bars: ±1 optimization-seed SD')
    fig.savefig(root/'antigen_and_regions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    subset=[r for r in spatial_summary if r['cutoff']==cfg['primary_contact_cutoff'] and r['region'] in ('full','cdr3','paratope')]
    for i,r in enumerate(subset):
        for ax,key in zip(axes,['attention_enrichment','occlusion_epitope_contrast']):
            value=r[key]
            if value['mean'] is not None:
                ax.errorbar(value['mean'],i,xerr=value['seed_sd'] or 0,fmt='o',capsize=4)
                ax.scatter(value['seed_points'],np.full(len(value['seed_points']),i),s=12,alpha=.5)
    labels=[r['arm']+' / '+r['region'] for r in subset]
    axes[0].axvline(1,color='gray',ls='--');axes[1].axvline(0,color='gray',ls='--')
    axes[0].set(xlabel='Epitope attention enrichment versus uniform memory')
    axes[1].set(xlabel='Epitope-weighted minus non-epitope token occlusion ΔNLL')
    for ax in axes:ax.set(yticks=range(len(subset)),yticklabels=labels)
    fig.suptitle('Held-out native structures; group-balanced within seed; bars: seed SD')
    fig.savefig(root/'contact_focus.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');conditions=['correct','wrong','mean','zero','none']
    for arm in ('MANGO_CE','MANGO_rank'):
        values=[next(r['gap'] for r in pair_summary if (r['arm'],r['condition'],r['region'])==(arm,c,'full')) for c in conditions]
        ax.errorbar(range(len(conditions)),[v['mean'] for v in values],yerr=[v['seed_sd'] or 0 for v in values],marker='o',capsize=4,label=arm)
    ax.set(xticks=range(len(conditions)),xticklabels=conditions,ylabel='WT NLL minus matured NLL',title='In-sample RNase-A preference under antigen interventions; bars: seed SD');ax.legend()
    fig.savefig(root/'training_pair_context.png',dpi=170);plt.close(fig)
    result=dict(complete=True,new_fits=0,models=len(checks),smoke_only=cfg['smoke'],external=ext_summary,correlations=corr_summary,
                native=native_summary,training_pair=pair_summary,structural=spatial_summary,
                maximum_reference_parity=max(c['reference_parity_max'] for c in checks),
                maximum_memory_permutation_change=max(c['memory_permutation_max'] for c in checks),
                maximum_attention_instrumentation_change=max(c['attention_instrumentation_parity_max'] for c in checks),
                geometry=read_json(root/'prepared.json'),biological_ci_available=False)
    write_json(root/'results_report.json',result)
    lines=['Epitope/paratope diagnostics — inference only',f'Models: {len(checks)}. New fits: 0. Smoke only: {cfg["smoke"]}.',
           'Attention maps are descriptive. All token occlusions were chosen without structural labels; contacts were joined afterward.',
           'Error bars are SD across optimization seeds, not biological confidence intervals.']
    for arm in ('MANGO_CE','MANGO_rank'):
        for region in ('full','cdr3','framework'):
            selected=[r for r in ext_summary if r['arm']==arm and r['region']==region]
            for r in selected:
                lines.append(f'{arm} {r["target"]} {region}: affinity rho correct={r["correct_rho"]["mean"]:.5f}, absent={r["none_rho"]["mean"]:.5f}; score rank agreement={r["rank_agreement_none"]["mean"]:.5f}; specificity rho={r["specificity_rho"]["mean"]:.5f}.')
    (root/'findings.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(Path(a.root))
