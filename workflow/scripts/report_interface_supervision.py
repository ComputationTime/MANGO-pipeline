"""Group-aware factorial effects with optimization-seed SD, no confirmation."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from affinity_study_common import digest,read_csv,read_json,write_csv,write_json
from interface_supervision_common import ARMS,average_precision


def stats(values):
    return dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,seed_points=[float(v) for v in values])


def grouped(values):
    bins=defaultdict(list)
    for group,value in values:
        if value is not None:bins[group].append(value)
    return float(np.mean([np.mean(v) for v in bins.values()])) if bins else None


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);p=read_json(root/'protocol.json');cfg=p['settings'];per_seed=[];contact_rows=[];fitmarkers={}
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_s{seed}';fit=read_json(folder/'complete.json');done=read_json(folder/'evaluation.json');fitmarkers[arm,seed]=fit
            if digest(folder/'final.pt')!=fit['checkpoint_sha256'] or done['checkpoint_sha256']!=fit['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
            if digest(folder/'native.csv')!=done['native_sha256'] or digest(folder/'external.csv')!=done['external_sha256']:raise ValueError('Scores changed')
            native=read_csv(folder/'native.csv');external=read_csv(folder/'external.csv')
            for path,value in done['contacts'].items():
                if digest(path)!=value:raise ValueError('Contact output changed')
            variants=sorted({r['variant'] for r in native})
            for variant in variants:
                selected=[r for r in native if r['variant']==variant]
                metrics=dict(heavy_nll=grouped([(r['group'],float(r['nll'])) for r in selected]),
                    cdr_nll=grouped([(r['group'],float(r['cdr_nll'])) for r in selected if r['cdr_nll']!='']))
                if arm!='iglm_ce':
                    values=[];global_values=[];prevalence=[]
                    from affinity_study_common import fingerprint
                    for row in selected:
                        path=folder/'contacts'/f'{fingerprint(row["id"])}_{variant}.npz';payload=np.load(path);y=payload['truth'];score=payload['logits']
                        rowaps=[average_precision(a,b) for a,b in zip(y,score) if a.any()]
                        ap=float(np.mean(rowaps)) if rowaps else None;allap=average_precision(y,score)
                        contact_rows.append(dict(arm=arm,seed=seed,id=row['id'],group=row['group'],variant=variant,
                            within_residue_ap=ap,global_ap=allap,prevalence=float(y.mean()),positive_residues=sum(a.any() for a in y)))
                        values.append((row['group'],ap));global_values.append((row['group'],allap));prevalence.append((row['group'],float(y.mean())))
                    metrics.update(contact_ap=grouped(values),global_contact_ap=grouped(global_values),contact_prevalence=grouped(prevalence))
                per_seed.append(dict(arm=arm,seed=seed,variant=variant,**metrics))
            correct=next(r for r in per_seed if (r['arm'],r['seed'],r['variant'])==(arm,seed,'correct'))
            rhos=[];contact_rhos=[]
            for target in ('HER2','PD1','TIGIT'):
                subset=[r for r in external if r['target']==target]
                rho=float(spearmanr([-float(r['nll']) for r in subset],[-float(r['affinity']) for r in subset]).statistic)
                correct['affinity_'+target]=rho;rhos.append(rho)
                if arm!='iglm_ce':
                    cr=float(spearmanr([float(r['contact_score']) for r in subset],[-float(r['affinity']) for r in subset]).statistic)
                    correct['contact_affinity_'+target]=cr;contact_rhos.append(cr)
            correct['affinity_macro']=float(np.mean(rhos))
            if contact_rhos:correct['contact_affinity_macro']=float(np.mean(contact_rhos))
    for seed in cfg['seeds']:
        markers=[fitmarkers[a,seed] for a in cfg['arms']]
        for key in ('shared_decoder_sha256','exposure_sha256','target_tokens','updates'):
            if len({x[key] for x in markers})!=1:raise ValueError('Matched study check failed: '+key)
        conditional=[fitmarkers[a,seed] for a in cfg['arms'] if a!='iglm_ce']
        for key in ('initial_sha256','parameters','generation_parameters'):
            if len({x[key] for x in conditional})!=1:raise ValueError('Factorial model mismatch: '+key)
        if abs(fitmarkers['iglm_ce',seed]['generation_parameters']/conditional[0]['generation_parameters']-1)>.001:raise ValueError('IgLM parameter mismatch')
    summary=[]
    metrics=['contact_ap','global_contact_ap','contact_prevalence','heavy_nll','cdr_nll','affinity_macro',
             'affinity_HER2','affinity_PD1','affinity_TIGIT','contact_affinity_macro']
    for arm in cfg['arms']:
        for variant in ('correct','scrambled'):
            values=[r for r in per_seed if r['arm']==arm and r['variant']==variant]
            if not values:continue
            for metric in metrics:
                points=[r[metric] for r in values if r.get(metric) is not None]
                if len(points)==len(cfg['seeds']):summary.append(dict(arm=arm,variant=variant,metric=metric,**stats(points)))
    lookup={(r['arm'],r['seed'],r['variant']):r for r in per_seed};contrasts=[]
    formulas={'geometry_without_aux':{'geom_ce':1,'seq_ce':-1},'contact_without_geometry':{'seq_contact':1,'seq_ce':-1},
              'geometry_with_aux':{'geom_contact':1,'seq_contact':-1},'contact_with_geometry':{'geom_contact':1,'geom_ce':-1},
              'interaction':{'geom_contact':1,'geom_ce':-1,'seq_contact':-1,'seq_ce':1}}
    for metric in ('contact_ap','heavy_nll','cdr_nll','affinity_macro','contact_affinity_macro'):
        for name,weights in formulas.items():
            points=[sum(w*lookup[a,seed,'correct'][metric] for a,w in weights.items()) for seed in cfg['seeds']]
            contrasts.append(dict(contrast=name,metric=metric,**stats(points)))
    for arm in cfg['arms']:
        if arm=='iglm_ce':continue
        points=[lookup[arm,s,'correct']['affinity_macro']-lookup['iglm_ce',s,'correct']['affinity_macro'] for s in cfg['seeds']]
        contrasts.append(dict(contrast=arm+'_minus_IgLM',metric='affinity_macro',**stats(points)))
    write_csv(root/'per_seed.csv',per_seed);write_csv(root/'contact_metrics.csv',contact_rows);write_csv(root/'summary.csv',summary);write_csv(root/'paired_contrasts.csv',contrasts)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,metric,label in zip(axes,['contact_ap','affinity_macro','heavy_nll'],['Within-residue contact AP','Macro affinity Spearman','Held-out heavy NLL']):
        selected=[r for r in summary if r['metric']==metric and r['variant']=='correct']
        for i,r in enumerate(selected):
            ax.errorbar(r['mean'],i,xerr=r['seed_sd'] or 0,fmt='o',capsize=4);ax.scatter(r['seed_points'],np.full(len(cfg['seeds']),i),s=13,alpha=.55)
        ax.set(yticks=range(len(selected)),yticklabels=[r['arm'] for r in selected],xlabel=label)
    fig.suptitle('Geometry × contact-backpropagation pilot; bars: ±1 optimization-seed SD')
    fig.savefig(root/'factorial_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,['contact_ap','affinity_macro']):
        selected=[r for r in contrasts if r['metric']==metric and r['contrast'] in formulas]
        for i,r in enumerate(selected):ax.errorbar(r['mean'],i,xerr=r['seed_sd'] or 0,fmt='o',capsize=4)
        ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(len(selected)),yticklabels=[r['contrast'] for r in selected],xlabel='Paired change in '+metric)
    fig.suptitle('Predeclared factorial contrasts; error bars are seed SD')
    fig.savefig(root/'factorial_contrasts.png',dpi=180);plt.close(fig)
    result=dict(complete=True,smoke_only=cfg['smoke'],fits=len(fitmarkers),summary=summary,contrasts=contrasts,
        parameters={a:dict(total=fitmarkers[a,cfg['seeds'][0]]['parameters'],generation=fitmarkers[a,cfg['seeds'][0]]['generation_parameters']) for a in cfg['arms']},
        exposure_and_initialization_verified=True,biological_ci_available=False,
        limitations=p['limitations']+['CE controls train contact probes with detached features; contact arms additionally update the generator.',
            'External geometry is antigen-only predicted structure; native geometry is experimental and canonicalized per antigen sequence.',
            'No affinity-ranking loss or affinity-head calibration is used. Contact-score correlations are a predeclared exploratory readout.'])
    write_json(root/'results_report.json',result)
    lines=['Geometry and contact-supervision factorial pilot',f'Fits completed: {len(fitmarkers)}. Smoke only: {cfg["smoke"]}.',
        'CE controls learn a contact probe on detached features; contact arms send the same contact loss into the generator.',
        'All bars are optimization-seed SD; no biological confidence intervals or significance claims.','',
        '| Arm | Contact AP | Macro affinity rho | Heavy NLL |','|---|---:|---:|---:|']
    for arm in cfg['arms']:
        values={r['metric']:r for r in summary if r['arm']==arm and r['variant']=='correct'}
        contact=f'{values["contact_ap"]["mean"]:.5f}' if 'contact_ap' in values else 'n/a'
        lines.append(f'| {arm} | {contact} | {values["affinity_macro"]["mean"]:.5f} | {values["heavy_nll"]["mean"]:.5f} |')
    lines+=['','Limitations:']+result['limitations'];(root/'findings.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(Path(a.root))
