"""Predeclared paired comparisons, seed bars and native-group uncertainty."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from affinity_study_common import digest, fingerprint, read_csv, read_json, write_csv, write_json
from interface_supervision_common import average_precision


def statistics(points):
    return dict(mean=float(np.mean(points)),seed_sd=float(np.std(points,ddof=1)) if len(points)>1 else None,
                seed_points=[float(x) for x in points])


def group_mean(rows, metric):
    groups=defaultdict(list)
    for row in rows:
        if row.get(metric) is not None and row.get(metric)!='':
            groups[row['group']].append(float(row[metric]))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def native_metrics(payload):
    truth=payload['truth']; contact=payload['logits']; attention=payload['attention']
    selected=truth.any(-1)
    y=truth[selected]; a=attention[selected]
    if not len(y):
        raise ValueError('Primary native evaluation requires contacting residues')
    if not np.isfinite(contact).all() or not np.isfinite(attention).all():
        raise ValueError('Nonfinite native scores')
    if not np.allclose(attention.sum(-1),1.,atol=2e-5):
        raise ValueError('Attention is not normalized over real antigen residues')
    return dict(contact_ap=float(np.mean([average_precision(t,s) for t,s in zip(y,contact[selected])])),
                attention_ap=float(np.mean([average_precision(t,s) for t,s in zip(y,a)])),
                attention_enrichment=float(np.mean((a*y).sum(-1)/y.mean(-1))),
                attention_entropy=float(np.mean(-(a*np.log(np.maximum(a,1e-30))).sum(-1)/np.log(a.shape[-1]))))


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root); protocol=read_json(root/'protocol.json'); cfg=protocol['settings']
    fits={}; per_seed=[]; per_record=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_s{seed}'
            fit=read_json(folder/'complete.json'); done=read_json(folder/'evaluation.json')
            fits[arm,seed]=fit
            if digest(folder/'final.pt')!=fit['checkpoint_sha256'] or done['checkpoint_sha256']!=fit['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            for name in ('native','external','generation'):
                extension='json' if name=='generation' else 'csv'
                if digest(folder/f'{name}.{extension}')!=done[name+'_sha256']:
                    raise ValueError('Evaluation output changed')
            for path,expected in done['contacts'].items():
                if digest(path)!=expected:raise ValueError('Native contact output changed')
            native=read_csv(folder/'native.csv'); external=read_csv(folder/'external.csv')
            for variant in sorted({r['variant'] for r in native}):
                records=[]
                for row in [r for r in native if r['variant']==variant]:
                    record=dict(arm=arm,seed=seed,id=row['id'],group=row['group'],variant=variant,
                                heavy_nll=float(row['nll']),cdr_nll=float(row['cdr_nll']))
                    if arm!='iglm':
                        path=folder/'contacts'/f'{fingerprint(row["id"])}_{variant}.npz'
                        record.update(native_metrics(np.load(path)))
                    records.append(record);per_record.append(record)
                metrics={m:group_mean(records,m) for m in records[0] if m not in ('arm','seed','id','group','variant')}
                rhos=[]
                for target in ('HER2','PD1','TIGIT'):
                    chosen=[r for r in external if r['variant']==variant and r['target']==target]
                    rho=float(spearmanr([-float(r['nll']) for r in chosen],[-float(r['affinity']) for r in chosen]).statistic)
                    metrics['affinity_'+target]=rho;rhos.append(rho)
                metrics['affinity_macro']=float(np.mean(rhos))
                per_seed.append(dict(arm=arm,seed=seed,variant=variant,**metrics))
    for seed in cfg['seeds']:
        markers=[fits[a,seed] for a in cfg['arms']]
        for key in ('shared_decoder_sha256','exposure_sha256','target_tokens','updates'):
            if len({m[key] for m in markers})!=1:raise ValueError('Unmatched '+key)
        conditional=[fits[a,seed] for a in cfg['arms'] if a!='iglm']
        for key in ('initial_sha256','parameters'):
            if len({m[key] for m in conditional})!=1:raise ValueError('Conditional mismatch: '+key)
        if abs(fits['iglm',seed]['parameters']/conditional[0]['parameters']-1)>.001:
            raise ValueError('IgLM parameter mismatch')
    summary=[]
    metrics=['affinity_macro','affinity_HER2','affinity_PD1','affinity_TIGIT','heavy_nll','cdr_nll',
             'attention_ap','attention_enrichment','attention_entropy','contact_ap']
    lookup={(r['arm'],r['seed'],r['variant']):r for r in per_seed}
    for arm in cfg['arms']:
        for variant in sorted({r['variant'] for r in per_seed if r['arm']==arm}):
            for metric in metrics:
                points=[lookup[arm,s,variant].get(metric) for s in cfg['seeds']]
                if all(x is not None for x in points):
                    summary.append(dict(arm=arm,variant=variant,metric=metric,**statistics(points)))
    formulas={
        'shared_tag_minus_control': [('shared_tag','correct',1),('control','correct',-1)],
        'contact_bias_minus_control': [('contact_bias','correct',1),('control','correct',-1)],
        'contact_bias_minus_shared_tag': [('contact_bias','correct',1),('shared_tag','correct',-1)]}
    for arm in ('control','shared_tag','contact_bias'):
        formulas[arm+'_minus_IgLM']=[(arm,'correct',1),('iglm','correct',-1)]
        for variant in ('off','shuffled'):
            formulas[arm+'_correct_minus_'+variant]=[(arm,'correct',1),(arm,variant,-1)]
    contrasts=[]; group_intervals=[]
    for name,terms in formulas.items():
        for metric in metrics:
            if any(metric not in lookup[a,cfg['seeds'][0],v] for a,v,w in terms):continue
            points=[sum(w*lookup[a,s,v][metric] for a,v,w in terms) for s in cfg['seeds']]
            contrasts.append(dict(contrast=name,metric=metric,**statistics(points)))
            if metric.startswith('affinity'):continue
            # Cluster resampling is paired across all arms and examples. These
            # intervals condition on the fitted seeds and are not affinity CIs.
            grouped=defaultdict(list)
            for record in per_record:
                if record['arm'] in {a for a,v,w in terms} and metric in record:
                    grouped[record['arm'],record['seed'],record['variant'],record['group']].append(record[metric])
            groups=sorted({r['group'] for r in per_record})
            values=[np.mean([sum(w*np.mean(grouped[a,s,v,g]) for a,v,w in terms) for s in cfg['seeds']]) for g in groups]
            rng=np.random.default_rng(73519)
            boot=np.asarray(values)[rng.integers(0,len(values),(cfg['bootstrap_replicates'],len(values)))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            group_intervals.append(dict(contrast=name,metric=metric,mean=float(np.mean(values)),
                                        lower95=float(lo),upper95=float(hi),groups=len(groups),conditional_on_fitted_seeds=True))
    write_csv(root/'per_seed.csv',per_seed);write_csv(root/'per_record.csv',per_record)
    write_csv(root/'summary.csv',summary);write_csv(root/'paired_contrasts.csv',contrasts)
    write_csv(root/'native_group_intervals.csv',group_intervals)
    for filename,chosen,title in [
        ('routing_results',['affinity_macro','attention_ap','attention_enrichment'], 'Predicted interface routing'),
        ('native_results',['contact_ap','heavy_nll','cdr_nll'], 'Native contact and sequence outcomes')]:
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        for ax,metric in zip(axes,chosen):
            selected=[r for r in summary if r['variant']=='correct' and r['metric']==metric]
            for i,r in enumerate(selected):
                ax.errorbar(r['mean'],i,xerr=r['seed_sd'] or 0,fmt='o',capsize=4)
                ax.scatter(r['seed_points'],np.full(len(cfg['seeds']),i),s=14,alpha=.5)
            ax.set(yticks=range(len(selected)),yticklabels=[r['arm'] for r in selected],xlabel=metric)
        fig.suptitle(title+'; bars = ±1 optimization-seed SD')
        fig.savefig(root/(filename+'.png'),dpi=180);fig.savefig(root/(filename+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),layout='constrained')
    wanted=['shared_tag_minus_control','contact_bias_minus_control','contact_bias_minus_shared_tag',
            'shared_tag_correct_minus_shuffled','contact_bias_correct_minus_shuffled']
    for ax,metric in zip(axes,['affinity_macro','attention_ap','heavy_nll']):
        chosen=[r for r in contrasts if r['contrast'] in wanted and r['metric']==metric]
        for i,r in enumerate(chosen):ax.errorbar(r['mean'],i,xerr=r['seed_sd'] or 0,fmt='o',capsize=4)
        ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(len(chosen)),yticklabels=[r['contrast'] for r in chosen],xlabel='Paired change: '+metric)
    fig.suptitle('Routing comparisons and localization interventions; bars = seed SD')
    fig.savefig(root/'routing_contrasts.png',dpi=180);plt.close(fig)
    write_json(root/'results_report.json',dict(complete=True,smoke_only=cfg['smoke'],fits=len(fits),
        summary=summary,contrasts=contrasts,native_group_intervals=group_intervals,
        parameters={a:fits[a,cfg['seeds'][0]]['parameters'] for a in cfg['arms']},
        exposure_initialization_and_parameter_matching_verified=True,
        limitations=protocol['limitations'],affinity_biological_ci_available=False))
    lines=['# Predicted interface routing experiment','',
           f'Completed {len(fits)} fits. Smoke only: {cfg["smoke"]}.',
           'All ± values below are sample SD across paired optimization seeds.','',
           '| Arm | Macro affinity rho | Attention AP | Attention enrichment | Heavy NLL |',
           '|---|---:|---:|---:|---:|']
    for arm in cfg['arms']:
        values={r['metric']:r for r in summary if r['arm']==arm and r['variant']=='correct'}
        text=[]
        for metric in ('affinity_macro','attention_ap','attention_enrichment','heavy_nll'):
            v=values.get(metric)
            text.append('n/a' if v is None else f'{v["mean"]:.5f} ± {v["seed_sd"] or 0:.5f}')
        lines.append('| '+arm+' | '+' | '.join(text)+' |')
    lines+=['','Native-group bootstrap intervals condition on these fitted seeds. No biological affinity CI is claimed.',
            'Higher AP/rho is better; lower NLL is better. Attention enrichment 1 is the uniform-attention reference.',
            '','## Limits','']+['- '+x for x in protocol['limitations']]
    (root/'findings.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(Path(parser.parse_args().root))
