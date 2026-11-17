"""All attempted arms, paired mechanisms, and explicitly scoped error bars."""
import argparse
from pathlib import Path
import numpy as np
from affinity_study_common import digest,read_csv,read_json,write_csv,write_json
from native_mechanism_common import arms,CONTRASTS,paired_arrays,summarize


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);cfg=read_json(root/'training_protocol.json')['settings'];seeds=cfg['seeds']
    audit=read_json(root/'cohort_audit.json') if (root/'cohort_audit.json').exists() else {'complete':False,'feasible':None}
    state=read_json(root/'state.json');tables={s:{} for s in seeds};runs=[];affinities={s:{} for s in seeds}
    for seed in seeds:
        for arm in arms():
            directory=root/'runs'/f'{arm}_s{seed}';marker=directory/'complete.json'
            if not marker.exists():continue
            done=read_json(marker)
            for name,value in done['files_sha256'].items():
                if digest(directory/name)!=value:raise ValueError('Completed run changed: '+str(directory))
            tables[seed][arm]=read_csv(directory/'scores.csv')
            runs.append(dict(arm=arm,seed=seed,optimizer_steps=done['optimizer_steps'],target_tokens=done['target_tokens'],
                parameters=done['parameters'],trainable_parameters=done['trainable_parameters'],
                forward_passes=done['forward_passes'],wall_seconds=done['wall_seconds']))
            marker=directory/'affinity_complete.json'
            if marker.exists():
                aff=read_json(marker)
                if digest(directory/'affinity_scores.csv')!=aff['scores_sha256'] or digest(directory/'final.pt')!=aff['checkpoint_sha256']:
                    raise ValueError('Affinity provenance changed')
                if digest(root/'affinity_selection.csv')!=aff['selection_sha256']:raise ValueError('Different affinity panels')
                affinities[seed][arm]={r['target']:r for r in aff['metrics']}
    comparisons={};means={}
    for name,contrast in CONTRASTS.items():
        try:
            losses,counts=paired_arrays(tables,contrast)
            comparisons[name]=summarize(losses,counts,cfg['bootstrap_repeats'],cfg['seed'])
        except ValueError as error:comparisons[name]=dict(available=False,reason=str(error))
    for arm in arms():
        try:
            losses,counts=paired_arrays(tables,{arm:1.})
            means[arm]=summarize(losses,counts,cfg['bootstrap_repeats'],cfg['seed'])
        except ValueError:continue
    affinity_rows=[];affinity_contrasts=[]
    for target in ('HER2','PD1','TIGIT'):
        for arm in arms():
            if not all(arm in affinities[s] and target in affinities[s][arm] for s in seeds):continue
            values=[affinities[s][arm][target]['spearman'] for s in seeds]
            pearson=[affinities[s][arm][target]['pearson_perplexity'] for s in seeds]
            if any(v is None for v in values+pearson):continue
            affinity_rows.append(dict(target=target,arm=arm,seeds=len(seeds),spearman=float(np.mean(values)),
                seed_sd=float(np.std(values,ddof=1)) if len(seeds)>1 else None,
                pearson_perplexity=float(np.mean(pearson)),biological_ci_available=False))
        for name,contrast in CONTRASTS.items():
            if not all(all(arm in affinities[s] and target in affinities[s][arm] for arm in contrast) for s in seeds):continue
            values=[-sum(c*affinities[s][a][target]['spearman'] for a,c in contrast.items()) for s in seeds]
            affinity_contrasts.append(dict(target=target,contrast=name,mean=float(np.mean(values)),
                seed_sd=float(np.std(values,ddof=1)) if len(seeds)>1 else None,seed_points=values,biological_ci_available=False))
    write_csv(root/'run_metrics.csv',runs);write_csv(root/'affinity_correlations.csv',affinity_rows)
    write_json(root/'uncertainty.json',dict(nll_means=means,nll_contrasts=comparisons,affinity_contrasts=affinity_contrasts,
        biological_interval_scope='component sampling conditional on realized optimization seeds',
        seed_errorbars='sample SD across matched training seeds; not a confidence interval',formal_inference=False))
    if means:
        final={k:v for k,v in means.items() if k not in ('source','wide_source')}
        fig,ax=plt.subplots(figsize=(10,6));names=list(final);y=np.arange(len(names));points=[final[n]['mean'] for n in names]
        ax.errorbar(points,y,xerr=[final[n]['seed_sd'] or 0 for n in names],fmt='o',capsize=3)
        for i,n in enumerate(names):ax.scatter(final[n]['seed_points'],np.full(len(seeds),i)+np.linspace(-.12,.12,len(seeds)),s=12,alpha=.6)
        ax.set(yticks=y,yticklabels=names,xlabel='Held-out heavy NLL (lower is better)',title='Points: seeds; error bars: ±1 training-seed SD')
        fig.tight_layout();fig.savefig(root/'nll_comparison.png',dpi=170);plt.close(fig)
    available={k:v for k,v in comparisons.items() if 'mean' in v}
    if available:
        fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True);names=list(available);y=np.arange(len(names))
        for i,n in enumerate(names):
            v=available[n];axes[0].errorbar(v['mean'],i,xerr=v['seed_sd'] or 0,fmt='o',capsize=3)
            axes[0].scatter(v['seed_points'],np.full(len(seeds),i)+np.linspace(-.12,.12,len(seeds)),s=10)
            if v['low'] is not None:axes[1].plot([v['low'],v['high']],[i,i],linewidth=2)
            axes[1].scatter(v['mean'],i,s=20)
        axes[0].set(yticks=y,yticklabels=names,title='Optimization variability: ±1 seed SD')
        axes[1].set(title='95% biological bootstrap interval\nconditional on realized seeds')
        for ax in axes:ax.axvline(0,color='grey',linestyle='--');ax.set_xlabel('NLL improvement (positive favors specified mechanism)')
        fig.tight_layout();fig.savefig(root/'paired_effects.png',dpi=170);plt.close(fig)
    if affinity_rows:
        fig,axes=plt.subplots(1,3,figsize=(15,6),sharex=True)
        for ax,target in zip(axes,('HER2','PD1','TIGIT')):
            values=[r for r in affinity_rows if r['target']==target and r['arm'] not in ('source','wide_source')]
            ax.errorbar([r['spearman'] for r in values],range(len(values)),xerr=[r['seed_sd'] or 0 for r in values],fmt='o',capsize=3)
            ax.set(yticks=range(len(values)),yticklabels=[r['arm'] for r in values],title=target,xlabel='Spearman ρ (higher is better)')
        fig.suptitle('Exploratory affinity correlations; bars are ±1 seed SD, not biological confidence intervals')
        fig.tight_layout();fig.savefig(root/'affinity_correlations.png',dpi=170);plt.close(fig)
    if audit.get('feasible') is False:
        capacity=read_json(root/'capacity_before_antibody_screen.json') if (root/'capacity_before_antibody_screen.json').exists() else {}
        fig,ax=plt.subplots(figsize=(8,4))
        values=[capacity.get('retained_records',0),audit.get('safe_records',0),cfg['minimum_training_records']+cfg['minimum_heavy_examples']]
        ax.bar(['After antigen protection','After all protections','Required record minimum'],values,color=['#6487a5','#427e69','#b36560'])
        for i,v in enumerate(values):ax.text(i,v,str(v),ha='center',va='bottom')
        ax.set(ylabel='Records',title='Data capacity: no scientific model fits permitted')
        fig.tight_layout();fig.savefig(root/'feasibility.png',dpi=170);plt.close(fig)
    status='data_infeasible' if audit.get('feasible') is False else 'incomplete' if state['failed_stages'] else 'complete'
    write_json(root/'results_report.json',dict(status=status,smoke_only=bool(cfg.get('smoke')),audit=audit,runs=runs,
        nll_contrasts=comparisons,affinity=affinity_rows,failed_stages=state['failed_stages'],
        verified_affinity_ranking='not run: independent curator-verified affinity labels unavailable',confirmed_advantages=[]))
    lines=['Native mechanism pilot',f'Execution status: {status}.',f'Completed fits: {len(runs)}.',
        f'Smoke only: {bool(cfg.get("smoke"))}.','',
        'Each causal contrast changes real antigen pairing, freezing, representation, or objective as explicitly specified.',
        'Seed SD describes optimization variability. Biological bootstrap intervals condition on the realized seeds.',
        'No statistical confirmation tests or affinity-generalization claims are made.',
        'Affinity ranking supervision is unavailable without independently verified, assay-compatible labels.']
    if audit.get('feasible') is False:lines += ['',f'Protected source: {audit.get("safe_records")} records in {audit.get("safe_components")} components.',
        'The feasibility gate blocked scientific training. Thresholds and prior holdout protection were retained.']
    for n,v in available.items():lines.append(f'{n}: {v["mean"]:+.6f} nats/token; seed SD={v["seed_sd"]}.')
    (root/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(**vars(p.parse_args()))
