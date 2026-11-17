"""Complete error-bar plots and paired contrasts, including missing/infeasible arms."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
from pathlib import Path
import math
import numpy as np

from affinity_study_common import read_json,read_csv,write_json,write_csv,digest
from controlled_antigen_common import PRIMARY,SECONDARY,score_arrays,crossed_interval


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr,spearmanr
    import pandas as pd
    root=Path(root);protocol=read_json(root/'protocol.json');settings=protocol['settings'];state=read_json(root/'state.json')
    audit=read_json(root/'cohort_audit.json') if (root/'cohort_audit.json').exists() else None
    tables=defaultdict(lambda:defaultdict(dict));runs=[]
    for path in sorted((root/'runs').glob('*/complete.json')):
        done=read_json(path);spec=done['spec']
        if digest(path.parent/'best.pt')!=done['checkpoint_sha256'] or digest(path.parent/'scores.csv')!=done['scores_sha256']:
            raise ValueError('Run artifact changed')
        rows=read_csv(path.parent/'scores.csv');tables[spec['phase'],spec['study']][spec['seed']][spec['arm']]=rows
        runs.append(dict(run=spec['id'],**{k:spec[k] for k in ('phase','study','arm','seed')},
            heavy_nll=done['development_heavy_nll'],parameters=done['parameters'],trainable_parameters=done['trainable_parameters'],
            optimizer_steps=done['optimizer_steps'],target_tokens=done['target_tokens'],forward_passes=done['forward_passes'],
            wall_seconds=done['wall_seconds'],initial_sha256=done['shared_initial_sha256'],
            membership_sha256=done['training_membership_sha256']))
    # The actual initialized shared decoder, membership, tokens and budgets must
    # agree within the causal comparison, not merely the declared settings.
    for phase in ('pilot','confirmation'):
        for study in ('primary','secondary'):
            for seed in sorted({r['seed'] for r in runs}):
                part=[r for r in runs if (r['phase'],r['study'],r['seed'])==(phase,study,seed)]
                for key in ('initial_sha256','optimizer_steps','target_tokens','membership_sha256'):
                    if len({r[key] for r in part})>1:raise ValueError(f'Matched-run invariant failed: {phase}/{study}/{seed}/{key}')
    write_csv(root/'run_metrics.csv',runs)
    means=[];comparisons=[];penalties=[]
    for (phase,study),table in tables.items():
        arms=sorted(set.intersection(*(set(v) for v in table.values())))
        for arm in arms:
            for region in ('full','cdr','cdr3'):
                try:
                    losses,tokens=score_arrays(table,[(arm,1.)],region=region)
                    interval=crossed_interval(losses,tokens,settings['bootstrap_repeats'],settings['seed'])
                except ValueError:continue
                means.append(dict(phase=phase,study=study,arm=arm,region=region,**interval))
            changed={}
            for seed,by_arm in table.items():
                scores=by_arm[arm];synthetic=[]
                correct={(r['id'],r['chain']):r for r in scores if r['variant']=='correct'}
                wrong=defaultdict(list)
                for r in scores:
                    if r['variant'].startswith('wrong'):wrong[r['id'],r['chain']].append(r)
                for key,base in correct.items():
                    if key not in wrong:continue
                    r=dict(base);r['loss_sum']=sum(float(v['loss_sum']) for v in wrong[key])/len(wrong[key])-float(base['loss_sum'])
                    synthetic.append(r)
                changed[seed]={arm:synthetic}
            if all(v[arm] for v in changed.values()):
                losses,tokens=score_arrays(changed,[(arm,1.)]);interval=crossed_interval(losses,tokens,settings['bootstrap_repeats'],settings['seed'])
                penalties.append(dict(phase=phase,study=study,arm=arm,**interval))
        for name,contrast in (PRIMARY if study=='primary' else SECONDARY).items():
            try:
                losses,tokens=score_arrays(table,contrast)
                interval=crossed_interval(losses,tokens,settings['bootstrap_repeats'],settings['seed'])
                comparisons.append(dict(phase=phase,study=study,contrast=name,status='available',**interval))
            except ValueError as error:
                comparisons.append(dict(phase=phase,study=study,contrast=name,status='unavailable',reason=str(error)))
        selected=[r for r in means if r['phase']==phase and r['study']==study and r['region']=='full']
        if selected:
            fig,ax=plt.subplots(figsize=(max(8,len(selected)*1.3),5))
            for x,r in enumerate(selected):
                if r['low'] is not None:ax.vlines(x,r['low'],r['high'],color='black',lw=2)
                ax.scatter(x,r['estimate'],color='black',zorder=3)
                points=r.get('seed_points',[])
                ax.scatter(x+np.linspace(-.12,.12,len(points)),points,alpha=.65,s=20)
            ax.set_xticks(range(len(selected)),[r['arm'] for r in selected],rotation=30,ha='right')
            ax.set(ylabel='Heavy-domain NLL',title=f'{phase} {study}: 95% crossed bootstrap; dots are seeds')
            fig.tight_layout();fig.savefig(root/f'{phase}_{study}_nll.png',dpi=180);plt.close(fig)
        if study=='secondary':
            fig,ax=plt.subplots(figsize=(8,5))
            for update in ('frozen','joint'):
                cells=[next((r for r in selected if r['arm']==f'{update}_{c}'),None) for c in ('correct','shuffled','constant')]
                if not all(cells):continue
                ax.plot(range(3),[r['estimate'] for r in cells],marker='o',label=update)
                for x,r in enumerate(cells):
                    if r['low'] is not None:ax.vlines(x,r['low'],r['high'])
            ax.set_xticks(range(3),['correct','shuffled','constant']);ax.set(ylabel='Heavy-domain NLL',title=f'{phase}: decoder update × antigen memory')
            ax.legend();fig.tight_layout();fig.savefig(root/f'{phase}_interaction.png',dpi=180);plt.close(fig)
    write_json(root/'uncertainty.json',dict(means=means,paired_contrasts=comparisons,wrong_antigen_penalties=penalties,
        method='paired crossed percentile bootstrap; seed and biological-component axes independently resampled',
        repeats=settings['bootstrap_repeats'],interval=.95))
    if comparisons:
        available=[r for r in comparisons if r['status']=='available']
        fig,ax=plt.subplots(figsize=(9,max(4,len(available)*.45)))
        for y,r in enumerate(available):
            if r['low'] is not None:ax.hlines(y,r['low'],r['high'],color='black',lw=2)
            ax.scatter(r['estimate'],y)
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(len(available)),[f'{r["phase"]}/{r["study"]}/{r["contrast"]}' for r in available])
        ax.set(xlabel='Paired NLL improvement (positive favors named effect)',title='95% crossed bootstrap intervals')
        fig.tight_layout();fig.savefig(root/'paired_difference_forest.png',dpi=180);plt.close(fig)
    if penalties:
        fig,ax=plt.subplots(figsize=(10,max(4,len(penalties)*.3)))
        for y,r in enumerate(penalties):
            if r['low'] is not None:ax.hlines(y,r['low'],r['high'],color='black')
            ax.scatter(r['estimate'],y)
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(len(penalties)),[f'{r["phase"]}/{r["study"]}/{r["arm"]}' for r in penalties])
        ax.set(xlabel='Wrong − correct antigen NLL',title='Held-out antigen interventions: 95% crossed bootstrap')
        fig.tight_layout();fig.savefig(root/'antigen_penalties.png',dpi=180);plt.close(fig)
    metrics=[];verifications=[]
    for category in ('amendment_v1','external'):
        for path in sorted((root/category).glob('*.complete.json')):
            record=read_json(path);run=path.name.removesuffix('.complete.json');scores=path.parent/f'{run}_pairs.csv.gz'
            if digest(scores)!=record['scores_sha256']:raise ValueError('External score hash changed')
            frame=pd.read_csv(scores)
            for item in record['metrics']:
                g=frame[frame.target==item['target']]
                r=float(pearsonr(g.perplexity,g.affinity).statistic);rho=float(spearmanr(g.perplexity,g.affinity).statistic)
                if item['pearson_r'] is not None and abs(item['pearson_r']-r)>1e-12:raise ValueError('Pearson crosscheck failed')
                if item['spearman_r'] is not None and abs(item['spearman_r']-rho)>1e-12:raise ValueError('Spearman crosscheck failed')
                verifications.append(dict(run=run,target=item['target'],category=category,scipy_verified=True))
                metrics.append(dict(category=category,**item))
                fig,ax=plt.subplots(figsize=(7,5));ax.hexbin(g.perplexity,g.affinity,gridsize=60,bins='log',mincnt=1)
                ax.set(xlabel='Heavy-domain perplexity',ylabel='AlphaSeq log10(Kd / nM)',title=f'{run}: {item["target"]}, r={r:.3f}, ρ={rho:.3f}')
                fig.tight_layout();fig.savefig(path.parent/f'{run}_{item["target"]}.png',dpi=150);plt.close(fig)
        part=[r for r in metrics if r['category']==category]
        if part:
            df=pd.DataFrame(part);df['arm']=df.run.str.replace(r'_s\d+$','',regex=True)
            fig,ax=plt.subplots(figsize=(max(8,df.arm.nunique()),5))
            for offset,(target,group) in enumerate(df.groupby('target')):
                grouped=group.groupby('arm').pearson_r.agg(['mean','std']);x=np.arange(len(grouped))+offset*.13
                ax.errorbar(x,grouped['mean'],yerr=grouped['std'].fillna(0),fmt='o',capsize=3,label=target)
            ax.set_xticks(np.arange(df.arm.nunique())+.13,sorted(df.arm.unique()),rotation=45,ha='right')
            ax.set(ylabel='Pearson r',title='Exploratory affinity correlation; error bars = ±1 training-seed SD')
            ax.legend();fig.tight_layout();fig.savefig(root/f'{category}_affinity_seeds.png',dpi=180);plt.close(fig)
    write_csv(root/'affinity_correlations.csv',metrics);write_json(root/'external_verification.json',dict(checks=verifications))
    result=dict(complete=not state['failed_stages'],smoke_only=bool(settings.get('smoke')),failed_stages=state['failed_stages'],
        protocol_sha256=digest(root/'protocol.json'),data_audit=audit,decisions=state['decisions'],runs=runs,
        uncertainty='uncertainty.json',affinity=metrics,
        limitations=['Pilot checkpoint-selection reuse is optimistic; confirmation panels are separate.',
            'Related source data and external OpenAlphaSeq have already been inspected in earlier work.',
            'Affinity biological confidence intervals are unavailable without defensible independent parent groups.',
            'NLL improvement does not establish improved binding affinity.'])
    write_json(root/'results_report.json',result)
    lines=['# Controlled antigen study v2','',f'Execution: {"complete" if result["complete"] else "incomplete"}.',
        f'Completed training runs: {len(runs)}.','',f'Training feasible: {audit.get("feasible") if audit else "audit failed"}.',
        f'Audit reason: {audit.get("reason","requirements satisfied") if audit else "missing"}.','',
        'Primary comparisons use scratch-trained native IgLM decoders with identical shared starting weights. '
        'Secondary adapters use a separately labeled scratch-trained source checkpoint. All training arms retain dropout.',
        '', 'Error bars resample biological components and paired seeds. See `uncertainty.json`, '
        '`paired_difference_forest.png`, NLL/interaction plots, and `run_metrics.csv` when training is feasible.',
        '', 'The repaired v1 evaluation is in `amendment_v1/`; original v1 artifacts are unchanged. '
        'Affinity coefficients are exploratory. Seed SD bars do not represent biological uncertainty.','']
    lines+=['- '+x for x in result['limitations']]
    if result['failed_stages']:lines+=['','Failed stages: '+', '.join(result['failed_stages'])]
    (root/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(**vars(p.parse_args()))
