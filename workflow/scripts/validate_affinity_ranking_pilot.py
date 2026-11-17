"""Independent SciPy and exposure checks after all fixed-budget fits finish."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr,pearsonr


def read(path):
    return json.loads(Path(path).read_text())


def csvrows(path):
    with open(path,newline='') as handle:return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    root=Path(root);protocol=read(root/'training_protocol.json');cfg=protocol['settings']
    if cfg['smoke']:raise ValueError('Independent scientific validation expects the full pilot')
    report=read(root/'results_report.json');arms=['IgLM_CE','MANGO_CE','IgLM_rank','MANGO_rank']
    for field in ('inputs_sha256','implementation_sha256'):
        for name,value in protocol[field].items():
            if sha(name)!=value:raise ValueError('Frozen input changed: '+name)
    rows=csvrows(root/'records.csv');byid={r['id']:r for r in rows}
    for key in ('ab_ag_cluster','biological_group'):
        assert not ({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='test'})
    assert all(r['source_split']=='train' for r in rows if r['split']=='train')
    pair=read(root/'ranking_pair.json');coefficients={};native={};differences=[];maximum_objective_error=0.
    for seed in cfg['seeds']:
        schedule=read(root/'schedules'/f'adapt_{seed}.json')['items']
        exposure=[];tokens=0
        for batch in schedule:
            values=[(x['id'],x['chain']) for x in batch]+[(pair[k]['id'],'heavy') for k in ('strong','weak')]
            exposure.extend(values)
            tokens+=sum(len(byid[rid]['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'])+1 for rid,chain in values)
        expected_exposure=hashlib.sha256(json.dumps(exposure).encode()).hexdigest()
        for arm in arms:
            directory=root/'runs'/f'{arm}_s{seed}';fit=read(directory/'complete.json');evaluation=read(directory/'evaluation.json')
            assert fit['checkpoint_sha256']==sha(directory/'final.pt')==evaluation['checkpoint_sha256']
            assert fit['exposure_sha256']==expected_exposure and fit['target_tokens']==tokens
            assert evaluation['external_scores_sha256']==sha(directory/'external_scores.csv')
            assert evaluation['native_scores_sha256']==sha(directory/'native_scores.csv')
            curve=csvrows(directory/'training_curve.csv');assert len(curve)==cfg['adaptation_updates']
            for r in curve:
                expected=float(r['nll'])+(cfg['ranking_weight']*float(r['ranking_loss']) if arm.endswith('_rank') else 0.)
                error=abs(float(r['objective'])-expected);maximum_objective_error=max(maximum_objective_error,error)
                assert error<1e-6
            scored=csvrows(directory/'external_scores.csv');target_rows=defaultdict(list)
            for r in scored:target_rows[r['target']].append(r)
            for target,values in target_rows.items():
                nll=np.array([float(r['nll']) for r in values]);kd=np.array([float(r['affinity']) for r in values])
                rho=float(spearmanr(-nll,-kd).statistic);pearson=float(pearsonr(np.exp(nll),kd).statistic)
                stored=next(r for r in evaluation['metrics'] if r['target']==target)
                differences.extend([abs(rho-stored['spearman']),abs(pearson-stored['pearson_perplexity'])])
                coefficients[arm,seed,target]=rho
            scored=csvrows(directory/'native_scores.csv')
            assert all(byid[r['id']]['split']=='test' for r in scored)
            nll=sum(float(r['loss_sum']) for r in scored)/sum(int(r['ntokens']) for r in scored)
            differences.append(abs(nll-evaluation['native_nll']));native[arm,seed]=nll
        for family in ('IgLM','MANGO'):
            a=read(root/'runs'/f'{family}_CE_s{seed}'/'complete.json')
            b=read(root/'runs'/f'{family}_rank_s{seed}'/'complete.json')
            for key in ('initial_sha256','parent_sha256','schedule_sha256','parameters'):assert a[key]==b[key]
            pre=root/'pretraining'/f'{family}_s{seed}'
            assert sha(pre/'final.pt')==a['parent_sha256']
            assert len(csvrows(pre/'training_curve.csv'))==cfg['pretraining_updates']
    for r in report['correlations']:
        targets=['HER2','PD1','TIGIT'] if r['target']=='macro' else [r['target']]
        values=[statistics.mean(coefficients[r['arm'],seed,t] for t in targets) for seed in cfg['seeds']]
        differences.extend([abs(statistics.mean(values)-r['mean']),abs(statistics.stdev(values)-r['seed_sd'])])
    primary=[]
    for seed in cfg['seeds']:
        primary.append(statistics.mean(coefficients['MANGO_rank',seed,t]-coefficients['MANGO_CE',seed,t]
                       -coefficients['IgLM_rank',seed,t]+coefficients['IgLM_CE',seed,t] for t in ['HER2','PD1','TIGIT']))
    stored=next(r for r in report['contrasts'] if r['contrast']=='interaction' and r['target']=='macro')
    differences.extend([abs(statistics.mean(primary)-stored['mean']),abs(statistics.stdev(primary)-stored['seed_sd'])])
    assert max(differences)<1e-12
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in report['contrasts'] if r['contrast']=='interaction']
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,r in enumerate(selected):
        ax.errorbar(r['mean'],i,xerr=r['seed_sd'],fmt='o',capsize=4,color='#356f9e')
        ax.scatter(r['seed_points'],np.full(len(cfg['seeds']),i)+np.linspace(-.08,.08,len(cfg['seeds'])),s=18,alpha=.7)
    ax.axvline(0,color='gray',ls='--')
    ax.set(yticks=range(len(selected)),yticklabels=[r['target'] for r in selected],
           xlabel='(MANGO rank - CE) - (IgLM rank - CE), Spearman rho',
           title='Ranking interaction; bars: ±1 paired optimization-seed SD')
    fig.savefig(root/'paired_interaction.png',dpi=180);plt.close(fig)
    result=dict(complete=True,models_validated=len(arms)*len(cfg['seeds']),
                scipy_coefficients_recomputed=len(coefficients)*2,
                maximum_numeric_discrepancy=max(differences),maximum_objective_roundoff=maximum_objective_error,
                exact_schedule_token_and_exposure_checks=True,all_prior_holdout_boundaries_verified=True,
                primary_interaction=statistics.mean(primary),paired_seed_sd=statistics.stdev(primary),
                validator_sha256=sha(__file__),report_sha256=sha(root/'results_report.json'),
                biological_ci_available=False)
    (root/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/affinity_ranking_pilot_v1')
    a=p.parse_args();main(a.root)
