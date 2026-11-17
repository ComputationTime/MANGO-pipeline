"""Completion-only study report; seed and biological uncertainty stay separate."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from affinity_study_common import digest, read_csv, read_json, write_csv, write_json


def affinity_confirmation(root,pairs,settings):
    """Paired group bootstrap of equal-target, within-assay Pearson improvement."""
    measurements=read_csv(root/'verified_affinities.csv')
    records={r['id']:r for r in read_csv(root/'pool.csv')}
    labels=[r for r in measurements if r['split']=='test']
    by_group=defaultdict(list)
    for r in labels:by_group[records[r['id']]['ab_ag_cluster']].append(r)
    groups=sorted(by_group);scored=[]
    for candidate,baseline in pairs:
        maps=[]
        for run in (candidate,baseline):
            maps.append({r['id']:math.exp(float(r['nll'])) for r in read_csv(root/'runs'/run/'scores.csv')
                         if r['chain']=='heavy' and r['variant']=='correct'})
        scored.append(maps)
    def statistic(draw):
        diffs=[]
        for candidate,baseline in scored:
            cells=defaultdict(list)
            for group in draw:
                for r in by_group[group]:
                    if r['id'] in candidate and r['id'] in baseline:
                        cells[r['resolved_ag_seq'],r['assay_id']].append((candidate[r['id']],baseline[r['id']],float(r['log10_kd_nm'])))
            targets=defaultdict(list)
            for (target,assay),values in cells.items():
                if len(values)<3:continue
                a,b,y=np.asarray(values).T
                if min(np.ptp(a),np.ptp(b),np.ptp(y))<=0:continue
                targets[target].append(float(np.corrcoef(a,y)[0,1]-np.corrcoef(b,y)[0,1]))
            if targets:diffs.append(float(np.mean([np.mean(v) for v in targets.values()])))
        return float(np.mean(diffs)) if len(diffs)==len(pairs) else None
    observed=statistic(groups)
    if len(groups)<settings['minimum_development_groups'] or observed is None:
        return dict(low=None,high=None,pvalue=1.,improvement=0.,groups=len(groups),reason='insufficient_affinity_groups')
    rng=np.random.default_rng(settings['seed']);values=[]
    for _ in range(settings['bootstrap_repeats']):
        value=statistic([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
        if value is not None:values.append(value)
    if len(values)<.95*settings['bootstrap_repeats']:
        return dict(low=None,high=None,pvalue=1.,improvement=observed,groups=len(groups),reason='degenerate_bootstrap')
    low,high=np.quantile(values,[.025,.975])
    return dict(low=float(low),high=float(high),improvement=observed,groups=len(groups),
                pvalue=float((1+sum(abs(v-observed)>=abs(observed) for v in values))/(len(values)+1)))


def report(root):
    root=Path(root);protocol=read_json(root/'protocol.json');state=read_json(root/'state.json')
    runs=[]
    for path in sorted((root/'runs').glob('*/complete.json')):
        record=read_json(path);spec=record['spec']
        if digest(path.parent/'best.pt')!=record['checkpoint_sha256'] or digest(path.parent/'scores.csv')!=record['scores_sha256']:
            raise ValueError('Completed run changed before final report')
        scores=read_csv(path.parent/'scores.csv')
        correct=[r for r in scores if r['chain']=='heavy' and r['variant']=='correct']
        wrong=[r for r in scores if r['chain']=='heavy' and r['variant'].startswith('wrong')]
        region_scores={}
        for region in ('cdr','cdr3'):
            count=sum(int(r[f'{region}_tokens']) for r in correct)
            region_scores[region+'_nll']=sum(float(r[f'{region}_sum']) for r in correct)/count if count else None
        wrong_penalty=(sum(float(r['loss_sum']) for r in wrong)/sum(int(r['ntokens']) for r in wrong)-
                       record['development_heavy_nll']) if wrong else None
        runs.append(dict(run=spec['id'],phase=spec['phase'],arm=spec['arm'],kind=spec['kind'],seed=spec['seed'],
            evaluation_split=record['evaluation_split'],development_heavy_nll=record['development_heavy_nll'],
            development_affinity=record['development_affinity'],epochs=record['epochs'],
            optimizer_steps=record['optimizer_steps'],target_tokens=record['target_tokens'],
            forward_passes=record['forward_passes_training_and_validation'],wall_seconds=record['wall_seconds'],
            trainable_parameters=record['trainable_parameters'],parameters=record['parameters'],
            wrong_antigen_nll_penalty=wrong_penalty,**region_scores))
    write_csv(root/'run_comparisons.csv',runs)
    external=root/'external/correlations.csv'
    if external.exists():
        # A separate implementation checks all exported correlations directly
        # from matched pairs; missing SciPy is not replaced with invented values.
        checks=[]
        for path in (root/'external').glob('*_pairs.csv.gz'):
            import pandas as pd
            frame=pd.read_csv(path);name=path.name.removesuffix('_pairs.csv.gz')
            expected=[r for r in read_csv(external) if r['run']==name]
            for item in expected:
                group=frame[frame.target==item['target']]
                x=group.perplexity.to_numpy();y=group.affinity.to_numpy()
                a=x-x.mean();b=y-y.mean();r=float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
                if abs(r-float(item['pearson_r']))>1e-12:raise ValueError('External Pearson implementation mismatch')
                checks.append(dict(run=name,target=item['target'],pearson_r=r,pairs=len(group)))
        write_json(root/'external/verification.json',dict(complete=True,checks=checks))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        frame=pd.DataFrame(read_csv(external));frame['pearson_r']=frame.pearson_r.astype(float)
        fig,ax=plt.subplots(figsize=(max(8,len(frame.run.unique())*.5),5))
        for target,part in frame.groupby('target'):
            ax.plot(part.run,part.pearson_r,marker='o',label=target)
        ax.set(ylabel='Pearson r',title='External binding affinity: exploratory evaluation');ax.tick_params(axis='x',rotation=90)
        ax.legend();fig.tight_layout();fig.savefig(root/'affinity_comparison.png',dpi=180);plt.close(fig)
    if runs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(max(9,len(runs)*.15),5))
        ax.scatter(range(len(runs)),[r['development_heavy_nll'] for r in runs])
        ax.set_xticks(range(len(runs)),[r['run'] for r in runs],rotation=90,fontsize=6)
        ax.set(ylabel='Heavy-domain NLL',title='Predeclared development comparisons');fig.tight_layout()
        fig.savefig(root/'development_nll.png',dpi=180);plt.close(fig)
    audit=read_json(root/'cohort_audit.json') if (root/'cohort_audit.json').exists() else None
    output=dict(complete=not state['failed_stages'],smoke_only=protocol['settings'].get('smoke',False),
        failed_stages=state['failed_stages'],jobs=state['jobs'],decisions=state['decisions'],runs=runs,
        protocol_sha256=digest(root/'protocol.json'),cohort_audit=audit,
        inference_limits=['Screening is exploratory and selects candidates, not confirmed improvements.',
            'Held-out source groups are biological units; seeds measure training variability.',
            'OpenAlphaSeq has already been inspected; stronger generalization claims need independent data.',
            'Sequence-modeling gains do not establish improved affinity or antigen specificity.',
            'Synthetic/tiny smoke outputs are plumbing checks only.'])
    write_json(root/'results_report.json',output)
    lines=['# Sequential affinity-improvement study','',
        'Execution: '+('complete' if output['complete'] else 'incomplete')+'.',
        f'Completed training runs: {len(runs)}.','',
        'Each intervention uses the fixed comparison protocol. No external affinity value selects a model.',
        '', '| Intervention | Screening outcome | Advance |', '|---|---|---|']
    for arm,decision in state['decisions'].items():
        if arm not in ('balance','diversity','decoder','ranking','cdr','study'):continue
        lines.append(f'| {arm} | {decision.get("reason",decision.get("status",""))} | {decision.get("advance",False)} |')
    lines+=['','Full run metrics: `run_comparisons.csv`. Decisions, seed differences and confirmation '
            'uncertainty: `results_report.json`. All logs are retained under `logs/` and `controller_logs/`.','']
    lines+=['- '+item for item in output['inference_limits']]
    if state['failed_stages']:lines+=['','Failed stages: '+', '.join(state['failed_stages'])]
    (root/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    report(**vars(p.parse_args()))
