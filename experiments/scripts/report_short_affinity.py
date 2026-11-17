"""Paired screening effects, honest infeasibility, and exploratory affinity diagnostics."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
from pathlib import Path
import re
import numpy as np

from affinity_study_common import read_json,read_csv,write_json,write_csv,digest
from controlled_antigen_common import score_arrays,crossed_interval
from short_affinity_common import ARMS

CONTRASTS={
 'conditional_vs_matched_iglm': [('U',1.),('C',-1.)],
 'real_pairing_ce': [('S',1.),('C',-1.)],
 'pair_objective': [('C',1.),('P',-1.)],
 'pair_objective_real_vs_shuffled': [('PS',1.),('P',-1.)],
 'pair_vs_matched_iglm': [('U',1.),('P',-1.)],
 'cdr_objective': [('C',1.),('C_CDR',-1.)],
 'cdr_iglm_objective': [('U',1.),('U_CDR',-1.)],
 'cdr_vs_matched_iglm': [('U_CDR',1.),('C_CDR',-1.)],
 'cdr_interaction': [('U_CDR',1.),('C_CDR',-1.),('U',-1.),('C',1.)],
 'learned_pooling': [('C',1.),('L',-1.)],
 'learned_pooling_shuffled': [('S',1.),('LS',-1.)],
 'learned_real_vs_shuffled': [('LS',1.),('L',-1.)],
 'learned_vs_matched_iglm': [('U_L',1.),('L',-1.)]}


def correlation(x,y,rank=True):
    from scipy.stats import pearsonr,spearmanr
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    value=float((spearmanr if rank else pearsonr)(x,y).statistic)
    return value if np.isfinite(value) else None


def diagnostic_report(root,cfg):
    metrics=[];seed_comparisons=[]
    for path in sorted((root/'diagnostics').glob('*.complete.json')):
        done=read_json(path);run=done['run'];scores=root/'diagnostics'/f'{run}.csv'
        if digest(scores)!=done['scores_sha256'] or digest(root/'diagnostic_selection.csv')!=done['selection_sha256']:raise ValueError('Diagnostic artifacts changed')
        rows=read_csv(scores);groups=defaultdict(list)
        for row in rows:groups[row['target']].append(row)
        for target,group in groups.items():
            for method in ('nll','cdr3_nll','residual'):
                x=[float(r[method])*(-1 if method!='residual' else 1) for r in group];y=[-float(r['affinity']) for r in group]
                metrics.append(dict(run=run,seed=int(run.rsplit('_s',1)[1]),family='MANGO' if '_base_' in run else 'IgLM',target=target,method=method,
                    pairs=len(group),spearman=correlation(x,y),pearson_perplexity=correlation([float(r['perplexity']) for r in group],[float(r['affinity']) for r in group],False),biological_ci_available=False))
    for target in sorted({r['target'] for r in metrics}):
        for method in ('cdr3_nll','residual'):
            differences=[]
            for seed in cfg['seeds']:
                def get(family,kind):return next((r['spearman'] for r in metrics if (r['target'],r['seed'],r['family'],r['method'])==(target,seed,family,kind)),None)
                candidate=get('MANGO',method);base=get('MANGO','nll');iglm=get('IgLM','nll')
                if candidate is not None and base is not None and iglm is not None:differences.append(dict(seed=seed,vs_mango=candidate-base,vs_iglm=candidate-iglm))
            flag=len(differences)==len(cfg['seeds']) and len(differences)>=3 and all(np.mean([d[k] for d in differences])>=cfg['correlation_improvement'] and sum(d[k]>0 for d in differences)>=2 for k in ('vs_mango','vs_iglm'))
            seed_comparisons.append(dict(target=target,method=method,seed_effects=differences,exploratory_flag=bool(flag),used_for_training_selection=False))
    return metrics,seed_comparisons


def affinity_contrasts(root,cfg):
    """Crossed component/seed bootstrap of within-assay rank differences."""
    from scipy.stats import spearmanr
    rows={}
    for seed in cfg['seeds']:
        rows[seed]={}
        for arm in ('C_Rank','U_Rank','C_AffCE','U_AffCE'):
            path=root/'runs'/f'{arm}_s{seed}'
            if (path/'complete.json').exists() and (path/'affinity_scores.csv').exists():rows[seed][arm]=read_csv(path/'affinity_scores.csv')
    if not all(len(v)==4 for v in rows.values()):return dict(status='unavailable',reason='incomplete_matched_affinity_arms')
    seeds=sorted(rows);reference=rows[seeds[0]]['C_Rank'];mapping={r['id']:i for i,r in enumerate(reference)}
    groups=sorted({r['biological_group'] for r in reference});indices={g:[i for i,r in enumerate(reference) if r['biological_group']==g] for g in groups}
    values={};assays=np.array([r['assay_id'] for r in reference]);y=np.array([-float(r['log10_kd_nm']) for r in reference])
    for seed in seeds:
        values[seed]={}
        for arm,part in rows[seed].items():
            if len(part)!=len(reference) or {r['id'] for r in part}!=set(mapping):raise ValueError('Unpaired affinity targets')
            part=sorted(part,key=lambda r:mapping[r['id']])
            for a,b in zip(reference,part):
                if any(a[k]!=b[k] for k in ('assay_id','biological_group','log10_kd_nm')):raise ValueError('Affinity metadata differ')
            values[seed][arm]=np.array([-float(r['nll']) for r in part])
    def macro(seed,arm,selected):
        result=[]
        for assay in sorted(set(assays[selected])):
            chosen=selected[assays[selected]==assay];r=correlation(values[seed][arm][chosen],y[chosen])
            if r is not None:result.append(r)
        return float(np.mean(result)) if result else None
    out={}
    for name,candidate,control in [('fair_affinity','C_Rank','U_Rank'),('conditional_rank_objective','C_Rank','C_AffCE'),('iglm_rank_objective','U_Rank','U_AffCE')]:
        all_indices=np.arange(len(reference));points=[]
        for seed in seeds:
            a=macro(seed,candidate,all_indices);b=macro(seed,control,all_indices)
            if a is None or b is None:break
            points.append(a-b)
        if len(points)!=len(seeds):out[name]=dict(status='unavailable',reason='undefined_within_assay_correlation');continue
        result=dict(status='available',estimate=float(np.mean(points)),seed_points=points,seeds=len(seeds),groups=len(groups),low=None,high=None)
        if len(groups)>=2 and len(seeds)>=2:
            rng=np.random.default_rng(cfg['seed']);draws=[]
            for _ in range(cfg['bootstrap_repeats']):
                chosen=np.concatenate([indices[groups[i]] for i in rng.integers(len(groups),size=len(groups))]);effects=[]
                for i in rng.integers(len(seeds),size=len(seeds)):
                    a=macro(seeds[i],candidate,chosen);b=macro(seeds[i],control,chosen)
                    if a is not None and b is not None:effects.append(a-b)
                if len(effects)==len(seeds):draws.append(np.mean(effects))
            result['valid_bootstrap_draws']=len(draws)
            if len(draws)>=.95*cfg['bootstrap_repeats']:result['low'],result['high']=map(float,np.quantile(draws,[.025,.975]))
        out[name]=result
    return out


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);protocol=read_json(root/'protocol.json');cfg=protocol['settings'];state=read_json(root/'state.json')
    audit=read_json(root/'cohort_audit.json') if (root/'cohort_audit.json').exists() else None
    ranking=read_json(root/'ranking_audit.json') if (root/'ranking_audit.json').exists() else None
    tables={s:{} for s in cfg['seeds']};runs=[]
    for path in sorted((root/'runs').glob('*/complete.json')):
        d=read_json(path);spec=d['spec'];directory=path.parent
        if digest(directory/'final.pt')!=d['checkpoint_sha256'] or digest(directory/'scores.csv')!=d['scores_sha256']:raise ValueError('Training output changed')
        if d['updates']!=cfg['updates']:raise ValueError('Unequal update budget')
        tables[spec['seed']][spec['arm']]=read_csv(directory/'scores.csv')
        runs.append(dict(run=spec['id'],arm=spec['arm'],seed=spec['seed'],parameters=d['parameters'],updates=d['updates'],target_tokens=d['target_tokens'],
            membership=d['training_membership_sha256'],shared_initial=d['shared_initial_sha256'],forward_passes=d['forward_passes'],wall_seconds=d['wall_seconds']))
    for seed in cfg['seeds']:
        all_runs=[r for r in runs if r['seed']==seed]
        if len({r['shared_initial'] for r in all_runs})>1:raise ValueError('Shared decoder initialization mismatch')
        for affinity in (False,True):
            subset=[r for r in all_runs if (ARMS[r['arm']]['test']==6)==affinity]
            for key in ('membership','target_tokens'):
                if len({r[key] for r in subset})>1:raise ValueError('Training exposures mismatch: '+key)
        for a,b in [('C','U'),('P','U'),('C_CDR','U_CDR'),('L','U_L'),('C_Rank','U_Rank'),('C_AffCE','U_AffCE')]:
            pair={r['arm']:r for r in all_runs if r['arm'] in (a,b)}
            if len(pair)==2 and abs(pair[a]['parameters']/pair[b]['parameters']-1)>.001:raise ValueError('Unfair parameter comparison')
    write_csv(root/'run_metrics.csv',runs);effects=[]
    for name,contrast in CONTRASTS.items():
        for region in ('full','cdr3'):
            try:
                losses,tokens=score_arrays(tables,contrast,region=region)
                interval=crossed_interval(losses,tokens,cfg['bootstrap_repeats'],cfg['seed'])
                effects.append(dict(contrast=name,region=region,status='available',**interval))
            except ValueError as e:effects.append(dict(contrast=name,region=region,status='unavailable',reason=str(e)))
    penalties={s:{} for s in cfg['seeds']}
    for seed,by_arm in tables.items():
        for arm,rows in by_arm.items():
            correct={(r['id'],r['chain']):r for r in rows if r['variant']=='correct'};wrong=defaultdict(list)
            for r in rows:
                if r['variant'].startswith('wrong'):wrong[r['id'],r['chain']].append(r)
            changed=[]
            for k,r in correct.items():
                if k in wrong:
                    copy=dict(r)
                    for column in ('loss_sum','cdr_sum','cdr3_sum'):copy[column]=np.mean([float(v[column]) for v in wrong[k]])-float(r[column])
                    changed.append(copy)
            penalties[seed][arm]=changed
    for name,contrast in [('pair_gap_gain',[('P',1.),('C',-1.)]),('cdr_gap_gain',[('C_CDR',1.),('C',-1.)]),('learned_gap_gain',[('L',1.),('C',-1.)])]:
        try:
            losses,tokens=score_arrays(penalties,contrast,region='cdr3');effects.append(dict(contrast=name,region='cdr3',status='available',**crossed_interval(losses,tokens,cfg['bootstrap_repeats'],cfg['seed'])))
        except ValueError as e:effects.append(dict(contrast=name,region='cdr3',status='unavailable',reason=str(e)))
    diagnostics,flags=diagnostic_report(root,cfg);affinity=affinity_contrasts(root,cfg) if ranking and ranking['feasible'] else dict(status='infeasible',reason=ranking['reason'] if ranking else 'audit_incomplete')
    decisions={}
    for test,main,gap,control in [(3,'pair_objective','pair_gap_gain','pair_objective_real_vs_shuffled'),(4,'cdr_objective','cdr_gap_gain','cdr_vs_matched_iglm'),(5,'learned_pooling','learned_gap_gain','learned_real_vs_shuffled')]:
        def effect(name,region='cdr3'):return next((e for e in effects if e['contrast']==name and e['region']==region and e['status']=='available'),None)
        a=effect(main);b=effect(gap);c=effect(control);full=effect(main,'full')
        available=all(x is not None for x in (a,b,c,full)) and len(cfg['seeds'])>=3
        flag=available and a['estimate']>=cfg['nll_improvement'] and sum(v>0 for v in a['seed_points'])>=2 and b['estimate']>=cfg['nll_improvement'] and c['estimate']>0 and full['estimate']>=-cfg['nll_tolerance']
        decisions[str(test)]=dict(exploratory_flag=bool(flag),complete_comparison=bool(available),confirmed=False)
    if isinstance(affinity,dict) and 'fair_affinity' in affinity:
        a=affinity['fair_affinity'];b=affinity['conditional_rank_objective']
        flag=all(v.get('status')=='available' and v['estimate']>=cfg['correlation_improvement'] and sum(p>0 for p in v['seed_points'])>=2 for v in (a,b)) and len(cfg['seeds'])>=3
        decisions['6']=dict(exploratory_flag=bool(flag),confirmed=False)
    available=[e for e in effects if e['status']=='available' and e['region']=='cdr3']
    if available:
        fig,ax=plt.subplots(figsize=(10,max(5,len(available)*.45)))
        for y,e in enumerate(available):
            if e['low'] is not None:ax.hlines(y,e['low'],e['high'],color='black')
            ax.scatter(e['estimate'],y,color='black');points=e.get('seed_points',[]);ax.scatter(points,y+np.linspace(-.1,.1,len(points)),s=15,alpha=.6)
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(len(available)),[e['contrast'] for e in available]);ax.set(xlabel='Paired CDR3 NLL improvement / gap change',title='Exploratory 95% crossed bootstrap; dots are seeds')
        fig.tight_layout();fig.savefig(root/'paired_effects.png',dpi=160);plt.close(fig)
    if diagnostics:
        fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
        for ax,target in zip(axes,sorted({r['target'] for r in diagnostics})):
            labels=[]
            for x,(family,method) in enumerate([('MANGO','nll'),('MANGO','cdr3_nll'),('MANGO','residual'),('IgLM','nll')]):
                values=[r['spearman'] for r in diagnostics if (r['target'],r['family'],r['method'])==(target,family,method) and r['spearman'] is not None]
                labels.append(f'{family}\n{method}')
                if values:ax.errorbar(x,np.mean(values),yerr=np.std(values,ddof=1) if len(values)>1 else 0,fmt='o',capsize=3);ax.scatter(x+np.linspace(-.08,.08,len(values)),values,s=14)
            ax.set(title=target,xticks=range(4),xticklabels=labels);ax.tick_params(axis='x',labelrotation=40)
        axes[0].set_ylabel('Spearman with −log10(Kd)');fig.suptitle('Already inspected external data; bars = ±1 seed SD, no biological CI');fig.tight_layout();fig.savefig(root/'diagnostic_scores.png',dpi=160);plt.close(fig)
    write_csv(root/'diagnostic_correlations.csv',diagnostics)
    write_json(root/'uncertainty.json',dict(effects=effects,affinity=affinity,method='paired crossed bootstrap; independent seed and biological-component axes',repeats=cfg['bootstrap_repeats'],formal_claims=False))
    incomplete=bool(state['failed_stages']) or any(v['status'] in ('incomplete','failed','blocked') for v in state['tests'].values())
    result=dict(complete=not incomplete,smoke_only=bool(cfg.get('smoke')),protocol_sha256=digest(root/'protocol.json'),tests=state['tests'],data_audit=audit,
        ranking_audit=ranking,runs=runs,decisions=decisions,diagnostic_flags=flags,failed_stages=state['failed_stages'],gpu_budget_used_seconds=state['gpu_budget_used_seconds'],
        limitations=['Short-budget screens are exploratory; no statistically confirmed improvements.',
            'Alternative antigens are putative controls, not proven non-binders.',
            'Existing external affinity assays were already inspected and cannot select training changes.',
            'Parameter matching uses widened IgLM FFNs; native IgLM is reported separately.',
            'Matched examples/updates do not imply identical forward counts or compute.'])
    write_json(root/'results_report.json',result)
    lines=['# Short affinity tests v3','',f'Execution complete: {result["complete"]}. Completed fits: {len(runs)}.',
        f'Training cohort feasible: {audit.get("feasible") if audit else "audit incomplete"}.',
        f'GPU-stage wall seconds (conservative budget accounting): {state["gpu_budget_used_seconds"]:.1f}.','']
    for number in map(str,range(1,7)):lines.append(f'- Test {number}: {state["tests"].get(number,{})}')
    lines+=['','No changes are statistically confirmed by this short screen.','',*['- '+v for v in result['limitations']]]
    (root/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(**vars(p.parse_args()))
