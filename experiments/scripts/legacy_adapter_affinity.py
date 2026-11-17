"""Prospectively specified inference-only test of the strongest legacy lead.

Primary: paired-vs-shuffled frozen antigen adapters, matched within seed.
No new fitting, affinity calibration, model selection or confirmation.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import fcntl
from pathlib import Path
import os
import signal
import subprocess
import time
import numpy as np
import yaml
from affinity_study_common import digest,freeze,read_csv,read_json,write_csv,write_json
from native_mechanism_common import affinity_correlations


def initialize(config):
    cfg=yaml.safe_load(Path(config).read_text());root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    reference=Path(cfg['reference_root']);sources=read_json(reference/'input_manifest.json')
    for path,value in sources['source_sha256'].items():
        if digest(path)!=value:raise ValueError('Legacy data changed: '+path)
    for path,value in read_json(reference/'protocol.json')['implementation_sha256'].items():
        if digest(path)!=value:raise ValueError('Legacy implementation changed: '+path)
    selections=Path(cfg['selection_root']);selection=selections/'diagnostic_selection.csv'
    if digest(selection)!=read_json(selections/'diagnostic_selection.json')['selection_sha256']:
        raise ValueError('Frozen sequence-hash selection changed')
    checkpoints={};exposure={}
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            name=f'pilot_{arm}_s{seed}';folder=reference/'runs'/name;done=read_json(folder/'complete.json')
            if digest(folder/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Legacy checkpoint changed')
            if arm.startswith('decoder') and not done['frozen_decoder_verified']:raise ValueError('Decoder was not frozen')
            exposure[name]=dict(optimizer_steps=done['optimizer_steps'],target_tokens=done['target_tokens'],
                training_membership_sha256=done['training_membership_sha256'],source=done['spec']['source'])
            checkpoints[name]=done['checkpoint_sha256']
        a=exposure[f'pilot_decoder_s{seed}'];b=exposure[f'pilot_decoder_shuffled_s{seed}']
        if a!=b:raise ValueError('Paired/shuffled training exposure differs')
    files=[Path(__file__),Path(config),Path('experiments/workflow/legacy_adapter_affinity.smk'),
           Path('experiments/scripts/native_mechanism_common.py'),Path('experiments/scripts/controlled_antigen_model.py'),
           Path('experiments/scripts/affinity_study_worker.py'),Path('experiments/scripts/train_antigen_diagnostic.py'),
           Path('experiments/scripts/pairing_prefix_score_cache.py'),Path('experiments/scripts/pairing_score_cache.py'),
           Path('experiments/scripts/antigen_diagnostics.py'),Path('workflow/scripts/iglm_baseline.py')]
    binding=Path(cfg['binding_root'])
    antigens=read_csv(binding/'antigens.csv')
    feature_paths=[binding/'embeddings/esm2/test'/f'{a["id"]}.pt' for a in antigens]
    protocol=dict(settings=cfg,checkpoints_sha256=checkpoints,training_exposure=exposure,
        implementation_sha256={str(p):digest(p) for p in files},
        inputs_sha256={str(p):digest(p) for p in [selection,binding/'antigens.csv',binding/'measurements.sqlite',*feature_paths]},
        primary='paired minus shuffled difference in equal-target mean Spearman(-NLL, -log10 Kd)',
        secondary='per-target differences; paired minus IgLM continuation is a composite contrast',
        uncertainty='sample SD across three paired optimization seeds; biological confidence intervals unavailable',
        checkpoints_selected_before_current_affinity_scoring=True,new_training=False,
        scope='exploratory previously inspected benchmark; no independent confirmation or advancement')
    freeze(root/'protocol.json',protocol)
    return root,protocol


def score(root,run):
    import torch
    from affinity_study_worker import restore
    from controlled_antigen_model import ConditionalScorer
    from train_antigen_diagnostic import set_training
    from antigen_diagnostics import iglm_token_nll
    from model_common import load_embedding
    from device_common import get_device
    root=Path(root);protocol=read_json(root/'protocol.json');cfg=protocol['settings']
    path=Path(cfg['reference_root'])/'runs'/run/'best.pt'
    if digest(path)!=protocol['checkpoints_sha256'][run]:raise ValueError('Checkpoint changed')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');target=get_device('frozen adapter affinity')
    model,checkpoint=restore(path,target);kind=checkpoint['spec']['kind'];model.branch=kind=='iglm_cross'
    set_training(model,kind,False)
    if model.branch:
        source_name=read_json(path.parent/'complete.json')['spec']['source']
        source_path=Path(cfg['reference_root'])/'runs'/source_name/'best.pt'
        source_done=read_json(source_path.parent/'complete.json')
        if digest(source_path)!=source_done['checkpoint_sha256']:raise ValueError('Scratch source changed')
        source=torch.load(source_path,map_location='cpu',weights_only=False)['weights']['lm']
        if any(not torch.equal(v,model.lm.state_dict()[k].cpu()) for k,v in source.items()):
            raise ValueError('Original decoder tensors changed in frozen adapter')
        del source
    binding=Path(cfg['binding_root']);selection_path=Path(cfg['selection_root'])/'diagnostic_selection.csv'
    if digest(selection_path)!=protocol['inputs_sha256'][str(selection_path)]:raise ValueError('Selection changed')
    selected=read_csv(selection_path);antigens=read_csv(binding/'antigens.csv')
    scorer=ConditionalScorer(model,{});scores=[];metrics=[];checks=[]
    with torch.inference_mode():
        for antigen in antigens:
            aid=antigen['id'];feature_path=binding/'embeddings/esm2/test'/f'{aid}.pt'
            if digest(feature_path)!=protocol['inputs_sha256'][str(feature_path)]:raise ValueError('Antigen features changed')
            memory=load_embedding(str(feature_path)).to(target) if model.branch else None
            scorer.antigens={aid:memory};scorer.prompts.clear()
            subset=[r for r in selected if r['antigen_id']==aid];sequences=sorted({r['heavy_sequence'] for r in subset})
            for chain in ('heavy','light'):
                sample=sequences[0];scalar=float(iglm_token_nll(model,chain,sample,h_ag=memory).double().mean())
                for size in (1,8,64):
                    scorer.prompts.clear();a=scorer.score(aid,chain,[sample]*size)[0].cpu().double().numpy()/(len(sample)+1)
                    scorer.prompts.clear();b=scorer.score(aid,chain,[sample]*size)[0].cpu().double().numpy()/(len(sample)+1)
                    discrepancy=float(max(np.max(abs(a-scalar)),np.max(abs(a-b))))
                    if discrepancy>cfg['parity_tolerance']:raise ValueError('Scalar/repeated/batched parity failed')
                    checks.append(dict(target=antigen['family'],chain=chain,batch_size=size,max_nll_error=discrepancy))
            values={}
            for start in range(0,len(sequences),64):
                batch=sequences[start:start+64];totals=scorer.score(aid,'heavy',batch)[0].cpu().tolist()
                values.update({s:float(v)/(len(s)+1) for s,v in zip(batch,totals)})
            rows=[dict(query_id=r['query_id'],target=r['target'],affinity=float(r['affinity']),nll=values[r['heavy_sequence']]) for r in subset]
            scores.extend(rows);metrics.append(dict(target=antigen['family'],pairs=len(rows),
                **affinity_correlations([r['nll'] for r in rows],[r['affinity'] for r in rows])))
    folder=root/'scores';folder.mkdir(exist_ok=True);write_csv(folder/f'{run}.csv',scores)
    write_json(folder/f'{run}.complete.json',dict(complete=True,run=run,metrics=metrics,parity=checks,
        decoder_frozen_verified=bool(model.branch),checkpoint_sha256=digest(path),
        scores_sha256=digest(folder/f'{run}.csv'),selection_sha256=digest(selection_path),biological_ci_available=False))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);protocol=read_json(root/'protocol.json');cfg=protocol['settings'];tables={};rows=[]
    for run in protocol['checkpoints_sha256']:
        marker=root/'scores'/f'{run}.complete.json'
        if not marker.exists():continue
        done=read_json(marker)
        if digest(root/'scores'/f'{run}.csv')!=done['scores_sha256']:raise ValueError('Scores changed')
        if done['checkpoint_sha256']!=protocol['checkpoints_sha256'][run]:raise ValueError('Different checkpoint')
        tables[run]={r['target']:r for r in done['metrics']}
    for arm in cfg['arms']:
        if not all(f'pilot_{arm}_s{s}' in tables for s in cfg['seeds']):continue
        for target in ('HER2','PD1','TIGIT'):
            values=[tables[f'pilot_{arm}_s{s}'][target]['spearman'] for s in cfg['seeds']]
            pearson=[tables[f'pilot_{arm}_s{s}'][target]['pearson_perplexity'] for s in cfg['seeds']]
            rows.append(dict(arm=arm,target=target,spearman=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)),
                pearson_perplexity=float(np.mean(pearson)),seed_points=values,biological_ci_available=False))
    contrasts=[]
    for reference in ('decoder_shuffled','iglm_continue'):
        for target in ('HER2','PD1','TIGIT','equal_target_mean'):
            values=[]
            for seed in cfg['seeds']:
                names=[f'pilot_{arm}_s{seed}' for arm in ('decoder',reference)]
                if not all(n in tables for n in names):break
                targets=('HER2','PD1','TIGIT') if target=='equal_target_mean' else (target,)
                values.append(float(np.mean([tables[names[0]][t]['spearman']-tables[names[1]][t]['spearman'] for t in targets])))
            if len(values)==len(cfg['seeds']):
                contrasts.append(dict(contrast='paired_minus_'+reference,target=target,mean=float(np.mean(values)),
                    seed_sd=float(np.std(values,ddof=1)),seed_points=values,biological_ci_available=False))
    write_csv(root/'affinity_correlations.csv',rows);write_csv(root/'paired_differences.csv',contrasts)
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True)
        for ax,target in zip(axes,('HER2','PD1','TIGIT')):
            subset=[r for r in rows if r['target']==target]
            for i,r in enumerate(subset):
                ax.errorbar(r['spearman'],i,xerr=r['seed_sd'],fmt='o',capsize=4)
                ax.scatter(r['seed_points'],np.full(len(cfg['seeds']),i)+np.linspace(-.1,.1,len(cfg['seeds'])),s=15,alpha=.7)
            ax.set(yticks=range(len(subset)),yticklabels=[r['arm'] for r in subset],title=target,xlabel='Spearman ρ (higher is better)')
        fig.suptitle('Frozen legacy checkpoints; bars: ±1 optimization-seed SD, not biological CIs')
        fig.tight_layout();fig.savefig(root/'affinity_correlations.png',dpi=170);plt.close(fig)
    if contrasts:
        fig,ax=plt.subplots(figsize=(9,5))
        for i,r in enumerate(contrasts):
            ax.errorbar(r['mean'],i,xerr=r['seed_sd'],fmt='o',capsize=4)
            ax.scatter(r['seed_points'],np.full(len(cfg['seeds']),i)+np.linspace(-.1,.1,len(cfg['seeds'])),s=15,alpha=.7)
        ax.axvline(0,color='grey',linestyle='--')
        ax.set(yticks=range(len(contrasts)),yticklabels=[r['contrast']+' / '+r['target'] for r in contrasts],
               xlabel='Paired change in Spearman ρ (positive favors paired antigen training)',title='Error bars: ±1 paired training-seed SD')
        fig.tight_layout();fig.savefig(root/'paired_effects.png',dpi=170);plt.close(fig)
    state=read_json(root/'state.json');complete=len(tables)==len(protocol['checkpoints_sha256'])
    write_json(root/'results_report.json',dict(complete=complete,new_fits=0,checkpoints_scored=len(tables),
        correlations=rows,contrasts=contrasts,failed_stages=state['failed_stages'],confirmed_advantages=[],
        limitations=['External libraries were already inspected; this is exploratory.',
            'Three optimization seeds and related library variants do not supply independent biological confidence intervals.',
            'Checkpoint selection used a development panel with only two source groups.',
            'IgLM continuation comparison combines freezing, branch architecture and dropout differences.',
            'The legacy cohort is not newly certified by the stricter native protection audit.']))
    text=['Frozen adapter affinity test',f'Complete: {complete}. Existing checkpoints scored: {len(tables)}. New fits: 0.',
          'Primary contrast isolates real versus shuffled antigen pairing within the same frozen-decoder architecture.',
          'Bars are sample SD over three matched optimization seeds, not biological confidence intervals.','',
          '| Arm | Target | Spearman mean | Seed SD |','|---|---|---:|---:|']
    text += [f'| {r["arm"]} | {r["target"]} | {r["spearman"]:.4f} | {r["seed_sd"]:.4f} |' for r in rows]
    text += ['', 'Paired differences:']+[f'{r["contrast"]}, {r["target"]}: {r["mean"]:+.4f} ± {r["seed_sd"]:.4f} seed SD.' for r in contrasts]
    text += ['', 'Previously inspected external panel; no formal significance, independent confirmation or advancement claim.']
    (root/'findings.md').write_text('\n'.join(text)+'\n')


def run(config):
    root,protocol=initialize(config);jobs={};failed=[]
    def state(status):write_json(root/'state.json',dict(status=status,jobs=jobs,failed_stages=failed,pid=os.getpid()))
    def stop(signum,frame):raise InterruptedError('Experiment stopped')
    signal.signal(signal.SIGTERM,stop)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state('running')
        targets=[(name,root/'scores'/f'{name}.complete.json') for name in protocol['checkpoints_sha256']]
        targets.append(('report',root/'results_report.json'))
        for name,target in targets:
            for path,value in protocol['implementation_sha256'].items():
                if digest(path)!=value:raise ValueError('Frozen code changed')
            folder=root/'controller_logs';folder.mkdir(exist_ok=True);log=folder/f'{name}.log'
            env=dict(os.environ,MANGO_ADAPTER_AFFINITY_CONFIG=os.path.relpath(Path(config).resolve(),Path.cwd()))
            cmd=['.snakemake/gpu-driver/bin/snakemake','-s','experiments/workflow/legacy_adapter_affinity.smk','--profile',
                 'workflow/profiles/gpu','--rerun-incomplete','--rerun-triggers','mtime','--',str(target)]
            start=time.monotonic()
            with log.open('w') as handle:
                child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:code=child.wait(timeout=600)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    state('interrupted');raise
            jobs[name]=dict(status='complete' if code==0 else 'failed',returncode=code,log=str(log),wall_seconds=time.monotonic()-start)
            if code:failed.append(name)
            state('running')
        state('incomplete' if failed else 'complete');print(root/'findings.md',flush=True)
    return int(bool(failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['run','score','report']);p.add_argument('--config',default='experiments/config/legacy_adapter_affinity.yaml')
    p.add_argument('--root');p.add_argument('--run');a=p.parse_args()
    if a.job=='run':raise SystemExit(run(a.config))
    elif a.job=='score':score(a.root,a.run)
    else:report(a.root)
