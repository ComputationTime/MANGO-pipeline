"""Exploratory four-arm objective pilot on unchanged protected native groups.

One published RNase-A VHH preference is an explicitly limited intervention.
Unreported assay uncertainty stays unknown. Old confirmatory feasibility gates
are not claimed to have passed; seed SD is the only reported error bar.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import time

from affinity_study_common import digest, fingerprint, freeze, read_csv, read_json, write_csv, write_json

ARMS={'IgLM_CE':(False,False),'MANGO_CE':(True,False),
      'IgLM_rank':(False,True),'MANGO_rank':(True,True)}


def verify(root):
    protocol=read_json(root/'training_protocol.json')
    for field in ('inputs_sha256','implementation_sha256'):
        for name,value in protocol[field].items():
            if digest(name)!=value:raise ValueError('Frozen pilot input changed: '+name)
    audit=read_json(root/'cohort_audit.json')
    for name,value in audit['files_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Pilot cohort changed: '+name)
    return protocol['settings']


def native_schedule(rows,updates,batch_size,seed):
    from chain_examples import example_chains
    bins=defaultdict(list)
    for r in rows:
        if r['split']=='train':
            for chain in example_chains(r):bins[r['biological_group']].append(dict(id=r['id'],chain=chain))
    rng=random.Random(seed);groups=sorted(bins)
    return [[dict(rng.choice(bins[rng.choice(groups)])) for _ in range(batch_size)] for _ in range(updates)]


def prepare(config):
    import yaml
    from prepare_native_pilot import deduplicate
    cfg=yaml.safe_load(Path(config).read_text());root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    prior=Path(cfg['protected_root']);audit=read_json(prior/'cohort_audit.json')
    for name,value in audit['files_sha256'].items():
        if digest(prior/name)!=value:raise ValueError('Protected source changed: '+name)
    source={r['id']:r for r in read_csv(cfg['source_records'])};rows=[]
    safe=[r for r in read_csv(prior/'component_membership.csv') if not r['excluded']]
    for split in ('train','reserved'):
        selected=[dict(source[r['id']],biological_group=r['group'],source_split=r['source_split'])
                  for r in safe if r['source_split']==split]
        rows.extend(dict(r,split='train' if split=='train' else 'test') for r in deduplicate(selected))
    ledger=read_json(cfg['primary_ledger'])['measurements']
    evidence=[next(m for m in ledger if m['measurement_id']==mid) for mid in ('rn05_2007_a','rn05_2007_wt')]
    pair=[]
    for e in evidence:
        seqs={r['expected_heavy_seq'] for r in source.values() if r['pdb_id'][-4:].lower() in e['pdb_codes']}
        choices=[r for r in rows if r['split']=='train' and r['expected_heavy_seq'] in seqs and not r['resolved_L_seq']]
        if not choices:raise ValueError('Measured RNase-A variant is not in protected training data')
        row=sorted(choices,key=lambda r:r['id'])[0]
        if row['resolved_H_seq']!=row['expected_heavy_seq']:raise ValueError('Incomplete measured variable-domain target')
        pair.append(dict(id=row['id'],chain='heavy',measurement_id=e['measurement_id'],kd_nm=e['kd_nm']))
    a,b=(next(r for r in rows if r['id']==p['id']) for p in pair)
    if a['resolved_ag_seq']!=b['resolved_ag_seq'] or a['biological_group']!=b['biological_group']:
        raise ValueError('RNase-A preference changes the antigen or biological group')
    if evidence[1]['kd_nm']/evidence[0]['kd_nm']<100:raise ValueError('Documented affinity ordering changed')
    if cfg['smoke']:
        train=[r for r in rows if r['split']=='train' and r['id'] not in {p['id'] for p in pair}][:2]
        rows=train+[a,b]+[r for r in rows if r['split']=='test'][:2]
    for field in ('ab_ag_cluster','biological_group'):
        splits=defaultdict(set)
        for r in rows:splits[r[field]].add(r['split'])
        if any(len(s)>1 for s in splits.values()):raise ValueError('Group crosses splits')
    if any(r['source_split']!='train' for r in rows if r['split']=='train'):
        raise ValueError('Prior protected holdout entered training')
    selection=Path(cfg['selection_root'])/'diagnostic_selection.csv'
    if digest(selection)!=read_json(selection.with_suffix('.json'))['selection_sha256']:
        raise ValueError('Previously frozen external selection changed')
    inputs=[config,cfg['source_records'],cfg['primary_ledger'],str(prior/'cohort_audit.json'),
            str(prior/'component_membership.csv'),str(selection),str(Path(cfg['binding_root'])/'antigens.csv')]
    implementation=[str(Path(__file__)),'experiments/scripts/affinity_study_common.py',
                    'experiments/scripts/short_affinity_model.py','experiments/scripts/controlled_antigen_model.py',
                    'experiments/scripts/antigen_diagnostics.py','workflow/scripts/iglm_baseline.py',
                    'experiments/scripts/native_mechanism_common.py','workflow/scripts/chain_examples.py',
                    'experiments/scripts/native_mechanism_embeddings.py','experiments/scripts/short_affinity_embeddings.py',
                    'experiments/scripts/prepare_native_pilot.py','experiments/workflow/affinity_ranking_pilot.smk']
    freeze(root/'training_protocol.json',dict(settings=cfg,
        inputs_sha256={p:digest(p) for p in inputs},implementation_sha256={p:digest(p) for p in implementation},
        intervention='CE plus a fixed 0.1 softplus preference on one same-assay RNase-A VHH pair, versus identical CE exposure with zero ranking weight.',
        primary='Interaction: (MANGO_rank - MANGO_CE) - (IgLM_rank - IgLM_CE) in equal-target mean Spearman(-NLL, -log10 KD).',
        affinity_evaluation='Previously inspected HER2/PD1/TIGIT panel, exploratory only. No new model selection or stopping on outcomes.',
        uncertainty='Sample SD across five paired optimization seeds. No biological confidence intervals, p-values or confirmation.',
        scope='Bounded exploratory intervention; one training parent and small native holdout. Original confirmatory gates remain unmet.',
        deviations_from_confirmatory_design=['Use unchanged existing native train/reserved allocation, without a minimum of 100 heavy test examples.',
            'Use a clearly ordered published pair with unknown replicate/error definition; do not impute uncertainty or claim it satisfies the strict label gate.',
            'Display variable domains are the targets; common assay tags/termini are not generated. Table/narrative KD disagreement does not change the binary order.'],
        biological_conditioning='Antigen only, plus learned H/L selector. All decoder weights start from scratch.',
        representation='Frozen pretrained ESM2 650M sequence embeddings, fixed 32-token mean pooling. Geometry is unchanged.',
        pair_evidence=evidence,smoke_only=cfg['smoke']))
    write_csv(root/'records.csv',rows);write_json(root/'ranking_pair.json',dict(strong=pair[0],weak=pair[1]))
    external=read_csv(selection)
    if cfg['smoke']:external=[r for t in ('HER2','PD1','TIGIT') for r in [x for x in external if x['target']==t][:3]]
    write_csv(root/'external_selection.csv',external)
    for seed in cfg['seeds']:
        for phase,updates in [('pretrain',cfg['pretraining_updates']),('adapt',cfg['adaptation_updates'])]:
            write_json(root/'schedules'/f'{phase}_{seed}.json',dict(items=native_schedule(rows,updates,cfg['native_batch_size'],seed+(0 if phase=='pretrain' else 100000))))
    names=['records.csv','ranking_pair.json','external_selection.csv']
    names += [str(p.relative_to(root)) for p in (root/'schedules').glob('*.json')]
    counts={s:dict(records=sum(r['split']==s for r in rows),groups=len({r['biological_group'] for r in rows if r['split']==s})) for s in ('train','test')}
    write_json(root/'cohort_audit.json',dict(feasible=True,exploratory_only=True,counts=counts,
               original_safe_records=len(safe),original_protection_unchanged=True,
               files_sha256={n:digest(root/n) for n in names}))
    print(json.dumps(counts),flush=True)


class BatchScorer:
    def __init__(self,model,root):
        import torch
        from controlled_antigen_model import memory
        self.model=model;self.target=next(model.lm.parameters()).device;self.context={}
        manifest=read_json(root/'embeddings_complete.json')
        for p,value in manifest['files_sha256'].items():
            if digest(p)!=value:raise ValueError('Antigen embedding changed')
            payload=torch.load(p,map_location='cpu',weights_only=False)
            self.context[payload['antigen_sha256']]=memory(payload['embedding'],32)[0].to(self.target)

    def losses(self,items):
        import torch
        import torch.nn.functional as F
        sequences=[r['resolved_H_seq' if c=='heavy' else 'resolved_L_seq'] for r,c in items]
        encoded=[[self.model.vocab[f'[{c.upper()}]'],self.model.vocab['[HUMAN]'],
                  *[self.model.vocab[a] for a in s],self.model.vocab['[SEP]']]
                 for (r,c),s in zip(items,sequences)]
        maximum=max(map(len,encoded))
        if maximum>self.model.base_config['n_positions']:raise ValueError('Target exceeds context window')
        ids=torch.zeros((len(items),maximum),dtype=torch.long,device=self.target)
        mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
        for i,values in enumerate(encoded):
            ids[i,:len(values)]=torch.tensor(values,device=self.target);mask[i,:len(values)]=1
            labels[i,2:len(values)]=ids[i,2:len(values)]
        kwargs={}
        if self.model.branch:
            contexts=torch.stack([self.context[fingerprint(r['resolved_ag_seq'])] for r,c in items])
            kwargs['encoder_hidden_states']=self.model.antigen_projection(contexts)
        logits=self.model.lm(input_ids=ids,attention_mask=mask,use_cache=False,**kwargs).logits
        targets=labels[:,1:];valid=targets!=-100
        values=F.cross_entropy(logits[:,:-1].transpose(1,2),targets,reduction='none')
        sums=(values*valid).sum(1);counts=valid.sum(1)
        return sums/counts,sums,counts


def loss_parts(native,pair,weight,pair_ce_weight):
    import torch.nn.functional as F
    ce=(1-pair_ce_weight)*native.mean()+pair_ce_weight*pair.mean()
    rank=F.softplus(pair[0]-pair[1])
    return ce+weight*rank,ce,rank


def curve(path,points):
    from short_affinity_worker import curve_plot
    curve_plot(path,[(r['iteration'],r['objective']) for r in points])


def fit(root,family,seed,arm=None):
    import hashlib
    import torch
    from controlled_antigen_model import device,mode
    from device_common import get_device
    from short_affinity_model import build,load,save,model_parameters
    root=Path(root);cfg=verify(root);branch=family=='MANGO'
    target=get_device('affinity ranking pilot');torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    phase='adapt' if arm else 'pretrain';name=f'{arm or family}_s{seed}'
    directory=root/('runs' if arm else 'pretraining')/name;directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();spec=dict(id=name,seed=seed,branch=branch,pool='mean',objective=arm or 'pretrain')
    parent=None
    if arm:
        parent=root/'pretraining'/f'{family}_s{seed}'/'final.pt'
        done=read_json(parent.parent/'complete.json')
        if digest(parent)!=done['checkpoint_sha256']:raise ValueError('Pretraining checkpoint changed')
        model,_=load(parent,target)
    else:
        model=build(spec);device(model,target)
    model.base_config=dict(model.base_config)
    state_hash=hashlib.sha256()
    for k,v in sorted(model.lm.state_dict().items()):state_hash.update(k.encode());state_hash.update(v.detach().cpu().contiguous().numpy().tobytes())
    if branch:
        for k,v in sorted(model.antigen_projection.state_dict().items()):state_hash.update(k.encode());state_hash.update(v.detach().cpu().contiguous().numpy().tobytes())
    initial_hash=state_hash.hexdigest()
    rows=read_csv(root/'records.csv');lookup={r['id']:r for r in rows};pair=read_json(root/'ranking_pair.json')
    schedule_path=root/'schedules'/f'{phase}_{seed}.json';items=read_json(schedule_path)['items']
    expected=cfg['adaptation_updates'] if arm else cfg['pretraining_updates']
    if len(items)!=expected:raise ValueError('Schedule length changed')
    scorer=BatchScorer(model,root);optimizer=torch.optim.Adam(model_parameters(model),lr=cfg['lr'])
    torch.manual_seed(seed+(2000000 if arm else 1000000));mode(model,True)
    points=[];tokens=0;exposure=[]
    with (directory/'training_curve.csv').open('w',newline='') as handle:
        fields=['iteration','phase','objective','nll','ranking_loss'];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for step,batch in enumerate(items,1):
            optimizer.zero_grad(set_to_none=True)
            examples=[(lookup[x['id']],x['chain']) for x in batch]
            if any(r['split']!='train' for r,c in examples):raise ValueError('Held-out row entered training')
            if arm:
                examples += [(lookup[pair[k]['id']],'heavy') for k in ('strong','weak')]
            losses,_,counts=scorer.losses(examples)
            if arm:
                loss,ce,rank=loss_parts(losses[:-2],losses[-2:],cfg['ranking_weight'] if ARMS[arm][1] else 0.,cfg['pair_ce_weight'])
            else:loss=ce=losses.mean();rank=losses.new_tensor(0.)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();optimizer.step();tokens+=int(counts.sum());exposure.extend((r['id'],c) for r,c in examples)
            point=dict(iteration=step,phase=phase,objective=float(loss.detach()),nll=float(ce.detach()),ranking_loss=float(rank.detach()))
            points.append(point);writer.writerow(point);handle.flush()
            if step%100==0:
                curve(directory/'training_curve.png',points)
                print(name,step,point['objective'],flush=True)
    mode(model,False);save(directory/'final.pt',model,spec,dict(updates=len(items),phase=phase,
        protocol_sha256=digest(root/'training_protocol.json'),records_sha256=digest(root/'records.csv')))
    curve(directory/'training_curve.png',points)
    write_json(directory/'complete.json',dict(complete=True,arm=arm,family=family,seed=seed,updates=len(items),
        checkpoint_sha256=digest(directory/'final.pt'),parent_sha256=digest(parent) if parent else None,
        initial_sha256=initial_hash,schedule_sha256=digest(schedule_path),
        exposure_sha256=fingerprint(json.dumps(exposure)),target_tokens=tokens,
        parameters=sum(p.numel() for p in model_parameters(model)),
        wall_seconds=time.monotonic()-started,smoke_only=cfg['smoke']))


def score(root,arm,seed):
    import numpy as np
    import torch
    from short_affinity_model import load
    from device_common import get_device
    from antigen_diagnostics import iglm_token_nll
    from controlled_antigen_model import mode
    from native_mechanism_common import affinity_correlations
    root=Path(root);cfg=verify(root);directory=root/'runs'/f'{arm}_s{seed}';done=read_json(directory/'complete.json')
    path=directory/'final.pt'
    if digest(path)!=done['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');model,_=load(path,get_device('affinity ranking evaluation'));mode(model,False)
    scorer=BatchScorer(model,root);records=read_csv(root/'records.csv');test=[r for r in records if r['split']=='test']
    selection=read_csv(root/'external_selection.csv');antigens=read_csv(Path(cfg['binding_root'])/'antigens.csv')
    checks=[];external_scores=[];metrics=[];native_scores=[]
    with torch.inference_mode():
        sample=test[0]
        for chain in ('heavy','light'):
            candidate=next(r for r in records if r['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'])
            memory=scorer.context[fingerprint(candidate['resolved_ag_seq'])][None] if model.branch else None
            sequence=candidate['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
            scalar=float(iglm_token_nll(model,chain,sequence,h_ag=memory).double().mean())
            for size in (1,8,32):
                means,_,_=scorer.losses([(candidate,chain)]*size);error=float((means-scalar).abs().max())
                if error>cfg['parity_tolerance']:raise ValueError('Scalar/batch likelihood parity failed')
                checks.append(dict(chain=chain,batch_size=size,max_error=error))
        from chain_examples import example_chains
        examples=[(r,c) for r in test for c in example_chains(r)]
        for start in range(0,len(examples),16):
            batch=examples[start:start+16];means,sums,counts=scorer.losses(batch)
            for (r,c),v,s,n in zip(batch,means.tolist(),sums.tolist(),counts.tolist()):
                native_scores.append(dict(id=r['id'],chain=c,group=r['biological_group'],nll=v,loss_sum=s,ntokens=n))
        pair=read_json(root/'ranking_pair.json');lookup={r['id']:r for r in records}
        means,_,_=scorer.losses([(lookup[pair[k]['id']],'heavy') for k in ('strong','weak')])
        training_gap=float(means[1]-means[0])
        for ag in antigens:
            subset=[r for r in selection if r['antigen_id']==ag['id']];sequences=sorted({r['heavy_sequence'] for r in subset});values={}
            for start in range(0,len(sequences),32):
                seqs=sequences[start:start+32]
                batch=[(dict(ag,resolved_H_seq=s),'heavy') for s in seqs]
                means,_,_=scorer.losses(batch);values.update(zip(seqs,means.tolist()))
            scored=[dict(query_id=r['query_id'],target=r['target'],affinity=float(r['affinity']),nll=values[r['heavy_sequence']]) for r in subset]
            external_scores.extend(scored)
            metrics.append(dict(target=ag['family'],pairs=len(scored),**affinity_correlations([r['nll'] for r in scored],[r['affinity'] for r in scored])))
        # Antigen invariance is mandatory for the unconditional control.
        if not model.branch:
            sequence=selection[0]['heavy_sequence']
            means,_,_=scorer.losses([(dict(ag,resolved_H_seq=sequence),'heavy') for ag in antigens])
            if float(means.max()-means.min())>cfg['parity_tolerance']:raise ValueError('IgLM depends on antigen')
    write_csv(directory/'external_scores.csv',external_scores);write_csv(directory/'native_scores.csv',native_scores)
    write_json(directory/'evaluation.json',dict(complete=True,arm=arm,seed=seed,metrics=metrics,parity=checks,
        training_pair_nll_gap=training_gap,training_pair_is_in_sample=True,
        native_nll=sum(r['loss_sum'] for r in native_scores)/sum(r['ntokens'] for r in native_scores),
        checkpoint_sha256=digest(path),external_scores_sha256=digest(directory/'external_scores.csv'),
        native_scores_sha256=digest(directory/'native_scores.csv'),biological_ci_available=False,smoke_only=cfg['smoke']))


def report(root):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);cfg=verify(root);seeds=cfg['seeds'];table={};fits={}
    for arm in ARMS:
        for seed in seeds:
            folder=root/'runs'/f'{arm}_s{seed}'
            fits[arm,seed]=read_json(folder/'complete.json');value=read_json(folder/'evaluation.json')
            for file,key in [('external_scores.csv','external_scores_sha256'),('native_scores.csv','native_scores_sha256')]:
                if digest(folder/file)!=value[key]:raise ValueError('Evaluation scores changed')
            table[arm,seed]=value
    for family in ('IgLM','MANGO'):
        for seed in seeds:
            a=fits[family+'_CE',seed];b=fits[family+'_rank',seed]
            for key in ('initial_sha256','parent_sha256','schedule_sha256','exposure_sha256','target_tokens','parameters'):
                if a[key]!=b[key]:raise ValueError('CE/ranking arm mismatch: '+key)
    parameters={a:fits[a,seeds[0]]['parameters'] for a in ARMS}
    if abs(parameters['MANGO_CE']/parameters['IgLM_CE']-1)>.001:raise ValueError('Parameter matching failed')
    summaries=[];contrasts=[]
    stats=lambda values:dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.,seed_points=values)
    def metric(arm,seed,target):
        values=table[arm,seed]['metrics'];selected=[r['spearman'] for r in values if target=='macro' or r['target']==target]
        if not selected or any(x is None for x in selected):return None
        return float(np.mean(selected))
    for target in ('HER2','PD1','TIGIT','macro'):
        if any(metric(a,s,target) is None for a in ARMS for s in seeds):continue
        for arm in ARMS:
            summaries.append(dict(arm=arm,target=target,**stats([metric(arm,s,target) for s in seeds])))
        formulas={
            'MANGO_minus_IgLM_CE':{'MANGO_CE':1,'IgLM_CE':-1},
            'MANGO_minus_IgLM_rank':{'MANGO_rank':1,'IgLM_rank':-1},
            'ranking_gain_MANGO':{'MANGO_rank':1,'MANGO_CE':-1},
            'ranking_gain_IgLM':{'IgLM_rank':1,'IgLM_CE':-1},
            'interaction':{'MANGO_rank':1,'MANGO_CE':-1,'IgLM_rank':-1,'IgLM_CE':1}}
        for name,weights in formulas.items():
            points=[sum(w*metric(a,s,target) for a,w in weights.items()) for s in seeds]
            contrasts.append(dict(contrast=name,target=target,**stats(points)))
    write_csv(root/'affinity_correlations.csv',summaries);write_csv(root/'paired_contrasts.csv',contrasts)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,target in zip(axes,('HER2','PD1','TIGIT')):
        for i,r in enumerate(x for x in summaries if x['target']==target):
            ax.errorbar(r['mean'],i,xerr=r['seed_sd'],fmt='o',capsize=4)
            ax.scatter(r['seed_points'],np.full(len(seeds),i),s=10,alpha=.5)
        ax.set(yticks=range(4),yticklabels=list(ARMS),title=target,xlabel='Spearman rho')
    fig.suptitle('Exploratory affinity pilot — bars: ±1 optimization-seed SD')
    fig.savefig(root/'affinity_correlations.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i,arm in enumerate(ARMS):
        for ax,key,label in [(axes[0],'training_pair_nll_gap','In-sample RNase-A preference gap'),(axes[1],'native_nll','Held-out native token NLL')]:
            values=[table[arm,s][key] for s in seeds];v=stats(values)
            ax.errorbar(v['mean'],i,xerr=v['seed_sd'],fmt='o',capsize=4);ax.scatter(values,np.full(len(values),i),s=10)
            ax.set(yticks=range(4),yticklabels=list(ARMS),xlabel=label)
    fig.suptitle('Bars: ±1 optimization-seed SD; no biological CIs')
    fig.savefig(root/'mechanism_and_native.png',dpi=170);plt.close(fig)
    report=dict(complete=True,smoke_only=cfg['smoke'],final_models=len(table),shared_pretraining_fits=2*len(seeds),
        correlations=summaries,contrasts=contrasts,parameters=parameters,
        native_nll={a:stats([table[a,s]['native_nll'] for s in seeds]) for a in ARMS},
        training_pair_gap={a:stats([table[a,s]['training_pair_nll_gap'] for s in seeds]) for a in ARMS},
        paired_exposure_and_initialization_verified=True,biological_ci_available=False,
        limitations=['One affinity-training parent; assay error type/replicate count unknown.',
            '99 deduplicated native training records and 36 held-out records; not a powered generalization study.',
            'External panels were previously inspected and lack independent lineage IDs.',
            'Seed SD quantifies optimization variability only; no formal significance or confirmation.',
            'MANGO and IgLM match total parameters within 0.1%; effective capacity and compute differ.',
            'Geometry is held fixed; this experiment isolates the ranking objective.'])
    write_json(root/'results_report.json',report)
    primary=next((r for r in contrasts if r['contrast']=='interaction' and r['target']=='macro'),None)
    lines=['Exploratory four-arm affinity ranking pilot',f'Completed final models: {len(table)}. Smoke only: {cfg["smoke"]}.',
           'The primary contrast is the change in the MANGO-minus-IgLM affinity gap caused by adding ranking.',
           'Error bars are sample SD across paired optimization seeds, not biological confidence intervals.']
    if primary:lines.append(f'Macro Spearman interaction: {primary["mean"]:+.6f} +/- {primary["seed_sd"]:.6f} seed SD.')
    lines+=['','| Arm | Target | Spearman mean | Seed SD |','|---|---|---:|---:|']
    lines += [f'| {r["arm"]} | {r["target"]} | {r["mean"]:.5f} | {r["seed_sd"]:.5f} |' for r in summaries]
    lines+=['','Limitations:']+report['limitations']
    (root/'findings.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['prepare','fit','score','report']);p.add_argument('--config',default='experiments/config/affinity_ranking_pilot.yaml')
    p.add_argument('--root');p.add_argument('--family',choices=['IgLM','MANGO']);p.add_argument('--arm',choices=list(ARMS));p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.job=='prepare':prepare(a.config)
    elif a.job=='fit':fit(Path(a.root),a.family,a.seed,a.arm)
    elif a.job=='score':score(Path(a.root),a.arm,a.seed)
    else:report(Path(a.root))
