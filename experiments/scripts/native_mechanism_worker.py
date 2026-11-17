"""Fixed-budget native IgLM factorial; all targets are autoregressive outputs."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from affinity_study_common import digest, fingerprint, freeze, read_csv, read_json, write_csv, write_json
from controlled_antigen_model import build as base_build, device, mode, parameters
from antigen_diagnostics import iglm_token_nll, wrong_antigen_map
from chain_examples import example_chains
from device_common import get_device
from native_mechanism_common import arms


def build(spec, source=None, config=None):
    if source is None and spec.get('wide'):
        from short_affinity_model import build as wide_build
        model=wide_build(dict(seed=spec['seed'],branch=False,pool='mean',native=False),config)
    else:
        model=base_build(spec['seed'],spec['branch'],spec['frozen'],
            source=source['weights']['lm'] if source else None,
            config=source['decoder_config'] if source else config)
    if spec['representation']=='one_hot':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['seed']+1000000)
            model.antigen_projection=torch.nn.Linear(21,model.decoder_config['n_embd'])
    return model


def save(path, model, spec, metadata):
    weights={'lm':{k:v.detach().cpu() for k,v in model.lm.state_dict().items()}}
    if model.branch:weights['antigen_projection']={k:v.detach().cpu() for k,v in model.antigen_projection.state_dict().items()}
    value=dict(format='native_mechanism_v1',spec=spec,decoder_config=model.decoder_config,weights=weights,**metadata)
    temporary=Path(path).with_suffix('.tmp.pt');torch.save(value,temporary);temporary.replace(path)


def load(path, target):
    value=torch.load(path,map_location='cpu',weights_only=False)
    spec=value['spec']
    model=base_build(spec['seed'],spec['branch'],spec['frozen'],config=value['decoder_config'])
    if spec['representation']=='one_hot':model.antigen_projection=torch.nn.Linear(21,value['decoder_config']['n_embd'])
    for k,v in value['weights'].items():getattr(model,k).load_state_dict(v,strict=True)
    device(model,target);mode(model,False)
    return model,value


class Features:
    def __init__(self,root,spec,target,tokens):
        self.root,self.spec,self.target,self.tokens=root,spec,target,tokens
        self.cache={}

    def __call__(self,row,constant=False):
        if not self.spec['branch']:return None
        key=fingerprint(row['resolved_ag_seq'])
        if key not in self.cache:
            if self.spec['representation']=='one_hot':
                alphabet={a:i for i,a in enumerate('ACDEFGHIKLMNPQRSTVWY|')}
                sequence=row['resolved_ag_seq'].replace(',','|')
                ids=torch.tensor([alphabet[a] for a in sequence])
                tensor=F.one_hot(ids,21).float()
            else:
                path=self.root/'embeddings'/f'{key}.pt'
                marker=read_json(self.root/'embeddings_complete.json')
                if marker['files_sha256'].get(str(path))!=digest(path):raise ValueError('Antigen feature changed')
                tensor=torch.load(path,map_location='cpu',weights_only=False)['embedding']
            self.cache[key]=F.adaptive_avg_pool1d(tensor.T.unsqueeze(0),self.tokens).transpose(1,2)
        value=self.cache[key].to(self.target)
        return torch.zeros_like(value) if constant or self.spec['memory']=='constant' else value


def loss(model,features,row,chain,donor=None,constant=False):
    sequence=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
    return iglm_token_nll(model,chain,sequence,h_ag=features(donor or row,constant))


def score(model,features,rows):
    mode(model,False)
    donors=wrong_antigen_map(rows,1729) if model.branch else {}
    result=[]
    with torch.inference_mode():
        for r in rows:
            for chain in example_chains(r):
                sequence=r['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
                for variant in (('correct','wrong','constant') if model.branch else ('correct',)):
                    values=loss(model,features,r,chain,donors[r['id']] if variant=='wrong' else None,variant=='constant').double()
                    if len(values)!=len(sequence)+1 or not torch.isfinite(values).all():raise ValueError('Invalid evaluation likelihood')
                    result.append(dict(id=r['id'],group=r['biological_group'],chain=chain,variant=variant,
                        sequence_sha256=fingerprint(sequence),loss_sum=float(values.sum()),ntokens=len(values),nll=float(values.mean())))
    return result


def train(root, arm, seed):
    cfg=read_json(root/'training_protocol.json')['settings']
    spec=dict(arms()[arm],id=f'{arm}_s{seed}',arm=arm,seed=seed)
    folder=root/'runs'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    freeze(folder/'spec.json',spec)
    audit=read_json(root/'cohort_audit.json')
    if not audit['feasible'] or digest(root/'records.csv')!=audit['files_sha256']['records.csv']:
        raise ValueError('Cohort is not eligible')
    marker=folder/'complete.json'
    if marker.exists():
        done=read_json(marker)
        for name,value in done['files_sha256'].items():
            if digest(folder/name)!=value:raise ValueError('Completed run changed')
        return
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    target=get_device('native mechanism training')
    source=None
    if spec['source']:
        path=root/'runs'/f'{spec["source"]}_s{seed}'/'final.pt'
        expected=read_json(path.parent/'complete.json')['files_sha256']['final.pt']
        if digest(path)!=expected:raise ValueError('Source checkpoint changed')
        source=torch.load(path,map_location='cpu',weights_only=False)
    model=build(spec,source);device(model,target)
    snapshot={k:v.detach().cpu().clone() for k,v in model.lm.state_dict().items()
              if spec['frozen'] and '.crossattention.' not in k and '.ln_cross_attn.' not in k}
    import hashlib
    initial=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()
        for k,v in sorted(model.lm.state_dict().items()) if '.crossattention.' not in k and '.ln_cross_attn.' not in k)).hexdigest()
    features=Features(root,spec,target,cfg['memory_tokens'])
    rows=read_csv(root/'records.csv');by_id={r['id']:r for r in rows}
    selection=read_json(root/'schedules'/f'{seed}.json')
    if selection['records_sha256']!=digest(root/'records.csv'):raise ValueError('Schedule source changed')
    items=selection['items']
    items=items[:cfg['pretraining_updates']] if spec['source'] is None else items[cfg['pretraining_updates']:]
    # Zero-output cross-attention must preserve the source before training.
    parity_error=0.
    if source and spec['branch']:
        reference,_=load(root/'runs'/f'{spec["source"]}_s{seed}'/'final.pt',target)
        mode(model,False)
        with torch.inference_mode():
            for item in items[:2]:
                row=by_id[item['id']];chain=item['chain']
                sequence=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
                a=loss(model,features,row,chain).double().mean()
                b=iglm_token_nll(reference,chain,sequence).double().mean()
                parity_error=max(parity_error,float(abs(a-b)))
        del reference
        if parity_error>3e-6:raise ValueError('Zero-output source parity failed')
    optimizer=torch.optim.Adam([p for p in parameters(model) if p.requires_grad],lr=cfg['lr'])
    mode(model,True);torch.manual_seed(seed+2000000+(0 if source is None else 1))
    target_tokens=0;auxiliary_tokens=0;forwards=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with (folder/'training_curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['iteration','nll','objective','target_tokens'])
        writer.writeheader()
        for step,item in enumerate(items,1):
            row=by_id[item['id']];donor=by_id[item['shuffled_id']] if spec['memory']=='shuffled' else row
            optimizer.zero_grad(set_to_none=True)
            values=loss(model,features,row,item['chain'],donor);nll=values.mean();objective=nll;forwards+=1
            if spec['objective']=='pairing':
                # Preserve the main CE dropout stream across objective controls.
                with torch.random.fork_rng(devices=[target.index or 0]):
                    decoy=by_id[item['decoy_'+spec['memory']]]
                    other=loss(model,features,row,item['chain'],decoy);forwards+=1;auxiliary_tokens+=len(other)
                objective=nll+cfg['pairing_weight']*F.softplus(nll-other.mean())
            if not torch.isfinite(objective):raise ValueError('Nonfinite training objective')
            objective.backward();optimizer.step();target_tokens+=len(values)
            writer.writerow(dict(iteration=step,nll=float(nll.detach()),objective=float(objective.detach()),target_tokens=target_tokens));handle.flush()
            if step%250==0 or step==len(items):training_plot(folder)
    for k,v in snapshot.items():
        if not torch.equal(v,model.lm.state_dict()[k].detach().cpu()):raise ValueError('Frozen decoder changed')
    save(folder/'final.pt',model,spec,dict(records_sha256=digest(root/'records.csv'),
        protocol_sha256=digest(root/'training_protocol.json'),optimizer_steps=len(items),target_tokens=target_tokens))
    # All models use the same independent test components; no checkpoint selection.
    evaluation=[r for r in rows if r['split']=='test']
    write_csv(folder/'scores.csv',score(model,features,evaluation))
    write_json(marker,dict(complete=True,spec=spec,optimizer_steps=len(items),target_tokens=target_tokens,
        auxiliary_forward_target_tokens=auxiliary_tokens,
        forward_passes=forwards,shared_initial_sha256=initial,initial_parity_error=parity_error,
        parameters=sum(p.numel() for p in parameters(model)),trainable_parameters=sum(p.numel() for p in parameters(model) if p.requires_grad),
        wall_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        training_schedule_sha256=digest(root/'schedules'/f'{seed}.json'),frozen_decoder_verified=bool(snapshot),
        public_pretrained_IgLM_weights=False,dropout_training=True,checkpoint_selection='fixed final update',
        files_sha256={n:digest(folder/n) for n in ('final.pt','scores.csv','training_curve.csv')}))


def training_plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    frame=pd.read_csv(folder/'training_curve.csv')
    fig,ax=plt.subplots(figsize=(6,3));ax.plot(frame.iteration,frame.nll,alpha=.2)
    ax.plot(frame.iteration,frame.nll.rolling(50,min_periods=1).mean())
    ax.set(xlabel='Optimizer updates',ylabel='Training chain NLL',title=folder.name)
    fig.tight_layout();temp=folder/'training_curve.tmp.png';fig.savefig(temp,dpi=130);plt.close(fig);temp.replace(folder/'training_curve.png')


def external(root,arm,seed):
    """Exploratory fixed-panel affinity scoring, never used for training selection."""
    import numpy as np
    from native_mechanism_common import affinity_correlations
    from controlled_antigen_model import ConditionalScorer
    cfg=read_json(root/'training_protocol.json')['settings']
    folder=root/'runs'/f'{arm}_s{seed}'
    done=read_json(folder/'complete.json')
    if digest(folder/'final.pt')!=done['files_sha256']['final.pt']:raise ValueError('Checkpoint changed')
    target=get_device('native exploratory affinity');torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    model,checkpoint=load(folder/'final.pt',target);features=Features(root,checkpoint['spec'],target,cfg['memory_tokens'])
    pairs=read_csv(root/'affinity_selection.csv')
    antigens={r['id']:r for r in read_csv(Path(cfg['binding_root'])/'antigens.csv')}
    scorer=ConditionalScorer(model,{})
    output=[];metrics=[];checks=[]
    with torch.inference_mode():
        for aid,row in antigens.items():
            selected=[p for p in pairs if p['antigen_id']==aid]
            seqs=sorted({p['heavy_sequence'] for p in selected})
            memory=features(row)
            scorer.antigens={aid:memory};scorer.prompts.clear()
            for chain in ('heavy','light'):
                sample=seqs[0]
                reference=float(iglm_token_nll(model,chain,sample,h_ag=memory).double().mean())
                for batch_size in (1,8):
                    scorer.prompts.clear();a=scorer.score(aid,chain,[sample]*batch_size)[0].cpu().double().numpy()/(len(sample)+1)
                    scorer.prompts.clear();b=scorer.score(aid,chain,[sample]*batch_size)[0].cpu().double().numpy()/(len(sample)+1)
                    error=float(max(np.max(abs(a-reference)),np.max(abs(a-b))))
                    if error>3e-6:raise ValueError('External scalar/cached parity failed')
                    checks.append(dict(target=row['family'],chain=chain,batch_size=batch_size,error=error))
            values={}
            for start in range(0,len(seqs),32):
                batch=seqs[start:start+32];totals=scorer.score(aid,'heavy',batch)[0].cpu().tolist()
                values.update({s:float(v)/(len(s)+1) for s,v in zip(batch,totals)})
            target_rows=[dict(query_id=p['query_id'],target=p['target'],affinity=float(p['affinity']),nll=values[p['heavy_sequence']]) for p in selected]
            output.extend(target_rows)
            nll=np.array([p['nll'] for p in target_rows]);aff=np.array([p['affinity'] for p in target_rows])
            metrics.append(dict(target=row['family'],pairs=len(target_rows),**affinity_correlations(nll,aff)))
    write_csv(folder/'affinity_scores.csv',output)
    write_json(folder/'affinity_complete.json',dict(complete=True,metrics=metrics,parity=checks,
        checkpoint_sha256=digest(folder/'final.pt'),selection_sha256=digest(root/'affinity_selection.csv'),
        scores_sha256=digest(folder/'affinity_scores.csv'),biological_ci_available=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=['train','external'])
    parser.add_argument('--root',required=True);parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();(train if args.job=='train' else external)(Path(args.root),args.arm,args.seed)
