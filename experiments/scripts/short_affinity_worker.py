"""Deadline-bounded fits and frozen-checkpoint antigen interventions."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from affinity_study_common import read_json,read_csv,write_json,write_csv,digest,fingerprint,freeze
from affinity_study_worker import evaluate as legacy_evaluate,restore
from antigen_diagnostics import iglm_token_nll,mango_token_nll
from chain_examples import example_chains
from controlled_antigen_model import mode,device,parity
from device_common import get_device
from model_common import load_embedding
from pairing_prefix_score_cache import SharedPrefixChainScorer
from short_affinity_model import build,model_parameters,save,load,cdr_loss,preference


class Deadline:
    def __init__(self,seconds):self.started=time.monotonic();self.seconds=seconds
    def check(self):
        if time.monotonic()-self.started>=self.seconds:raise TimeoutError('GPU stage time limit')


class Scoring:
    def __init__(self,root,model,deadline):
        self.root=root;self.model=model;self.deadline=deadline;self.forward_passes=0;self.cache={}
        self.kind='iglm_cross' if model.branch else 'iglm'
        self.embedding_files=read_json(root/'embeddings_complete.json')['files_sha256']

    def antigen(self,row,constant=False):
        if not self.model.branch:return None
        key=fingerprint(row['resolved_ag_seq'])
        if key not in self.cache:
            path=self.root/'embeddings'/f'{key}.pt'
            expected=self.embedding_files[str(path)]
            if digest(path)!=expected:raise ValueError('Frozen antigen features changed')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            self.cache[key]=payload['embedding'][None]
        tensor=self.cache[key].to(next(self.model.lm.parameters()).device)
        return torch.zeros_like(tensor) if constant else tensor

    def losses(self,row,chain,variant='correct',training=False,donor=None):
        self.deadline.check();self.forward_passes+=1
        context=self.antigen(donor or row,variant=='constant') if variant!='no_context' else None
        return iglm_token_nll(self.model,chain,row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'],context)


def evaluate(scorer,rows,regions):
    values=legacy_evaluate(scorer,rows,regions,scorer.model.branch)
    groups={r['id']:r['biological_group'] for r in rows}
    for r in values:r['group']=groups[r['id']]
    if scorer.model.branch:
        with torch.inference_mode():
            for row in rows:
                for chain in example_chains(row):
                    losses=scorer.losses(row,chain,'constant').double().cpu().numpy()
                    seq=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'];labels=regions['sequences'][fingerprint(seq)]['labels']
                    cdr=np.array([x.startswith('cdr') for x in labels]);cdr3=np.array([x=='cdr3' for x in labels])
                    values.append(dict(id=row['id'],chain=chain,split=row['split'],group=row['biological_group'],variant='constant',
                        loss_sum=float(losses.sum()),ntokens=len(losses),nll=float(losses.mean()),cdr_sum=float(losses[cdr].sum()),cdr_tokens=int(cdr.sum()),
                        cdr3_sum=float(losses[cdr3].sum()),cdr3_tokens=int(cdr3.sum())))
    return values


def curve_plot(path,points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([p[0] for p in points],[p[1] for p in points],lw=.8)
    ax.set(xlabel='Optimizer updates',ylabel='Training objective');fig.tight_layout()
    temp=path.with_suffix('.tmp.png');fig.savefig(temp,dpi=140);plt.close(fig);temp.replace(path)


def train(root,spec_path):
    cfg=read_json(root/'protocol.json')['settings'];spec=read_json(spec_path);directory=root/'runs'/spec['id'];directory.mkdir(parents=True,exist_ok=True)
    freeze(directory/'spec.json',spec);deadline=Deadline(cfg['budgets']['fit'])
    try:
        audit=read_json(root/'cohort_audit.json')
        if not audit['feasible'] or audit['files_sha256']['records.csv']!=digest(root/'records.csv'):raise ValueError('Unverified cohort')
        embedding=read_json(root/'embeddings_complete.json')
        if embedding['records_sha256']!=digest(root/'records.csv'):raise ValueError('Embedding cohort changed')
        rows=read_csv(root/'records.csv');by_id={r['id']:r for r in rows};regions=read_json(root/'regions.json')
        if regions['records_sha256']!=digest(root/'records.csv'):raise ValueError('Region cohort changed')
        if spec['objective'] in ('ranking','affinity_ce'):
            ra=read_json(root/'ranking_audit.json')
            if not ra['feasible']:raise ValueError('Missing independent affinity labels')
            for name,value in ra['files_sha256'].items():
                if digest(root/name)!=value:raise ValueError('Affinity data changed')
            all_pairs=[r for r in read_csv(root/'ranking_pairs.csv') if r['split']=='train'];rng=random.Random(spec['seed']);items=[]
            for i in range(cfg['updates']):items.append(rng.choice(all_pairs))
        else:
            schedule=read_json(root/'schedules'/f'{spec["seed"]}.json')
            if schedule['records_sha256']!=digest(root/'records.csv'):raise ValueError('Schedule changed')
            items=schedule['items']
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
        model=build(spec);target=get_device('short affinity fit');device(model,target)
        whole=hashlib.sha256();shared=hashlib.sha256()
        for k,v in sorted(model.lm.state_dict().items()):
            v=v.detach().cpu();whole.update(k.encode());whole.update(v.contiguous().numpy().tobytes())
            if '.crossattention.' in k or '.ln_cross_attn.' in k:continue
            if '.mlp.c_fc.weight' in k:v=v[:,:model.base_config['n_inner']]
            elif '.mlp.c_fc.bias' in k or '.mlp.c_proj.weight' in k:v=v[:model.base_config['n_inner']]
            shared.update(k.encode());shared.update(v.contiguous().numpy().tobytes())
        initial_hash=whole.hexdigest();shared_hash=shared.hexdigest()
        optimizer=torch.optim.Adam(model_parameters(model),lr=cfg['lr']);scorer=Scoring(root,model,deadline)
        torch.manual_seed(spec['seed']+2000000);mode(model,True);points=[];tokens=0;membership=[];exposure=[]
        with (directory/'training_curve.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['iteration','phase','objective','nll']);writer.writeheader()
            for index,item in enumerate(items,1):
                deadline.check();optimizer.zero_grad(set_to_none=True)
                if spec['objective'] in ('ranking','affinity_ce'):
                    a=by_id[item['strong']];b=by_id[item['weak']]
                    if a['split']!='train' or b['split']!='train' or a['resolved_ag_seq']!=b['resolved_ag_seq']:raise ValueError('Invalid ranking pair')
                    strong=scorer.losses(a,'heavy');weak=scorer.losses(b,'heavy');ce=.5*(strong.mean()+weak.mean());loss=ce
                    if spec['objective']=='ranking':loss=loss+cfg['ranking_weight']*preference(strong,weak)
                    tokens+=len(strong)+len(weak);membership.extend([(a['id'],'heavy'),(b['id'],'heavy')])
                else:
                    row=by_id[item['id']];chain=item['chain'];context=by_id[item['shuffled_id']] if spec['shuffled'] else row
                    losses=scorer.losses(row,chain,donor=context);ce=losses.mean();loss=ce
                    if spec['objective']=='cdr':
                        seq=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'];loss=cdr_loss(losses,regions['sequences'][fingerprint(seq)]['labels'])
                    if spec['objective']=='pairing':
                        alternate=by_id[item['decoy_shuffled' if spec['shuffled'] else 'decoy_correct']]
                        wrong=scorer.losses(row,chain,donor=alternate);loss=loss+cfg['pairing_weight']*preference(losses,wrong)
                    tokens+=len(losses);membership.append((row['id'],chain));exposure.append(dict(step=index,id=row['id'],chain=chain,memory_id=context['id']))
                if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
                loss.backward();optimizer.step();points.append((index,float(loss.detach())))
                writer.writerow(dict(iteration=index,phase='train',objective=points[-1][1],nll=float(ce.detach())));handle.flush()
                if index%250==0:
                    save(directory/f'step_{index}.pt',model,spec,dict(updates=index));curve_plot(directory/'training_curve.png',points)
        if len(points)!=cfg['updates']:raise ValueError('Training schedule length differs from budget')
        mode(model,False);deadline.check()
        scores=evaluate(scorer,[r for r in rows if r['split']=='val'],regions);write_csv(directory/'scores.csv',scores)
        if spec['objective'] in ('ranking','affinity_ce'):
            measured=[r for r in read_csv(root/'affinities.csv') if r['split']=='val'];lookup={(r['id'],r['chain']):r for r in scores if r['variant']=='correct'}
            write_csv(directory/'affinity_scores.csv',[dict(m,nll=lookup[m['id'],'heavy']['nll']) for m in measured])
        checks=[]
        for chain in ('heavy','light'):
            sample=next((r for r in rows if r['split']=='val' and chain in example_chains(r)),None)
            if sample:
                deadline.check();seq=sample['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
                checks.append(parity(model,scorer.antigen(sample),[seq,seq[:-1]],chain,cfg['parity_tolerance']))
        save(directory/'final.pt',model,spec,dict(updates=len(points),records_sha256=digest(root/'records.csv'),protocol_sha256=digest(root/'protocol.json')))
        curve_plot(directory/'training_curve.png',points);write_csv(directory/'exposure.csv',exposure)
        value=dict(status='complete',spec=spec,updates=len(points),target_tokens=tokens,forward_passes=scorer.forward_passes,
            training_membership_sha256=fingerprint(json.dumps(membership)),initial_sha256=initial_hash,shared_initial_sha256=shared_hash,
            parameters=sum(p.numel() for p in model_parameters(model)),trainable_parameters=sum(p.numel() for p in model_parameters(model) if p.requires_grad),
            checkpoint_sha256=digest(directory/'final.pt'),scores_sha256=digest(directory/'scores.csv'),
            wall_seconds=time.monotonic()-deadline.started,parity=checks,dropout_training=True)
        if value['wall_seconds']>cfg['budgets']['fit']:raise TimeoutError('Fit exceeded fixed wall-time ceiling')
        write_json(directory/'complete.json',value);write_json(directory/'status.json',value)
    except TimeoutError as error:
        write_json(directory/'status.json',dict(status='incomplete',reason='fit_time_limit',error=str(error),spec=spec,
            wall_seconds=time.monotonic()-deadline.started))


def diagnostic(root,run):
    cfg=read_json(root/'protocol.json')['settings'];selection=read_csv(root/'diagnostic_selection.csv');old=Path(cfg['reference_root'])
    path=old/'runs'/run/'best.pt'
    if run not in read_json(old/'selection.json')['external_runs']:raise ValueError('Not a frozen selected checkpoint')
    if digest(path)!=read_json(path.parent/'complete.json')['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    from train_antigen_diagnostic import set_training
    target=get_device('short frozen diagnostic');torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    model,payload=restore(path,target);kind=payload['spec']['kind'];set_training(model,kind,False)
    binding=Path(cfg['binding_root']);antigens={r['id']:load_embedding(str(binding/'embeddings/esm2/test'/f'{r["id"]}.pt')).to(target) for r in read_csv(binding/'antigens.csv')}
    scorer=SharedPrefixChainScorer(model,'iglm' if kind=='iglm' else 'mango',antigens)
    sequences=sorted({r['heavy_sequence'] for r in selection});cdr={r['heavy_sequence']:json.loads(r['cdr3_indices']) for r in selection};values={};checks=[]
    with torch.inference_mode():
        for aid,antigen in antigens.items():
            samples=sequences[:2];reference=[float((iglm_token_nll(model,'heavy',s) if kind=='iglm' else mango_token_nll(model,antigen,'heavy',s)).double().mean()) for s in samples]
            for size in (1,8,64):
                batch=[samples[i%len(samples)] for i in range(size)];first=scorer.score(aid,'heavy',batch)[0].cpu().numpy();scorer.prompts.clear();second=scorer.score(aid,'heavy',batch)[0].cpu().numpy()
                error=max(max(abs(first[i]/(len(s)+1)-reference[i%len(samples)]),abs(first[i]-second[i])/(len(s)+1)) for i,s in enumerate(batch))
                if error>cfg['parity_tolerance']:raise ValueError('Frozen diagnostic parity failed')
                checks.append(dict(antigen_id=aid,batch=size,error=float(error)))
            scorer.prompts.clear()
            for start in range(0,len(sequences),64):
                batch=sequences[start:start+64];full,region,_=scorer.score(aid,'heavy',batch,[cdr[s] for s in batch])
                for s,a,b in zip(batch,full.cpu().tolist(),region.cpu().tolist()):values[aid,s]=(a/(len(s)+1),b/len(cdr[s]))
    invariant=max(np.ptp([values[aid,s][0] for aid in antigens]) for s in sequences)
    if kind=='iglm' and invariant>cfg['parity_tolerance']:raise ValueError('Unconditional model depends on antigen')
    output=[]
    for r in selection:
        seq=r['heavy_sequence'];aid=r['antigen_id'];full,region=values[aid,seq]
        alternatives=[values[a,seq][0] for a in antigens if a!=aid]
        output.append(dict(query_id=r['query_id'],target=r['target'],antigen_id=aid,affinity=float(r['affinity']),
            nll=full,cdr3_nll=region,residual=0. if kind=='iglm' else float(np.mean(alternatives)-full),perplexity=math.exp(full)))
    directory=root/'diagnostics';directory.mkdir(exist_ok=True);write_csv(directory/f'{run}.csv',output)
    write_json(directory/f'{run}.complete.json',dict(status='complete',run=run,checkpoint_sha256=digest(path),
        selection_sha256=digest(root/'diagnostic_selection.csv'),scores_sha256=digest(directory/f'{run}.csv'),
        parity=checks,antigen_invariance_max=float(invariant),biological_ci_available=False,smoke_only=bool(cfg.get('smoke'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['train','diagnostic']);p.add_argument('--root',required=True);p.add_argument('--spec');p.add_argument('--run')
    a=p.parse_args();root=Path(a.root)
    if a.job=='train':train(root,a.spec)
    else:diagnostic(root,a.run)
