"""Matched training/evaluation for geometry by contact-supervision study."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import csv
import numpy as np
import torch
from affinity_study_common import digest,fingerprint,read_csv,read_json,write_csv,write_json
from interface_supervision_common import ARMS
from interface_supervision_model import InterfaceModel,balanced_contact_loss,save


def verify(root):
    p=read_json(root/'protocol.json')
    for field in ('inputs_sha256','implementation_sha256'):
        for name,value in p[field].items():
            if digest(name)!=value:raise ValueError('Frozen source changed: '+name)
    prepared=read_json(root/'prepared.json')
    for name,value in prepared['files_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Prepared source changed: '+name)
    return p,prepared


class Data:
    def __init__(self,root,target,arm):
        self.root=root;self.arm=arm;self.target=target;p,prepared=verify(root);self.cfg=p['settings'];self.prepared=prepared
        ref=Path(self.cfg['reference_root']);self.features={};self.graphs={};self.labels={}
        manifest=read_json(ref/'embeddings_complete.json')
        for path,value in manifest['files_sha256'].items():
            if digest(path)!=value:raise ValueError('Frozen PLM embedding changed')
            payload=torch.load(path,map_location='cpu',weights_only=False);key=payload['antigen_sha256']
            if key not in prepared['inputs']:continue
            info=prepared['inputs'][key]
            if digest(info['path'])!=info['sha256']:raise ValueError('Antigen input changed')
            raw=dict(np.load(info['path']));self.features[key]=payload['embedding'][raw['keep']].to(target)
            self.graphs[key]={name:torch.as_tensor(values,device=target) for name,values in raw.items() if name!='coords'}
        if set(self.features)!=set(prepared['inputs']):raise ValueError('Incomplete PLM/input matching')
        for rid,info in prepared['labels'].items():
            if digest(info['path'])!=info['sha256']:raise ValueError('Contact label changed')
            self.labels[rid]=dict(np.load(info['path']))

    def batch(self,model,examples,scrambled=False):
        sequences=[row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'] for row,chain in examples]
        n=len(examples);maximum=max(map(len,sequences))+3
        if maximum>model.base_config['n_positions']:raise ValueError('Target exceeds positional capacity')
        ids=torch.zeros((n,maximum),dtype=torch.long,device=self.target);mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
        for i,((row,chain),seq) in enumerate(zip(examples,sequences)):
            values=[model.vocab[f'[{chain.upper()}]'],model.vocab['[HUMAN]'],*[model.vocab[a] for a in seq],model.vocab['[SEP]']]
            ids[i,:len(values)]=torch.tensor(values,device=self.target);mask[i,:len(values)]=1;labels[i,2:len(values)]=ids[i,2:len(values)]
        kwargs={};lengths=[]
        if model.branch:
            keys=[fingerprint(row['resolved_ag_seq']) for row,chain in examples];lengths=[len(self.features[k]) for k in keys]
            m=max(lengths);k=self.cfg['neighbors']
            embeddings=torch.zeros((n,m,1280),device=self.target);agmask=torch.zeros((n,m),dtype=torch.long,device=self.target)
            neighbors=torch.zeros((n,m,k),dtype=torch.long,device=self.target);edges=torch.zeros((n,m,k,18),device=self.target)
            valid=torch.zeros((n,m,k),dtype=torch.bool,device=self.target)
            kind='scrambled' if scrambled and ARMS[self.arm][0]=='geometry' else ARMS[self.arm][0]
            for i,(key,length) in enumerate(zip(keys,lengths)):
                embeddings[i,:length]=self.features[key];agmask[i,:length]=1
                neighbors[i,:length]=self.graphs[key][kind+'_neighbors'];edges[i,:length]=self.graphs[key][kind+'_edges'];valid[i,:length]=self.graphs[key][kind+'_valid']
            kwargs=dict(embeddings=embeddings,antigen_mask=agmask,neighbors=neighbors,edges=edges,neighbor_valid=valid)
        logits,contacts=model(ids,mask,**kwargs)
        target=labels[:,1:];validtargets=target!=-100
        values=torch.nn.functional.cross_entropy(logits[:,:-1].transpose(1,2),target,reduction='none')
        sums=(values*validtargets).sum(1);counts=validtargets.sum(1);means=sums/counts
        return dict(means=means,sums=sums,counts=counts,tokens=[values[i,1:len(s)+2] for i,s in enumerate(sequences)],contacts=contacts,
                    lengths=lengths,sequences=sequences)

    def supervised_loss(self,output,examples):
        if output['contacts'] is None:return output['means']*0
        values=[]
        for i,(row,chain) in enumerate(examples):
            label=torch.as_tensor(self.labels[row['id']][chain],device=self.target)
            score=output['contacts'][i,:len(output['sequences'][i]),:output['lengths'][i]]
            if score.shape!=label.shape:raise ValueError('Contact label/output shape differs')
            values.append(balanced_contact_loss(score,label))
        return torch.stack(values)


def model_hash(model,shared=False):
    h=hashlib.sha256()
    for name,value in sorted((model.lm if shared else model).state_dict().items()):
        if shared:
            if '.crossattention.' in name or '.ln_cross_attn.' in name:continue
            if '.mlp.c_fc.weight' in name:value=value[:,:model.base_config['n_inner']]
            elif '.mlp.c_fc.bias' in name or '.mlp.c_proj.weight' in name:value=value[:model.base_config['n_inner']]
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit(root,arm,seed):
    from device_common import get_device
    from short_affinity_worker import curve_plot
    root=Path(root);p,prepared=verify(root);cfg=p['settings'];directory=root/'runs'/f'{arm}_s{seed}';directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');target=get_device('interface supervision fit')
    model=InterfaceModel(seed,arm!='iglm_ce',cfg).to(target);model.contact_to_backbone=ARMS[arm][1];initial=model_hash(model);shared=model_hash(model,True)
    data=Data(root,target,arm);rows=read_csv(root/'records.csv');byid={r['id']:r for r in rows};pair=read_json(root/'ranking_pair.json')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);torch.manual_seed(seed+1000000);model.train()
    points=[];exposure=[];total_tokens=0;started=time.monotonic();iteration=0
    with (directory/'training_curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['iteration','phase','objective','nll','contact_loss']);writer.writeheader()
        for phase in ('pretrain','adapt'):
            schedule=read_json(root/'schedules'/f'{phase}_{seed}.json')['items']
            for item in schedule:
                iteration+=1;examples=[(byid[x['id']],x['chain']) for x in item]
                if phase=='adapt':examples += [(byid[pair[k]['id']],'heavy') for k in ('strong','weak')]
                if any(r['split']!='train' for r,c in examples):raise ValueError('Holdout entered training')
                optimizer.zero_grad(set_to_none=True);out=data.batch(model,examples);contact=data.supervised_loss(out,examples)
                if phase=='adapt':
                    w=cfg['pair_ce_weight'];ce=(1-w)*out['means'][:-2].mean()+w*out['means'][-2:].mean()
                    auxiliary=(1-w)*contact[:-2].mean()+w*contact[-2:].mean()
                else:ce=out['means'].mean();auxiliary=contact.mean()
                # CE controls train the same probe head on detached features.
                loss=ce+(cfg['contact_weight'] if model.branch else 0.)*auxiliary
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();optimizer.step();total_tokens+=int(out['counts'].sum());exposure.extend((r['id'],c) for r,c in examples)
                point=dict(iteration=iteration,phase=phase,objective=float(loss.detach()),nll=float(ce.detach()),contact_loss=float(auxiliary.detach()))
                writer.writerow(point);handle.flush();points.append(point)
                if iteration%100==0:
                    curve_plot(directory/'training_curve.png',[(x['iteration'],x['objective']) for x in points]);print(arm,seed,iteration,point,flush=True)
    save(directory/'final.pt',model,dict(arm=arm,seed=seed,settings=cfg,protocol_sha256=digest(root/'protocol.json')))
    curve_plot(directory/'training_curve.png',[(x['iteration'],x['objective']) for x in points])
    head=sum(x.numel() for x in model.contact_head.parameters()) if model.branch else 0
    write_json(directory/'complete.json',dict(complete=True,arm=arm,seed=seed,updates=iteration,initial_sha256=initial,shared_decoder_sha256=shared,
        exposure_sha256=fingerprint(json.dumps(exposure)),target_tokens=total_tokens,
        parameters=sum(x.numel() for x in model.parameters()),generation_parameters=sum(x.numel() for x in model.parameters())-head,
        checkpoint_sha256=digest(directory/'final.pt'),wall_seconds=time.monotonic()-started,smoke_only=cfg['smoke'],contact_gradients_to_backbone=ARMS[arm][1]))


def load_model(path,target):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=InterfaceModel(payload['seed'],payload['arm']!='iglm_ce',payload['settings'],payload['base_config'])
    model.contact_to_backbone=ARMS[payload['arm']][1]
    model.load_state_dict(payload['state'],strict=True);return model.to(target).eval()


def score(root,arm,seed):
    from device_common import get_device
    from chain_examples import example_chains
    root=Path(root);p,prepared=verify(root);cfg=p['settings'];directory=root/'runs'/f'{arm}_s{seed}';marker=read_json(directory/'complete.json')
    if digest(directory/'final.pt')!=marker['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');target=get_device('interface supervision evaluation')
    model=load_model(directory/'final.pt',target);data=Data(root,target,arm);annotations=read_json(Path(cfg['diagnostic_root'])/'regions.json')['sequences']
    rows=read_csv(root/'records.csv');native=[];contact_outputs=[];parity=[]
    with torch.inference_mode():
        test=[(r,'heavy') for r in rows if r['split']=='test' and 'heavy' in example_chains(r)]
        sample=test[0];single=data.batch(model,[sample])['means'][0]
        batch=data.batch(model,[sample]*8)['means'];error=float((batch-single).abs().max())
        if error>cfg['parity_tolerance']:raise ValueError('Padded/batched NLL parity failed')
        parity.append(error)
        # Ensure padding with a different antigen length cannot alter this example.
        other=max(test,key=lambda x:len(x[0]['resolved_ag_seq']))
        padded=data.batch(model,[sample,other])['means'][0];error=float(abs(padded-single))
        if error>cfg['parity_tolerance']:raise ValueError('Antigen padding changes likelihood')
        parity.append(error)
        variants=['correct','scrambled'] if ARMS[arm][0]=='geometry' else ['correct']
        for variant in variants:
            for start in range(0,len(test),4):
                examples=test[start:start+4];out=data.batch(model,examples,scrambled=variant=='scrambled')
                for i,(row,chain) in enumerate(examples):
                    seq=row['resolved_H_seq'];labels=annotations[fingerprint(seq)]['labels'];losses=out['tokens'][i].double().cpu().numpy()
                    cdr=np.array([x.startswith('cdr') for x in labels]);cdr3=np.array([x=='cdr3' for x in labels])
                    native.append(dict(id=row['id'],group=row['biological_group'],variant=variant,nll=float(losses.mean()),
                        loss_sum=float(losses.sum()),ntokens=len(losses),cdr_nll=float(losses[cdr].mean()) if cdr.any() else '',
                        cdr3_nll=float(losses[cdr3].mean()) if cdr3.any() else ''))
                    if model.branch:
                        logits=out['contacts'][i,:len(seq),:out['lengths'][i]].cpu().numpy();truth=data.labels[row['id']]['heavy']
                        contact_outputs.append(dict(id=row['id'],group=row['biological_group'],variant=variant,logits=logits,truth=truth))
        external=[];antigens=read_csv(root/'antigens.csv');selected=read_csv(root/'external_selection.csv')
        for ag in antigens:
            subset=[r for r in selected if r['antigen_id']==ag['id']]
            for start in range(0,len(subset),cfg['evaluation_batch_size']):
                batch=subset[start:start+cfg['evaluation_batch_size']];examples=[(dict(ag,resolved_H_seq=r['heavy_sequence']),'heavy') for r in batch]
                out=data.batch(model,examples)
                for i,r in enumerate(batch):
                    contact_score=None
                    if model.branch:
                        seq=r['heavy_sequence'];lab=annotations[fingerprint(seq)]['labels'][:-1];indices=[j for j,x in enumerate(lab) if x.startswith('cdr')]
                        logits=out['contacts'][i,indices,:out['lengths'][i]]
                        contact_score=float((logits.logsumexp(-1)-np.log(out['lengths'][i])).mean())
                    external.append(dict(query_id=r['query_id'],target=r['target'],affinity=float(r['affinity']),nll=float(out['means'][i]),contact_score=contact_score))
            print(arm,seed,ag['family'],'affinity scoring complete',flush=True)
    write_csv(directory/'native.csv',native);write_csv(directory/'external.csv',external)
    contact_dir=directory/'contacts';contact_dir.mkdir(exist_ok=True);files={}
    for row in contact_outputs:
        path=contact_dir/f'{fingerprint(row["id"])}_{row["variant"]}.npz'
        np.savez_compressed(path,logits=row['logits'],truth=row['truth']);files[str(path)]=digest(path)
    write_json(directory/'evaluation.json',dict(complete=True,arm=arm,seed=seed,checkpoint_sha256=digest(directory/'final.pt'),
        native_sha256=digest(directory/'native.csv'),external_sha256=digest(directory/'external.csv'),contacts=files,parity=parity,smoke_only=cfg['smoke']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['fit','score']);p.add_argument('--root',required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();(fit if a.job=='fit' else score)(Path(a.root),a.arm,a.seed)
