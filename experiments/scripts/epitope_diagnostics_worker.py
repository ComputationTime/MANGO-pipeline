"""Checkpoint interventions chosen without structural contact information."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from affinity_study_common import digest,fingerprint,read_csv,read_json,write_csv,write_json
from epitope_diagnostics_common import masks,REGIONS


def verify(root):
    p=read_json(root/'protocol.json')
    for key in ('inputs_sha256','implementation_sha256','checkpoints_sha256'):
        for name,value in p[key].items():
            if digest(name)!=value:raise ValueError('Frozen diagnostic source changed: '+name)
    for name,value in read_json(root/'prepared.json')['files_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Diagnostic annotation changed: '+name)
    return p


def forward(model,sequences,contexts=None,attention=False):
    import torch.nn.functional as F
    device=next(model.lm.parameters()).device;maximum=max(map(len,sequences))+3
    ids=torch.zeros((len(sequences),maximum),dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    labels=torch.full_like(ids,-100)
    for i,s in enumerate(sequences):
        values=[model.vocab['[HEAVY]'],model.vocab['[HUMAN]'],*[model.vocab[a] for a in s],model.vocab['[SEP]']]
        ids[i,:len(values)]=torch.tensor(values,device=device);mask[i,:len(values)]=1;labels[i,2:len(values)]=ids[i,2:len(values)]
    kwargs={}
    if model.branch and contexts is not None:kwargs['encoder_hidden_states']=model.antigen_projection(contexts)
    captures={};hooks=[]
    if attention:
        if not model.branch or contexts is None:raise ValueError('Attention instrumentation needs antigen memory')
        for i,block in enumerate(model.lm.transformer.h):
            captures[i]={}
            def record(kind,index):
                def hook(module,inputs,output):captures[index][kind]=output.detach()
                return hook
            hooks.append(block.crossattention.q_attn.register_forward_hook(record('q',i)))
            hooks.append(block.crossattention.c_attn.register_forward_hook(record('kv',i)))
    try:
        # Keep the production attention backend and output_attentions setting.
        result=model.lm(input_ids=ids,attention_mask=mask,use_cache=False,**kwargs)
    finally:
        for hook in hooks:hook.remove()
    values=F.cross_entropy(result.logits[:,:-1].transpose(1,2),labels[:,1:],reduction='none')[:,1:]
    losses=[values[i,:len(s)+1].double().cpu().numpy() for i,s in enumerate(sequences)]
    attentions=None
    if attention:
        matrices=[]
        for i,block in enumerate(model.lm.transformer.h):
            module=block.crossattention;q=captures[i]['q'];k=captures[i]['kv'].split(module.split_size,dim=-1)[0]
            q=q.view(q.shape[0],q.shape[1],module.num_heads,module.head_dim).transpose(1,2).float()
            k=k.view(k.shape[0],k.shape[1],module.num_heads,module.head_dim).transpose(1,2).float()
            scores=torch.matmul(q,k.transpose(-1,-2))
            if module.scale_attn_weights:scores=scores/(module.head_dim**.5)
            if module.scale_attn_by_inverse_layer_idx:scores=scores/(i+1)
            matrices.append(scores.softmax(-1))
        # Queries 1..L+1 are the positions whose logits predict residues and EOS.
        matrix=torch.stack(matrices).mean(dim=(0,2))
        attentions=[matrix[i,1:len(s)+2].double().cpu().numpy() for i,s in enumerate(sequences)]
        if any(np.max(abs(a.sum(1)-1))>1e-5 for a in attentions):raise ValueError('Attention does not normalize')
    return losses,attentions


def summarize(values,region_masks):
    return {name:float(values[mask].mean()) if mask.any() else None for name,mask in region_masks.items()}


def main(root,arm,seed):
    from short_affinity_model import load
    from controlled_antigen_model import memory,mode
    from device_common import get_device
    root=Path(root);protocol=verify(root);cfg=protocol['settings'];reference=Path(cfg['reference_root']);old=protocol['reference_settings']
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    checkpoint=reference/'runs'/f'{arm}_s{seed}'/'final.pt';model,_=load(checkpoint,get_device('epitope/paratope diagnostics'));mode(model,False)
    features={};lengths={}
    for path,expected in read_json(reference/'embeddings_complete.json')['files_sha256'].items():
        if digest(path)!=expected:raise ValueError('Reference embedding changed')
        payload=torch.load(path,map_location='cpu',weights_only=False);key=payload['antigen_sha256']
        features[key]=memory(payload['embedding'],32)[0].to(next(model.lm.parameters()).device);lengths[key]=payload['embedding'].shape[0]
    regions=read_json(root/'regions.json')['sequences'];contacts=read_json(root/'contacts.json')['records']
    external=read_csv(root/'external_selection.csv');native=read_csv(root/'native_records.csv')
    antigens=read_csv(Path(old['binding_root'])/'antigens.csv');agbyid={r['id']:r for r in antigens}
    conditions=['correct','wrong_0','wrong_1','mean','zero','none'] if model.branch else ['none']
    output=[];maximum_parity=0.;existing={r['query_id']:float(r['nll']) for r in read_csv(checkpoint.parent/'external_scores.csv')}
    def context(original,condition,alternate=None):
        if not model.branch or condition=='none':return None
        original=features[fingerprint(original)]
        if condition.startswith('wrong'):return features[fingerprint(alternate)]
        if condition=='mean':return original.mean(0,keepdim=True).expand_as(original)
        if condition=='zero':return torch.zeros_like(original)
        return original
    with torch.inference_mode():
        for aid in sorted(agbyid):
            ag=agbyid[aid];subset=[r for r in external if r['antigen_id']==aid]
            donors=[agbyid[k] for k in sorted(agbyid) if k!=aid]
            for condition in conditions:
                donor=donors[int(condition[-1])] if condition.startswith('wrong') else None
                ctx=context(ag['resolved_ag_seq'],condition,donor['resolved_ag_seq'] if donor else None)
                for start in range(0,len(subset),cfg['batch_size']):
                    batch=subset[start:start+cfg['batch_size']];seqs=[r['heavy_sequence'] for r in batch]
                    losses,_=forward(model,seqs,ctx[None].expand(len(batch),-1,-1) if ctx is not None else None)
                    for row,values in zip(batch,losses):
                        scores=summarize(values,masks(regions[fingerprint(row['heavy_sequence'])]['labels']))
                        if condition==('correct' if model.branch else 'none'):
                            err=abs(scores['full']-existing[row['query_id']]);maximum_parity=max(maximum_parity,err)
                            if err>cfg['parity_tolerance']:raise ValueError('Reference/batched per-token parity failed')
                        output.append(dict(query_id=row['query_id'],target=row['target'],condition=condition,affinity=row['affinity'],**scores))
            print(arm,seed,ag['family'],'external complete',flush=True)
        native_scores=[];token_scores=[];attention_rows=[];occlusions=[];permutation_errors=[];eager_errors=[]
        for row in native:
            sequence=row['resolved_H_seq'];rm=masks(regions[fingerprint(sequence)]['labels']);geometry=contacts.get(row['id'])
            if geometry:
                if geometry['embedding_length']!=lengths[fingerprint(row['resolved_ag_seq'])]:raise ValueError('Coordinate/embedding mapping mismatch')
                p=geometry['contacts'][str(float(cfg['primary_contact_cutoff']))]['paratope']
                rm['paratope']=np.array([i in p for i in range(len(sequence)+1)])
            correct=None
            native_conditions=['correct','wrong','mean','zero','none'] if model.branch else ['none']
            for condition in native_conditions:
                ctx=context(row['resolved_ag_seq'],condition,row['wrong_antigen_sequence'])
                losses,_=forward(model,[sequence],ctx[None] if ctx is not None else None);values=losses[0]
                if condition=='correct':correct=values
                native_scores.append(dict(id=row['id'],split=row['split'],group=row['biological_group'],condition=condition,**summarize(values,rm)))
                for i,value in enumerate(values):token_scores.append(dict(id=row['id'],condition=condition,index=i,
                    residue=sequence[i] if i<len(sequence) else 'EOS',region=regions[fingerprint(sequence)]['labels'][i],nll=float(value)))
            if model.branch:
                ctx=features[fingerprint(row['resolved_ag_seq'])]
                shuffled,_=forward(model,[sequence],ctx.roll(7,0)[None]);permutation_errors.append(float(np.max(abs(shuffled[0]-correct))))
                if geometry:
                    # Masks are ALL 32 positions, fixed before any contact labels are read.
                    altered=ctx[None].repeat(32,1,1);positions=torch.arange(32,device=ctx.device)
                    altered[positions,positions]=ctx.mean(0)
                    losses,_=forward(model,[sequence]*32,altered)
                    baseline=summarize(correct,rm)
                    for i,values in enumerate(losses):
                        for region,value in summarize(values,rm).items():
                            if value is not None:occlusions.append(dict(id=row['id'],split=row['split'],group=row['biological_group'],token=i,region=region,delta_nll=value-baseline[region]))
                    losses,attn=forward(model,[sequence],ctx[None],attention=True)
                    error=float(np.max(abs(losses[0]-correct)));eager_errors.append(error)
                    if error>cfg['parity_tolerance']:raise ValueError('Passive attention recording changes likelihood')
                    for region,mask in rm.items():
                        if mask.any():attention_rows.append(dict(id=row['id'],split=row['split'],group=row['biological_group'],region=region,weights=json.dumps(attn[0][mask].mean(0).tolist())))
        if permutation_errors and max(permutation_errors)>cfg['token_parity_tolerance']:
            raise ValueError('Unexpected memory-token order dependence')
    directory=root/'runs'/f'{arm}_s{seed}';directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory/'external.csv',output);write_csv(directory/'native.csv',native_scores);write_csv(directory/'native_tokens.csv',token_scores)
    write_csv(directory/'attention.csv',attention_rows);write_csv(directory/'occlusions.csv',occlusions)
    files=['external.csv','native.csv','native_tokens.csv','attention.csv','occlusions.csv']
    write_json(directory/'complete.json',dict(complete=True,arm=arm,seed=seed,checkpoint_sha256=digest(checkpoint),
        reference_parity_max=maximum_parity,memory_permutation_max=max(permutation_errors,default=0.),
        attention_instrumentation_parity_max=max(eager_errors,default=0.),files_sha256={n:digest(directory/n) for n in files},
        no_fitting=True,smoke_only=cfg['smoke']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();main(Path(a.root),a.arm,a.seed)
