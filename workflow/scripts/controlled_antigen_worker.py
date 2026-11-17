"""Isolated jobs for controlled scratch/adapter comparisons and evaluation repair."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import time
import math

import torch
import numpy as np

from affinity_study_common import digest,freeze,read_csv,read_json,write_csv,write_json,fingerprint
from controlled_antigen_model import build,parameters,mode,device,memory,save,load,parity,ConditionalScorer
from affinity_study_worker import Scoring as LegacyScoring,evaluate as legacy_evaluate,ordinary_nll,training_plot
from antigen_diagnostics import iglm_token_nll,mango_token_nll
from chain_examples import example_chains
from device_common import get_device


class Scoring(LegacyScoring):
    def __init__(self,root,model,spec,target,rows):
        super().__init__(root,model,'iglm_cross' if model.branch else 'iglm',target,
                         False,rows,spec['seed'])
        self.tokens=read_json(root/'protocol.json')['settings']['memory_tokens'];self.constant=spec['memory']=='constant'
        if spec['memory']=='shuffled':
            from asd_scaling_studies import maximal_record_shuffle
            for chain in ('heavy','light'):
                members=[r for r in rows if r['split']=='train' and chain in example_chains(r)]
                mapping=maximal_record_shuffle(members,spec['seed'])
                if any(r['resolved_ag_seq']==mapping[r['id']]['resolved_ag_seq'] for r in members):
                    raise ValueError('A shuffled training identity was retained')
                self.donors.update({(r['id'],chain):mapping[r['id']] for r in members})

    def losses(self,row,chain,variant='correct',training=False,donor=None):
        if training and self.donors:donor=self.donors[row['id'],chain]
        return super().losses(row,chain,variant,training=False,donor=donor)

    def antigen(self,row,variant='correct',training=False):
        value=super().antigen(row,variant,training)
        return memory(value,self.tokens,self.constant) if value is not None else None


def evaluate(scorer,rows,regions,variants=False):
    scores=legacy_evaluate(scorer,rows,regions,variants)
    groups={r['id']:r['biological_group'] for r in rows}
    for row in scores:row['group']=groups[row['id']]
    return scores


def train(root,spec_path):
    settings=read_json(root/'protocol.json')['settings'];spec=read_json(spec_path)
    directory=root/'runs'/spec['id'];directory.mkdir(parents=True,exist_ok=True);freeze(directory/'spec.json',spec)
    if (directory/'complete.json').exists():
        done=read_json(directory/'complete.json')
        if digest(directory/'best.pt')!=done['checkpoint_sha256'] or digest(directory/'scores.csv')!=done['scores_sha256']:
            raise ValueError('Completed controlled run changed')
        return
    audit=read_json(root/'cohort_audit.json')
    if not audit['feasible'] or digest(root/'records.csv')!=audit['files_sha256']['records.csv']:raise ValueError('Unverified cohort')
    rows=read_csv(root/'records.csv');regions=read_json(root/'regions.json')
    if regions['records_sha256']!=digest(root/'records.csv'):raise ValueError('Region records changed')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='val']
    source=None
    if spec['source']:
        path=root/'runs'/spec['source']/'best.pt'
        if digest(path)!=read_json(path.parent/'complete.json')['checkpoint_sha256']:raise ValueError('Source changed')
        source=torch.load(path,map_location='cpu',weights_only=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');target=get_device('controlled antigen training')
    model=build(spec['seed'],spec['branch'],spec['frozen'],source=source['weights']['lm'] if source else None)
    device(model,target)
    frozen={k:v.detach().cpu().clone() for k,v in model.lm.state_dict().items()
            if spec['frozen'] and '.crossattention.' not in k and '.ln_cross_attn.' not in k}
    # Record the shared initial tensors independently of added capacity.
    shared={k:v.detach().cpu() for k,v in model.lm.state_dict().items() if '.crossattention.' not in k and '.ln_cross_attn.' not in k}
    import hashlib
    init_hash=hashlib.sha256(b''.join(k.encode()+v.contiguous().numpy().tobytes() for k,v in sorted(shared.items()))).hexdigest()
    scorer=Scoring(root,model,spec,target,rows)
    initial=evaluate(scorer,validation,regions)
    if source and abs(ordinary_nll(initial)-source['validation_nll'])>settings['parity_tolerance']:
        raise ValueError('Zero-output initialization failed to preserve source IgLM')
    optimizer=torch.optim.Adam([p for p in parameters(model) if p.requires_grad],lr=settings['lr'])
    torch.manual_seed(spec['seed']+2000000)
    examples=[(r,c) for r in training for c in example_chains(r)]
    if scorer.donors:
        write_csv(directory/'shuffle.csv',[dict(id=r['id'],chain=c,donor=scorer.donors[r['id'],c]['id']) for r,c in examples])
    membership=fingerprint(str(sorted((r['id'],c) for r,c in examples)))
    best=math.inf;history=[];iteration=0;tokens=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with (directory/'training_curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['iteration','epoch','phase','objective','nll']);writer.writeheader()
        for epoch in range(settings['epochs']):
            ordered=list(examples);random.Random(spec['seed']+epoch).shuffle(ordered)
            mode(model,True)  # Frozen parameters retain the SAME dropout policy.
            for row,chain in ordered:
                optimizer.zero_grad(set_to_none=True);losses=scorer.losses(row,chain,training=True);loss=losses.mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite CE')
                loss.backward();optimizer.step();iteration+=1;tokens+=len(losses)
                writer.writerow(dict(iteration=iteration,epoch=epoch,phase='train',objective=float(loss.detach()),nll=float(loss.detach())))
                handle.flush()
            scores=evaluate(scorer,validation,regions);nll=ordinary_nll(scores)
            history.append(dict(epoch=epoch,validation_nll=nll,validation_heavy_nll=ordinary_nll(scores,'heavy')))
            metadata=dict(epoch=epoch,validation_nll=nll,records_sha256=digest(root/'records.csv'),protocol_sha256=digest(root/'protocol.json'))
            save(directory/'latest.pt',model,spec,metadata)
            if nll<best:best=nll;save(directory/'best.pt',model,spec,metadata)
            writer.writerow(dict(iteration=iteration,epoch=epoch,phase='val',objective=nll,nll=nll));handle.flush()
            write_json(directory/'epochs.json',history);training_plot(directory,history,spec['id'])
    for key,value in frozen.items():
        if not torch.equal(value,model.lm.state_dict()[key].detach().cpu()):raise ValueError('Frozen decoder changed')
    forwards=scorer.forward_passes;del scorer,model
    model,checkpoint=load(directory/'best.pt',target);scorer=Scoring(root,model,spec,target,rows);scorer.donors={}
    selected=[r for r in rows if r['split']==('val' if spec['phase']=='pilot' else 'test')]
    scores=evaluate(scorer,selected,regions,True);write_csv(directory/'scores.csv',scores)
    checks=[]
    for chain in ('heavy','light'):
        sample=next(r for r in validation if chain in example_chains(r));seq=sample['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
        checks.append(parity(model,scorer.antigen(sample),[seq,seq[:-1]],chain,settings['parity_tolerance']))
    if scorer.donors:raise ValueError('Evaluation accidentally uses shuffled training donors')
    write_json(directory/'complete.json',dict(complete=True,spec=spec,checkpoint_sha256=digest(directory/'best.pt'),
        scores_sha256=digest(directory/'scores.csv'),shared_initial_sha256=init_hash,validation_nll=checkpoint['validation_nll'],
        development_heavy_nll=ordinary_nll(scores,'heavy'),training_membership_sha256=membership,
        optimizer_steps=iteration,target_tokens=tokens,forward_passes=forwards,wall_seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in parameters(model)),trainable_parameters=sum(p.numel() for p in parameters(model) if p.requires_grad),
        frozen_decoder_verified=bool(frozen),dropout_training=True,scalar_parity=checks,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def external_inputs(root):
    import sqlite3
    cfg=read_json(root/'protocol.json')['settings'];source=Path(cfg['binding_root'])
    queries={r['id']:r for r in read_csv(source/'heldout_antibodies.csv')}
    with sqlite3.connect(f'file:{(source/"measurements.sqlite").resolve()}?mode=ro',uri=True) as db:
        pairs=db.execute('SELECT query_id,antigen_id,reliable_sum/reliable_count FROM pairs WHERE reliable_count>0').fetchall()
    antigens={r['id']:r['family'] for r in read_csv(source/'antigens.csv')}
    if cfg.get('smoke'):
        pairs=[p for aid in antigens for p in [x for x in pairs if x[1]==aid][:8]]
    return cfg,source,queries,pairs,antigens


def run_external(root,run,amendment=False):
    """One checkpoint per process; deterministic resumable inference only."""
    import pandas as pd
    from model_common import load_embedding
    from train_antigen_diagnostic import set_training
    from pairing_prefix_score_cache import SharedPrefixChainScorer
    cfg,source,queries,pairs,antigens=external_inputs(root)
    output=root/('amendment_v1' if amendment else 'external');output.mkdir(parents=True,exist_ok=True)
    marker=output/f'{run}.complete.json'
    target=get_device('controlled external evaluation');torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    if amendment:
        from affinity_study_worker import restore
        old=Path(cfg['reference_root']);path=old/'runs'/run/'best.pt'
        if run not in read_json(old/'selection.json')['external_runs']:raise ValueError('Not in frozen v1 selection')
        if digest(path)!=read_json(path.parent/'complete.json')['checkpoint_sha256']:raise ValueError('V1 checkpoint changed')
        model,checkpoint=restore(path,target);kind=checkpoint['spec']['kind'];set_training(model,kind,False)
        scorer=SharedPrefixChainScorer(model,'iglm' if kind=='iglm' else 'mango',{})
    else:
        path=root/'runs'/run/'best.pt'
        if digest(path)!=read_json(path.parent/'complete.json')['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        model,checkpoint=load(path,target);scorer=ConditionalScorer(model,{})
    expected=dict(checkpoint_sha256=digest(path),source_measurements_sha256=digest(source/'measurements.sqlite'),
                  queries_sha256=digest(source/'heldout_antibodies.csv'),antigen_records_sha256=digest(source/'antigens.csv'),
                  embeddings_marker_sha256=digest(source/'embeddings/esm2/.batch_complete.json'),
                  smoke_only=bool(cfg.get('smoke')),implementation_sha256=digest(__file__))
    if marker.exists():
        record=read_json(marker)
        if record['inputs']!=expected or digest(output/f'{run}_pairs.csv.gz')!=record['scores_sha256']:raise ValueError('External provenance changed')
        return
    checks=[];rows=[]
    with torch.inference_mode():
        for aid,family in antigens.items():
            antigen=load_embedding(str(source/'embeddings/esm2/test'/f'{aid}.pt')).to(target)
            if not amendment:
                antigen=memory(antigen,cfg['memory_tokens'],checkpoint['spec']['memory']=='constant')
            scorer.antigens={aid:antigen};scorer.prompts.clear()
            comparisons=[(q,aff) for q,a,aff in pairs if a==aid]
            seqs=sorted({queries[q]['heavy_sequence'] for q,a in comparisons})
            def scalar(seq,chain):
                if amendment:
                    return iglm_token_nll(model,chain,seq) if kind=='iglm' else mango_token_nll(model,antigen,chain,seq)
                return iglm_token_nll(model,chain,seq,h_ag=antigen if model.branch else None)
            for chain in ('heavy','light'):
                samples=seqs[:2] if chain=='heavy' else sorted({queries[q]['light_sequence'] for q,a in comparisons if queries[q]['light_sequence']})[:2]
                if not samples:continue
                references=[float(scalar(s,chain).double().mean()) for s in samples]
                for size in (1,8,64):
                    batch=[samples[i%len(samples)] for i in range(size)];scorer.prompts.clear()
                    first=scorer.score(aid,chain,batch)[0].cpu().numpy();scorer.prompts.clear()
                    second=scorer.score(aid,chain,batch)[0].cpu().numpy()
                    error=max(max(abs(first[i]/(len(s)+1)-references[i%len(samples)]),abs(first[i]-second[i])/(len(s)+1)) for i,s in enumerate(batch))
                    if error>cfg['parity_tolerance']:raise ValueError(f'External parity {run}/{family}/{chain}/{size}: {error}')
                    checks.append(dict(target=family,chain=chain,batch=size,max_nll_error=float(error)))
            scorer.prompts.clear();values={}
            for start in range(0,len(seqs),64):
                batch=seqs[start:start+64];totals=scorer.score(aid,'heavy',batch)[0].cpu().tolist()
                values.update({s:float(v)/(len(s)+1) for s,v in zip(batch,totals)})
            rows.extend(dict(query_id=q,target=family,antigen_id=aid,affinity=float(aff),nll=values[queries[q]['heavy_sequence']],
                perplexity=math.exp(values[queries[q]['heavy_sequence']])) for q,aff in comparisons)
    frame=pd.DataFrame(rows);frame.to_csv(output/f'{run}_pairs.csv.gz',index=False,compression='gzip')
    metrics=[]
    for target,g in frame.groupby('target'):
        r=float(g.perplexity.corr(g.affinity));rho=float(g.perplexity.rank().corr(g.affinity.rank()))
        metrics.append(dict(run=run,target=target,pairs=len(g),pearson_r=r if math.isfinite(r) else None,
                            spearman_r=rho if math.isfinite(rho) else None))
    write_json(marker,dict(complete=True,inputs=expected,metrics=metrics,parity=checks,
                          scores_sha256=digest(output/f'{run}_pairs.csv.gz'),biological_uncertainty='Unavailable: no defensible independent parent-lineage identifiers supplied'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['train','amendment','external'])
    p.add_argument('--root',required=True);p.add_argument('--spec');p.add_argument('--run')
    a=p.parse_args();root=Path(a.root)
    if a.job=='train':train(root,a.spec)
    else:run_external(root,a.run,a.job=='amendment')
