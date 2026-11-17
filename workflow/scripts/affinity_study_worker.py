"""One isolated training/probe/embedding job; controller consumes only completion files."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from affinity_study_common import (digest, fingerprint, freeze, read_csv, read_json,
                                   training_weights, write_csv, write_json)
from antigen_diagnostics import iglm_token_nll, mango_token_nll, wrong_antigen_map
from chain_examples import example_chains
from device_common import get_device
from train_antigen_diagnostic import build_model, parameter_list, set_training, to_device


def ranking_loss(strong_nll, weak_nll):
    return F.softplus(strong_nll - weak_nll)


def region_objective(losses, labels, cdr_weight=1.):
    if len(labels) != len(losses):
        raise ValueError('Region labels do not cover target residues plus EOS')
    weights = losses.new_tensor([cdr_weight if name.startswith('cdr') else 1. for name in labels])
    return (losses * weights).sum() / weights.sum()


def model_spec(kind):
    return dict(kind=kind, architecture='cross_attention', mode='correct', freeze_decoder=kind=='iglm_cross')


def save_model(path, model, spec, extra):
    weights = {name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()}
               for name in ('lm','antigen_projection','chain_embedding') if hasattr(model,name)}
    temporary = Path(path).with_suffix('.tmp.pt')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format='affinity_improvement_checkpoint_v1', spec=spec, weights=weights, **extra),temporary)
    temporary.replace(path)


def restore(path, device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(checkpoint['spec'])
    for name, weights in checkpoint['weights'].items():
        getattr(model,name).load_state_dict(weights,strict=True)
    to_device(model,checkpoint['spec']['kind'],device)
    return model,checkpoint


class Scoring:
    def __init__(self, root, model, kind, device, shuffled=False, rows=(), seed=0):
        from model_common import load_embedding
        self.root,self.model,self.kind,self.device=root,model,kind,device
        self.load_embedding=load_embedding; self.cache={}; self.forward_passes=0
        self.donors={}
        if shuffled:
            from asd_scaling_studies import maximal_record_shuffle
            self.donors=maximal_record_shuffle([r for r in rows if r['split']=='train'],seed)

    def antigen(self,row,variant='correct',training=False):
        if self.kind=='iglm':return None
        if training and self.donors:row=self.donors[row['id']]
        key=fingerprint(row['resolved_ag_seq'])
        if key not in self.cache:
            path=self.root/'embedding/embeddings/antigen/esm2'/row['split']/(row['id']+'.pt')
            self.cache[key]=self.load_embedding(str(path))
        return self.cache[key].to(self.device)

    def losses(self,row,chain,variant='correct',training=False,donor=None):
        sequence=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
        antigen=self.antigen(donor or row,variant,training)
        self.forward_passes+=1
        if self.kind in ('iglm','iglm_cross'):
            return iglm_token_nll(self.model,chain,sequence,h_ag=None if variant=='no_context' else antigen)
        return mango_token_nll(self.model,antigen,chain,sequence,mode=variant)


def evaluate(scorer,rows,regions,variants=False):
    set_training(scorer.model,scorer.kind,False)
    donors=[wrong_antigen_map(rows,seed) for seed in (1729,1730)] if variants and scorer.kind!='iglm' else []
    scores=[]
    with torch.no_grad():
        for row in rows:
            for chain in example_chains(row):
                seq=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
                labels=regions['sequences'][fingerprint(seq)]['labels']
                modes=['correct','wrong_0','wrong_1','no_context'] if donors else ['correct']
                for mode in modes:
                    donor=donors[int(mode[-1])][row['id']] if mode.startswith('wrong') else None
                    losses=scorer.losses(row,chain,mode,donor=donor).double().cpu().numpy()
                    if not np.isfinite(losses).all() or len(losses)!=len(seq)+1:
                        raise ValueError('Invalid target likelihood')
                    cdr3=np.asarray([x=='cdr3' for x in labels]); cdr=np.asarray([x.startswith('cdr') for x in labels])
                    scores.append(dict(id=row['id'],chain=chain,split=row['split'],group=row['ab_ag_cluster'],
                        antigen_id=fingerprint(row['resolved_ag_seq']),variant=mode,loss_sum=float(losses.sum()),
                        ntokens=len(losses),nll=float(losses.mean()),cdr3_sum=float(losses[cdr3].sum()),
                        cdr3_tokens=int(cdr3.sum()),cdr_sum=float(losses[cdr].sum()),cdr_tokens=int(cdr.sum())))
    return scores


def ordinary_nll(scores,chain=None):
    rows=[r for r in scores if r['variant']=='correct' and (chain is None or r['chain']==chain)]
    return sum(r['loss_sum'] for r in rows)/sum(r['ntokens'] for r in rows)


def training_plot(directory,history,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([h['epoch']+1 for h in history],[h['validation_nll'] for h in history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Validation NLL',title=title);fig.tight_layout()
    temporary=directory/'training_curve.tmp.png';fig.savefig(temporary,dpi=150);plt.close(fig)
    temporary.replace(directory/'training_curve.png')


def affinity_metric(scores,root,split):
    path=root/'verified_affinities.csv'
    if not path.exists():return None
    values={r['id']:math.exp(r['nll']) for r in scores if r['chain']=='heavy' and r['variant']=='correct'}
    groups=defaultdict(list)
    for row in read_csv(path):
        if row['split']==split and row['id'] in values:
            groups[fingerprint(row['resolved_ag_seq']),row['assay_id']].append((values[row['id']],float(row['log10_kd_nm'])))
    by_target=defaultdict(list)
    for (target,assay), pairs in groups.items():
        if len(pairs)>=3:
            x,y=np.asarray(pairs).T
            if np.ptp(x)>0 and np.ptp(y)>0:by_target[target].append(float(np.corrcoef(x,y)[0,1]))
    if not by_target:return None
    return float(np.mean([np.mean(v) for v in by_target.values()]))


def train(root,spec_path):
    settings=read_json(root/'protocol.json')['settings']; spec=read_json(spec_path)
    directory=root/'runs'/spec['id']; directory.mkdir(parents=True,exist_ok=True)
    freeze(directory/'spec.json',spec)
    if (directory/'complete.json').exists():
        done=read_json(directory/'complete.json')
        if digest(directory/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(spec['seed']);random.seed(spec['seed']);np.random.seed(spec['seed'])
    device=get_device('affinity improvement training')
    data_path=root/f'{spec["phase"]}_{"diversity" if "diversity" in spec["changes"] else "base"}.csv'
    audit=read_json(root/'cohort_audit.json')
    if digest(data_path)!=audit['files_sha256'][data_path.name]:raise ValueError('Frozen training cohort changed')
    rows=read_csv(data_path);regions=read_json(root/'regions.json')
    if regions['records_sha256']!=digest(root/'pool.csv'):raise ValueError('Region annotations cover another cohort')
    train_rows=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='val']
    kind=spec['kind'];m_spec=model_spec(kind)
    source=None
    if spec.get('source'):
        source_path=root/'runs'/spec['source']/'best.pt'
        if digest(source_path)!=read_json(source_path.parent/'complete.json')['checkpoint_sha256']:
            raise ValueError('Source IgLM checkpoint changed')
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        # Existing adapter builder accepts a standard IgLM source. The base
        # tensors are loaded strictly; only new cross-attention keys may differ.
        from iglm_baseline import MODEL_CONFIG
        packed=dict(lm=source['weights']['lm'],metadata=dict(model_config=MODEL_CONFIG,public_pretrained_weights_loaded=False))
        model=build_model(m_spec,packed)
    else:model=build_model(m_spec)
    to_device(model,kind,device)
    frozen={k:v.detach().cpu().clone() for k,v in model.lm.state_dict().items()
            if kind=='iglm_cross' and '.crossattention.' not in k and '.ln_cross_attn.' not in k}
    scorer=Scoring(root,model,kind,device,spec.get('shuffled',False),rows,spec['seed'])
    if source:
        initial=evaluate(scorer,validation,regions)
        reference=read_json(root/'runs'/spec['source']/'complete.json')['validation_nll']
        if abs(ordinary_nll(initial)-reference)>3e-6:raise ValueError('Warm start does not preserve IgLM NLL')
    optimizer=torch.optim.Adam([p for p in parameter_list(model,kind) if p.requires_grad],lr=settings['lr'])
    examples=[dict(row=r,chain=c,antigen_id=fingerprint(r['resolved_ag_seq'])) for r in train_rows for c in example_chains(r)]
    weights=training_weights(examples,'balance' in spec['changes'])
    for e,w in zip(examples,weights):e['weight']=w
    exposure=defaultdict(lambda:[0,0.])
    for e in examples:
        exposure[e['antigen_id'],e['chain']][0]+=1
        exposure[e['antigen_id'],e['chain']][1]+=e['weight']
    write_csv(directory/'exposure.csv',[dict(antigen_id=ag,chain=chain,examples=v[0],effective_weight=v[1])
                                      for (ag,chain),v in exposure.items()])
    pairs={}
    if 'ranking' in spec['changes']:
        all_rows={r['id']:r for r in rows}
        pairs={r['strong']:all_rows[r['weak']] for r in read_csv(root/'ranking_pairs.csv')
               if r['split']=='train' and r['strong'] in all_rows and r['weak'] in all_rows}
        if not pairs:raise ValueError('No verified ranking comparisons in selected cohort')
    best=math.inf;best_affinity=-math.inf;stale=0;iteration=0;tokens=0;started=time.monotonic();history=[]
    membership=[];torch.cuda.reset_peak_memory_stats()
    reference_nll=None
    if pairs:
        reference_nll=read_json(root/'runs'/spec['ce_reference']/'complete.json')['validation_heavy_nll']
    with (directory/'training_curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['iteration','epoch','phase','objective','nll']);writer.writeheader()
        for epoch in range(spec['epochs']):
            ordered=list(examples);random.Random(spec['seed']+epoch).shuffle(ordered)
            set_training(model,kind,True)
            if kind=='iglm_cross':
                # Disable frozen decoder dropout while retaining trainable branch
                # gradients; paired/shuffled controls use the identical policy.
                model.lm.eval()
            epoch_members=[]
            for e in ordered:
                optimizer.zero_grad(set_to_none=True)
                losses=scorer.losses(e['row'],e['chain'],training=True)
                seq=e['row']['resolved_H_seq' if e['chain']=='heavy' else 'resolved_L_seq']
                objective=region_objective(losses,regions['sequences'][fingerprint(seq)]['labels'],
                    settings['cdr_weight'] if 'cdr' in spec['changes'] else 1.)*e['weight']
                if e['chain']=='heavy' and e['row']['id'] in pairs:
                    weak=scorer.losses(pairs[e['row']['id']],'heavy',training=True)
                    objective=objective+settings['ranking_coefficient']*ranking_loss(losses.mean(),weak.mean())
                if not torch.isfinite(objective):raise ValueError('Nonfinite training objective')
                objective.backward();optimizer.step();iteration+=1;tokens+=len(losses)
                epoch_members.append((e['row']['id'],e['chain']))
                writer.writerow(dict(iteration=iteration,epoch=epoch,phase='train',objective=float(objective.detach()),
                                     nll=float(losses.detach().mean())));handle.flush()
            if sorted(epoch_members)!=sorted((e['row']['id'],e['chain']) for e in examples):
                raise ValueError('Training target membership changed')
            membership.append(fingerprint(json.dumps(sorted(epoch_members))))
            scores=evaluate(scorer,validation,regions);nll=ordinary_nll(scores);hnll=ordinary_nll(scores,'heavy')
            affinity=affinity_metric(scores,root,'val') if pairs else None
            history.append(dict(epoch=epoch,validation_nll=nll,validation_heavy_nll=hnll,affinity=affinity))
            writer.writerow(dict(iteration=iteration,epoch=epoch,phase='val',objective=nll,nll=nll));handle.flush()
            eligible=not pairs or (affinity is not None and hnll<=reference_nll+settings['nll_tolerance'])
            improved=eligible and (affinity>best_affinity if pairs else nll<best)
            extra=dict(epoch=epoch,validation_nll=nll,validation_heavy_nll=hnll,affinity=affinity,
                spec_id=spec['id'],records_sha256=digest(data_path),protocol_sha256=digest(root/'protocol.json'))
            save_model(directory/'latest.pt',model,m_spec,extra)
            if improved:
                best=nll;best_affinity=affinity if affinity is not None else -math.inf;stale=0
                save_model(directory/'best.pt',model,m_spec,extra)
            else:stale+=1
            write_json(directory/'epochs.json',history)
            training_plot(directory,history,spec['id'])
            if spec['phase']!='pilot' and stale>=settings['patience']:break
    if not (directory/'best.pt').exists():raise ValueError('No checkpoint satisfies the predefined selection constraint')
    for key,value in frozen.items():
        if not torch.equal(value,model.lm.state_dict()[key].detach().cpu()):raise ValueError('Frozen decoder changed')
    forward_passes=scorer.forward_passes
    unchanged=sum(r['resolved_ag_seq']==scorer.donors[r['id']]['resolved_ag_seq'] for r in train_rows) if scorer.donors else None
    del scorer,model
    model,checkpoint=restore(directory/'best.pt',device)
    scorer=Scoring(root,model,kind,device)
    evaluation_split='val' if spec['phase']=='pilot' else 'test'
    scored_rows=[r for r in rows if r['split']==evaluation_split]
    scores=evaluate(scorer,scored_rows,regions,variants=True)
    write_csv(directory/'scores.csv',scores)
    # Real production scalar scorer parity is checked against the model's own
    # public loss for each chain selector; adapter forward uses the same native
    # IgLM control-token/EOS contract.
    checks=[]
    with torch.no_grad():
        for chain in ('heavy','light'):
            row=next(r for r in validation if chain in example_chains(r))
            seq=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
            losses=scorer.losses(row,chain)
            if kind=='mango':reference=model.loss(scorer.antigen(row),chain,model.target_token_ids(seq).to(device))
            elif kind=='iglm':reference=model.loss(seq,chain)
            else:
                ids,labels=model.encode(seq,chain)
                reference=model.lm(input_ids=ids,labels=labels,use_cache=False,
                    encoder_hidden_states=model.antigen_projection(scorer.antigen(row))).loss
            delta=abs(float(reference)-float(losses.mean()))
            if delta>3e-6:raise ValueError(f'Scalar likelihood parity failed: {delta}')
            checks.append(dict(chain=chain,error=delta))
    # Live CSV is the complete trajectory; PNG is refreshed at epoch completion.
    training_plot(directory,history,spec['id'])
    write_json(directory/'complete.json',dict(complete=True,spec=spec,checkpoint_sha256=digest(directory/'best.pt'),
        scores_sha256=digest(directory/'scores.csv'),validation_nll=checkpoint['validation_nll'],
        validation_heavy_nll=checkpoint['validation_heavy_nll'],development_heavy_nll=ordinary_nll(scores,'heavy'),
        development_affinity=affinity_metric(scores,root,evaluation_split),evaluation_split=evaluation_split,
        epochs=len(history),optimizer_steps=iteration,target_tokens=tokens,training_membership_sha256=membership,
        forward_passes_training_and_validation=forward_passes,
        shuffled_correct_antigen_fraction=unchanged/len(train_rows) if unchanged is not None else None,
        trainable_parameters=sum(p.numel() for p in parameter_list(model,kind) if p.requires_grad),
        parameters=sum(p.numel() for p in parameter_list(model,kind)),frozen_decoder_verified=bool(frozen),
        scalar_parity=checks,wall_seconds=time.monotonic()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,seed=spec['seed']))


def embeddings(root):
    from asd_transfer_embeddings import embed_esm2
    settings=read_json(root/'protocol.json')['settings']
    embed_esm2(dict(settings,experiment_root=str(root)), 'embedding',str(root/'pool.csv'),str(root/'embeddings_complete.json'))


def latest_probes(root):
    """All original native test targets, two fixed wrong-antigen draws per model."""
    from model_common import MangoModel,load_embedding
    from iglm_baseline import IgLMScratch
    study=read_json(Path(read_json(root/'protocol.json')['settings']['binding_root'])/'study.json')
    rows=[r for r in read_csv(study['training_records']) if r['split']=='test']
    donors=[wrong_antigen_map(rows,s) for s in (1729,1730)]
    device=get_device('latest checkpoint antigen probes');torch.set_num_threads(2)
    outputs=[];failures=[]
    for name,spec in study['models'].items():
        try:
            path=Path(spec['directory'])/'checkpoints/best.pt'
            if digest(path)!=spec['checkpoint_sha256']:raise ValueError('Latest checkpoint changed since affinity analysis')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if spec['family']=='iglm':
                meta=checkpoint['metadata'];model=IgLMScratch(dict(initialization='random_from_scratch',
                    objective=meta['objective'],species_token=meta['species_token']),meta['model_config'])
                model.lm.load_state_dict(checkpoint['lm']);model.lm.to(device).eval()
            else:model=MangoModel.from_checkpoint(checkpoint).to(device).eval()
            scores=[]
            with torch.no_grad():
                for row in rows:
                    variants=['correct'] if spec['family']=='iglm' else ['correct','wrong_0','wrong_1']
                    for mode in variants:
                        donor=donors[int(mode[-1])][row['id']] if mode.startswith('wrong') else row
                        ag=None
                        if spec['family']!='iglm':
                            p=Path('artifacts/data/sabdab2_v0.1.0/embeddings/antigen')/name/'test'/(donor['id']+'.pt')
                            ag=load_embedding(str(p)).to(device)
                            if not model.uses_cross_attention and mode!='correct':
                                expected=load_embedding(str(p.parent/(row['id']+'.pt'))).shape[1]
                                ag=F.interpolate(ag.transpose(1,2),size=expected,mode='linear',align_corners=False).transpose(1,2)
                        for chain in ('heavy','light'):
                            seq=row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']
                            losses=iglm_token_nll(model,chain,seq) if spec['family']=='iglm' else mango_token_nll(model,ag,chain,seq)
                            scores.append(dict(model=name,id=row['id'],group=row['ab_ag_cluster'],variant=mode,
                                               chain=chain,loss_sum=float(losses.sum()),ntokens=len(losses)))
            write_csv(root/'probes'/f'{name}.csv',scores)
            outputs.append(dict(model=name,checkpoint_sha256=digest(path),examples=len(scores)))
            del model,checkpoint;torch.cuda.empty_cache()
        except Exception as error:
            failures.append(dict(model=name,error=repr(error)))
            torch.cuda.empty_cache()
    write_json(root/'probes/complete.json',dict(complete=not failures,models=outputs,failures=failures,donor_seeds=[1729,1730]))
    if failures:raise RuntimeError(f'Incomplete latest-checkpoint probes: {failures}')


def external(root):
    """Single final external evaluation, after all selection decisions are frozen."""
    from model_common import load_embedding
    settings=read_json(root/'protocol.json')['settings'];selection=read_json(root/'selection.json')
    source=Path(settings['binding_root']);pairs=read_csv(source/'heldout_antibodies.csv')
    queries={r['id']:r for r in pairs}
    with __import__('sqlite3').connect(f'file:{(source/"measurements.sqlite").resolve()}?mode=ro',uri=True) as db:
        measured=db.execute('SELECT query_id,antigen_id,reliable_sum/reliable_count FROM pairs WHERE reliable_count>0').fetchall()
    labels={r['id']:r['family'] for r in read_csv(source/'antigens.csv')}
    import pandas as pd
    results=[];device=get_device('final external affinity evaluation');torch.set_num_threads(2)
    for run in selection['external_runs']:
        model,checkpoint=restore(root/'runs'/run/'best.pt',device);kind=checkpoint['spec']['kind']
        buffers=[];cache={};antigens={}
        from pairing_prefix_score_cache import SharedPrefixChainScorer
        scorer=SharedPrefixChainScorer(model,'iglm' if kind=='iglm' else 'mango',{}) if kind!='iglm_cross' else None
        for aid in labels:
            antigen=load_embedding(str(source/'embeddings/esm2/test'/f'{aid}.pt')).to(device)
            comparisons=[(q,aff) for q,a,aff in measured if a==aid]
            sequences={queries[q]['heavy_sequence'] for q,aff in comparisons}
            if scorer:
                scorer.antigens={aid:antigen};scorer.prompts.clear()
                sequences=sorted(sequences)
                for start in range(0,len(sequences),64):
                    batch=sequences[start:start+64]
                    totals,_,_=scorer.score(aid,'heavy',batch)
                    for seq,value in zip(batch,totals.cpu().tolist()):cache[seq]=value/(len(seq)+1)
                # Per-antigen batched/scalar check before exporting scores.
                seq=sequences[0]
                scalar=iglm_token_nll(model,'heavy',seq) if kind=='iglm' else mango_token_nll(model,antigen,'heavy',seq)
                if abs(float(scalar.mean())-cache[seq])>3e-6:raise ValueError('External scalar parity failed')
            else:
                with torch.no_grad():
                    for seq in sequences:cache[seq]=float(iglm_token_nll(model,'heavy',seq,h_ag=antigen).double().mean())
            for q,aff in comparisons:
                value=cache[queries[q]['heavy_sequence']]
                buffers.append(dict(query_id=q,antigen_id=aid,target=labels[aid],affinity=aff,nll=value,perplexity=math.exp(value)))
        frame=pd.DataFrame(buffers)
        frame.to_csv(root/'external'/f'{run}_pairs.csv.gz',index=False,compression='gzip')
        for target,g in frame.groupby('target'):
            results.append(dict(run=run,target=target,pairs=len(g),pearson_r=float(g.perplexity.corr(g.affinity)),
                                spearman_r=float(g.perplexity.rank().corr(g.affinity.rank()))))
        del model,scorer;torch.cuda.empty_cache()
    write_csv(root/'external/correlations.csv',results)
    write_json(root/'external/complete.json',dict(complete=True,selection_sha256=digest(root/'selection.json'),
        results=results,limitations='Already-inspected external panel; exploratory. Parent/source lineage independence is not established.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',choices=['embeddings','train','probes','external'])
    p.add_argument('--root',required=True);p.add_argument('--spec')
    args=p.parse_args();root=Path(args.root)
    if args.job=='train':train(root,args.spec)
    elif args.job=='probes':latest_probes(root)
    elif args.job=='external':external(root)
    else:embeddings(root)
