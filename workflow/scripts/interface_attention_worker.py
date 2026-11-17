"""New, isolated worker for the interface-routing experiment."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from affinity_study_common import digest, fingerprint, read_csv, read_json, write_csv, write_json
from interface_attention_model import InterfaceAttentionModel
from interface_supervision_model import save
from interface_supervision_worker import Data as PreviousData, model_hash, verify


class Data(PreviousData):
    def __init__(self, root, target, arm):
        super().__init__(root, target, 'iglm_ce' if arm=='iglm' else 'geom_contact')

    def batch(self, model, examples, scrambled=False):
        output = super().batch(model, examples, scrambled=scrambled)
        output['attention'] = model.attention
        return output

    def supervised_loss(self, model, output, examples):
        if not model.branch:
            return output['means'] * 0.
        losses = []
        for i, (row, chain) in enumerate(examples):
            truth = torch.as_tensor(self.labels[row['id']][chain], device=self.target)
            losses.append(model.supervised_contact_loss(i, truth))
        return torch.stack(losses)


def fit(root, arm, seed):
    from device_common import get_device
    from short_affinity_worker import curve_plot
    root = Path(root)
    protocol, prepared = verify(root)
    cfg = protocol['settings']
    folder = root/'runs'/f'{arm}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = get_device('interface attention fit')
    model = InterfaceAttentionModel(seed, arm, cfg).to(device)
    initial, shared = model_hash(model), model_hash(model, True)
    data = Data(root, device, arm)
    byid = {r['id']:r for r in read_csv(root/'records.csv')}
    pair = read_json(root/'ranking_pair.json')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    torch.manual_seed(seed+1000000)
    model.train()
    points, exposure = [], []
    total_tokens, iteration = 0, 0
    started = time.monotonic()
    with (folder/'training_curve.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['iteration','phase','objective','nll','contact_loss'])
        writer.writeheader()
        for phase in ('pretrain','adapt'):
            for item in read_json(root/'schedules'/f'{phase}_{seed}.json')['items']:
                iteration += 1
                examples = [(byid[x['id']],x['chain']) for x in item]
                if phase=='adapt':
                    examples += [(byid[pair[k]['id']], 'heavy') for k in ('strong','weak')]
                if any(r['split']!='train' for r,c in examples):
                    raise ValueError('Holdout entered training')
                optimizer.zero_grad(set_to_none=True)
                out = data.batch(model, examples)
                contact = data.supervised_loss(model, out, examples)
                if phase=='adapt':
                    weight = cfg['pair_ce_weight']
                    ce = (1-weight)*out['means'][:-2].mean()+weight*out['means'][-2:].mean()
                    auxiliary = (1-weight)*contact[:-2].mean()+weight*contact[-2:].mean()
                else:
                    ce, auxiliary = out['means'].mean(), contact.mean()
                loss = ce + (cfg['contact_weight'] if model.branch else 0.)*auxiliary
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite objective')
                loss.backward()
                optimizer.step()
                total_tokens += int(out['counts'].sum())
                exposure.extend((r['id'],c) for r,c in examples)
                point = dict(iteration=iteration,phase=phase,objective=float(loss.detach()),
                             nll=float(ce.detach()),contact_loss=float(auxiliary.detach()))
                writer.writerow(point); handle.flush(); points.append(point)
                if iteration%100==0:
                    curve_plot(folder/'training_curve.png', [(x['iteration'],x['objective']) for x in points])
                    print(arm,seed,iteration,point,flush=True)
    save(folder/'final.pt', model, dict(arm=arm,seed=seed,settings=cfg,protocol_sha256=digest(root/'protocol.json')))
    curve_plot(folder/'training_curve.png', [(x['iteration'],x['objective']) for x in points])
    write_json(folder/'complete.json',dict(
        complete=True,arm=arm,seed=seed,updates=iteration,initial_sha256=initial,
        shared_decoder_sha256=shared,exposure_sha256=fingerprint(json.dumps(exposure)),
        target_tokens=total_tokens,parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_sha256=digest(folder/'final.pt'),wall_seconds=time.monotonic()-started,
        smoke_only=cfg['smoke'],next_residue_contact_supervision=True))


def load_model(path, device):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    model = InterfaceAttentionModel(payload['seed'],payload['arm'],payload['settings'],payload['base_config'])
    model.load_state_dict(payload['state'],strict=True)
    return model.to(device).eval()


def score(root, arm, seed):
    from device_common import get_device
    from chain_examples import example_chains
    root = Path(root)
    protocol, prepared = verify(root)
    cfg = protocol['settings']
    folder = root/'runs'/f'{arm}_s{seed}'
    marker = read_json(folder/'complete.json')
    if digest(folder/'final.pt') != marker['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    device = get_device('interface attention evaluation')
    model = load_model(folder/'final.pt', device)
    data = Data(root, device, arm)
    annotations = read_json(Path(cfg['diagnostic_root'])/'regions.json')['sequences']
    rows = read_csv(root/'records.csv')
    test = [(r,'heavy') for r in rows if r['split']=='test' and 'heavy' in example_chains(r)]
    native, external, contact_files, parity = [], [], {}, []
    contact_dir = folder/'contacts'; contact_dir.mkdir(exist_ok=True)
    variants = ['correct','off','shuffled'] if model.branch else ['correct']
    with torch.inference_mode():
        for variant in variants:
            model.intervention = variant
            model.capture = False
            single = data.batch(model,[test[0]])['means'][0]
            repeated = data.batch(model,[test[0]]*8)['means']
            padded = data.batch(model,[test[0],max(test,key=lambda x:len(x[0]['resolved_ag_seq']))])['means'][0]
            model.capture = True
            captured = data.batch(model,[test[0]])['means'][0]
            error = max(float((repeated-single).abs().max()),float(abs(padded-single)),float(abs(captured-single)))
            parity.append(error)
            if error > cfg['parity_tolerance']:
                raise ValueError('Padding/batching/passive attention capture changes likelihood')
            for start in range(0,len(test),4):
                examples = test[start:start+4]
                out = data.batch(model,examples)
                for i,(row,chain) in enumerate(examples):
                    seq = row['resolved_H_seq']
                    labels = annotations[fingerprint(seq)]['labels']
                    losses = out['tokens'][i].double().cpu().numpy()
                    cdr = np.array([x.startswith('cdr') for x in labels])
                    native.append(dict(id=row['id'],group=row['biological_group'],variant=variant,
                                       nll=float(losses.mean()),loss_sum=float(losses.sum()),ntokens=len(losses),
                                       cdr_nll=float(losses[cdr].mean()) if cdr.any() else ''))
                    if model.branch:
                        path = contact_dir/f'{fingerprint(row["id"])}_{variant}.npz'
                        length = out['lengths'][i]
                        np.savez_compressed(path,
                            logits=out['contacts'][i,:len(seq),:length].cpu().numpy(),
                            attention=out['attention'][i,:len(seq),:length].cpu().numpy(),
                            truth=data.labels[row['id']]['heavy'])
                        contact_files[str(path)] = digest(path)
            model.capture = False
            for ag in read_csv(root/'antigens.csv'):
                selected = [r for r in read_csv(root/'external_selection.csv') if r['antigen_id']==ag['id']]
                for start in range(0,len(selected),cfg['evaluation_batch_size']):
                    batch = selected[start:start+cfg['evaluation_batch_size']]
                    examples = [(dict(ag,resolved_H_seq=r['heavy_sequence']),'heavy') for r in batch]
                    out = data.batch(model,examples)
                    for i,r in enumerate(batch):
                        external.append(dict(query_id=r['query_id'],target=r['target'],variant=variant,
                                             affinity=float(r['affinity']),nll=float(out['means'][i])))
                print(arm,seed,variant,ag['family'],'affinity scoring complete',flush=True)
        # Real generation with only antigen and a chain selector. No target
        # sequence is used to seed, stop, or select these samples.
        model.intervention = 'correct'; model.capture = False
        generated = []
        allowed = [model.vocab[a] for a in 'ACDEFGHIKLMNPQRSTVWY']+[model.vocab['[SEP]']]
        inverse = {v:k for k,v in model.vocab.items()}
        for row,chain in test[:1]:
            prefix = ''
            stopped = False
            for step in range(160):
                # Full-prefix logits are obtained without ever reading a
                # native antibody sequence.
                ids = [model.vocab['[HEAVY]'],model.vocab['[HUMAN]']]+[model.vocab[a] for a in prefix]
                tensor = torch.tensor([ids],device=device)
                if model.branch:
                    key = fingerprint(row['resolved_ag_seq']); raw = data.graphs[key]
                    logits,_ = model(tensor,torch.ones_like(tensor),data.features[key][None],
                                     torch.ones((1,len(data.features[key])),device=device,dtype=torch.long),
                                     raw['geometry_neighbors'][None],raw['geometry_edges'][None],raw['geometry_valid'][None])
                else:
                    logits,_ = model(tensor,torch.ones_like(tensor))
                token = allowed[int(logits[0,-1,allowed].argmax())]
                if token==model.vocab['[SEP]']:
                    stopped=True; break
                prefix += inverse[token]
            generated.append(dict(antigen_id=row['id'],chain='heavy',sequence=prefix,ended_with_eos=stopped,
                                  max_residues=160,antibody_prompt_used=False,selection='first frozen held-out record; greedy'))
    write_csv(folder/'native.csv',native); write_csv(folder/'external.csv',external)
    write_json(folder/'generation.json',dict(samples=generated))
    write_json(folder/'evaluation.json',dict(
        complete=True,arm=arm,seed=seed,checkpoint_sha256=marker['checkpoint_sha256'],
        native_sha256=digest(folder/'native.csv'),external_sha256=digest(folder/'external.csv'),
        generation_sha256=digest(folder/'generation.json'),contacts=contact_files,parity=parity,
        smoke_only=cfg['smoke']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',choices=['fit','score'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    (fit if args.job=='fit' else score)(Path(args.root),args.arm,args.seed)
