"""Isolated bounded ESM2 windows, with full residue coverage and chain breaks."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import hashlib
import json
from pathlib import Path
import torch
from affinity_study_common import read_json,read_csv,write_json,digest,fingerprint


def windows(length,size,overlap):
    if not 0<=overlap<size:raise ValueError('Invalid window overlap')
    start=0;result=[]
    while start<length:
        end=min(length,start+size);result.append((start,end))
        if end==length:break
        start+=size-overlap
    return result


def main(root):
    import embed_common as ec
    from embed_antigen_esm2 import load_runtime
    root=Path(root);cfg=read_json(root/'protocol.json')['settings'];rows=read_csv(root/'records.csv')
    if not read_json(root/'cohort_audit.json')['feasible']:raise ValueError('Infeasible cohort')
    output=root/'embeddings';output.mkdir(exist_ok=True);runtime=None;files={}
    for row in rows:
        key=fingerprint(row['resolved_ag_seq']);path=output/f'{key}.pt'
        if str(path) in files:continue
        sequences=list(ec.antigen_sequences(row,'resolved').items());length=sum(len(s) for c,s in sequences)+len(sequences)-1
        cache_key=hashlib.sha256(json.dumps(sequences).encode()).hexdigest();cached=Path(cfg['esm2_cache_dir'])/f'{cache_key}.pt'
        maps=[dict(chain=c,start=a,end=b) for c,s in sequences for a,b in windows(len(s),cfg['window_size'],cfg['window_overlap'])]
        payload=None
        if cached.exists() and all(len(s)<=cfg['window_size'] for c,s in sequences):
            old=torch.load(cached,map_location='cpu',weights_only=False)
            if old.get('model_name')=='esm2_t33_650M_UR50D' and old.get('shape')==[length,1280]:payload=dict(old,source_cache_sha256=digest(cached))
        if payload is None:
            if runtime is None:runtime=load_runtime('t33_650M')
            model,alphabet,layer,target=runtime;converter=alphabet.get_batch_converter();parts=[]
            for chain,seq in sequences:
                total=torch.zeros(len(seq),1280);counts=torch.zeros(len(seq),1);cls=[]
                for begin,end in windows(len(seq),cfg['window_size'],cfg['window_overlap']):
                    _,_,tokens=converter([(chain,seq[begin:end])])
                    with torch.inference_mode():rep=model(tokens.to(target),repr_layers=[layer],return_contacts=False)['representations'][layer][0].cpu()
                    total[begin:end]+=rep[1:1+end-begin];counts[begin:end]+=1;cls.append(rep[:1])
                if (counts==0).any():raise ValueError('Uncovered antigen residues')
                parts.extend([torch.stack(cls).mean(0),total/counts])
            tensor=torch.cat(parts)[1:]
            payload=dict(embedding=tensor,shape=list(tensor.shape),model_name='esm2_t33_650M_UR50D')
        payload.update(windows=maps,window_size=cfg['window_size'],overlap=cfg['window_overlap'],antigen_sha256=key)
        if tuple(payload['embedding'].shape)!=(length,1280) or not torch.isfinite(payload['embedding']).all():raise ValueError('Invalid windowed antigen tensor')
        temporary=path.with_suffix('.tmp.pt');torch.save(payload,temporary);temporary.replace(path);files[str(path)]=digest(path)
    write_json(root/'embeddings_complete.json',dict(complete=True,records_sha256=digest(root/'records.csv'),files_sha256=files,
        window_size=cfg['window_size'],overlap=cfg['window_overlap'],chain_breaks=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(**vars(p.parse_args()))
