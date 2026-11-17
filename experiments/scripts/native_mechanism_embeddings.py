"""Full-coverage antigen windows shared by training and exploratory scoring."""
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
from affinity_study_common import digest,fingerprint,read_csv,read_json,write_json
from short_affinity_embeddings import windows


def main(root):
    import embed_common as ec
    from embed_antigen_esm2 import load_runtime
    root=Path(root);cfg=read_json(root/'training_protocol.json')['settings']
    audit=read_json(root/'cohort_audit.json')
    if not audit['feasible'] or digest(root/'records.csv')!=audit['files_sha256']['records.csv']:
        raise ValueError('Cohort is not verified')
    rows=read_csv(root/'records.csv')+read_csv(Path(cfg['binding_root'])/'antigens.csv')
    output=root/'embeddings';output.mkdir(exist_ok=True);files={};runtime=None
    for row in rows:
        key=fingerprint(row['resolved_ag_seq']);path=output/f'{key}.pt'
        if str(path) in files:continue
        sequences=list(ec.antigen_sequences(row,'resolved').items())
        length=sum(len(s) for c,s in sequences)+len(sequences)-1
        mapping=[dict(chain=c,start=a,end=b) for c,s in sequences for a,b in windows(len(s),cfg['window_size'],cfg['window_overlap'])]
        expected=dict(antigen_sha256=key,window_size=cfg['window_size'],overlap=cfg['window_overlap'],windows=mapping)
        payload=None
        if path.exists():
            previous=torch.load(path,map_location='cpu',weights_only=False)
            if any(previous.get(k)!=v for k,v in expected.items()):raise ValueError('Feature provenance changed')
            payload=previous
        cached=Path(cfg['esm2_cache_dir'])/(hashlib.sha256(json.dumps(sequences).encode()).hexdigest()+'.pt')
        if payload is None and cached.exists() and all(len(s)<=cfg['window_size'] for c,s in sequences):
            previous=torch.load(cached,map_location='cpu',weights_only=False)
            if previous.get('model_name')=='esm2_t33_650M_UR50D' and previous.get('shape')==[length,1280]:
                payload=dict(previous,source_cache_sha256=digest(cached),**expected)
        if payload is None:
            if runtime is None:runtime=load_runtime('t33_650M')
            model,alphabet,layer,target=runtime;convert=alphabet.get_batch_converter();parts=[]
            for chain,sequence in sequences:
                total=torch.zeros(len(sequence),1280);counts=torch.zeros(len(sequence),1);cls=[]
                for begin,end in windows(len(sequence),cfg['window_size'],cfg['window_overlap']):
                    _,_,tokens=convert([(chain,sequence[begin:end])])
                    with torch.inference_mode():
                        rep=model(tokens.to(target),repr_layers=[layer],return_contacts=False)['representations'][layer][0].cpu()
                    total[begin:end]+=rep[1:1+end-begin];counts[begin:end]+=1;cls.append(rep[:1])
                if (counts==0).any():raise ValueError('Missing antigen residues')
                parts.extend([torch.stack(cls).mean(0),total/counts])
            tensor=torch.cat(parts)[1:]
            payload=dict(embedding=tensor,shape=list(tensor.shape),model_name='esm2_t33_650M_UR50D',**expected)
        if tuple(payload['embedding'].shape)!=(length,1280) or not torch.isfinite(payload['embedding']).all():
            raise ValueError('Invalid antigen embedding')
        temp=path.with_suffix('.tmp.pt');torch.save(payload,temp);temp.replace(path);files[str(path)]=digest(path)
    write_json(root/'embeddings_complete.json',dict(complete=True,records_sha256=digest(root/'records.csv'),
        antigens_sha256=digest(Path(cfg['binding_root'])/'antigens.csv'),files_sha256=files,
        chain_breaks=True,window_size=cfg['window_size'],overlap=cfg['window_overlap']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(**vars(p.parse_args()))
