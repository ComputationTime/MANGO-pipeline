"""Matched scratch decoders, parameter controls, and antigen-only resampling."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import math
from pathlib import Path
import torch
from controlled_antigen_model import build as base_build,memory,mode,device,ConditionalScorer
from iglm_baseline import MODEL_CONFIG


class MemoryProjection(torch.nn.Module):
    def __init__(self,linear,pool,seed,tokens=32):
        super().__init__();self.linear=linear;self.pool=pool;self.tokens=tokens
        if pool=='learned':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+3000000)
                width=linear.out_features
                self.queries=torch.nn.Parameter(torch.randn(tokens,width)*.02)
                self.attention=torch.nn.MultiheadAttention(width,8,batch_first=True,dropout=0.)

    def forward(self,tensor):
        if tensor.ndim==2:tensor=tensor.unsqueeze(0)
        if self.pool=='mean':return self.linear(memory(tensor,self.tokens))
        values=self.linear(tensor)
        return self.attention(self.queries[None].expand(tensor.shape[0],-1,-1),values,values,need_weights=False)[0]


def model_parameters(model):
    return list(model.lm.parameters())+(list(model.antigen_projection.parameters()) if model.branch else [])


def build(spec,config=None):
    cfg=dict(MODEL_CONFIG if config is None else config)
    native=base_build(spec['seed'],False,config=cfg);native.base_config=cfg
    if spec['branch']:
        model=base_build(spec['seed'],True,config=cfg)
        model.antigen_projection=MemoryProjection(model.antigen_projection,spec['pool'],spec['seed']);model.base_config=cfg
        return model
    if spec.get('native'):return native
    # Match total trainable parameters using only IgLM feed-forward width.
    reference=base_build(spec['seed'],True,config=cfg)
    reference.antigen_projection=MemoryProjection(reference.antigen_projection,spec['pool'],spec['seed'])
    target=sum(p.numel() for p in model_parameters(reference));original=sum(p.numel() for p in native.lm.parameters())
    slope=cfg['n_layer']*(2*cfg['n_embd']+1)
    expanded=dict(cfg,n_inner=cfg['n_inner']+round((target-original)/slope))
    model=base_build(spec['seed'],False,config=expanded)
    with torch.no_grad():
        actual=model.lm.state_dict()
        for name,value in native.lm.state_dict().items():
            destination=actual[name]
            if destination.shape==value.shape:destination.copy_(value)
            elif '.mlp.c_fc.weight' in name:destination[:,:value.shape[1]].copy_(value)
            elif '.mlp.c_fc.bias' in name:destination[:value.shape[0]].copy_(value)
            elif '.mlp.c_proj.weight' in name:
                destination.zero_();destination[:value.shape[0]].copy_(value)
            else:raise ValueError('Unexpected parameter-matching shape: '+name)
    model.decoder_config=expanded;model.base_config=cfg
    model.parameter_matching=dict(target=target,actual=sum(p.numel() for p in model.lm.parameters()))
    if abs(model.parameter_matching['actual']/target-1)>.001:raise ValueError('Parameter control differs by more than 0.1%')
    return model


def save(path,model,spec,metadata):
    weights={'lm':{k:v.detach().cpu() for k,v in model.lm.state_dict().items()}}
    if model.branch:weights['antigen_projection']={k:v.detach().cpu() for k,v in model.antigen_projection.state_dict().items()}
    temp=Path(path).with_suffix('.tmp.pt');temp.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format='affinity_short_v3',spec=spec,base_config=model.base_config,weights=weights,**metadata),temp);temp.replace(path)


def load(path,target='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False);model=build(payload['spec'],config=payload['base_config'])
    for key,state in payload['weights'].items():getattr(model,key).load_state_dict(state,strict=True)
    device(model,target);mode(model,False);return model,payload


def cdr_loss(losses,labels):
    mask=torch.tensor([x.startswith('cdr') for x in labels],device=losses.device)
    if len(mask)!=len(losses) or not mask.any() or mask.all():raise ValueError('Invalid region-balanced target')
    return .5*(losses[mask].mean()+losses[~mask].mean())


def preference(strong_losses,weak_losses):
    return torch.nn.functional.softplus(strong_losses.mean()-weak_losses.mean())
