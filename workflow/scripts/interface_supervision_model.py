"""Full-residue antigen memory and an auxiliary contact head; scratch decoders."""
import math
import torch
from torch import nn
from controlled_antigen_model import build as base_build
from iglm_baseline import MODEL_CONFIG,VOCAB


class GraphMemory(nn.Module):
    def __init__(self,projection,width=64,layers=2):
        super().__init__();self.projection=projection;hidden=projection.out_features
        self.values=nn.ModuleList(nn.Linear(hidden,width) for _ in range(layers))
        self.gates=nn.ModuleList(nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Linear(32,1)) for _ in range(layers))
        self.outputs=nn.ModuleList(nn.Linear(width,hidden) for _ in range(layers))
        for layer in self.outputs:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,embeddings,neighbors,edges,valid):
        x=self.projection(embeddings);batch=torch.arange(x.shape[0],device=x.device)[:,None,None]
        for value,gate,out in zip(self.values,self.gates,self.outputs):
            values=value(x)[batch,neighbors];weights=gate(edges).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1)
            x=x+out(torch.nn.functional.gelu((weights[...,None]*values).sum(2)))
        return x


class ContactHead(nn.Module):
    def __init__(self,hidden,width):
        super().__init__();self.q=nn.Linear(hidden,width);self.k=nn.Linear(hidden,width);self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,hidden,memory):return torch.matmul(self.q(hidden),self.k(memory).transpose(-1,-2))/math.sqrt(self.q.out_features)+self.bias


class InterfaceModel(nn.Module):
    def __init__(self,seed,branch=True,settings=None,base_config=None):
        super().__init__();settings=settings or {};self.base_config=dict(base_config or MODEL_CONFIG);self.branch=branch;self.contact_to_backbone=False
        base=base_build(seed,True,config=self.base_config)
        self.lm=base.lm;self.vocab={t:i for i,t in enumerate(VOCAB)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+4000000)
            self.memory=GraphMemory(base.antigen_projection,settings.get('message_width',64),settings.get('message_layers',2))
            self.contact_head=ContactHead(self.base_config['n_embd'],settings.get('contact_width',64))
        if not branch:
            target=sum(p.numel() for p in self.lm.parameters())+sum(p.numel() for p in self.memory.parameters())
            native=base_build(seed,False,config=self.base_config);original=sum(p.numel() for p in native.lm.parameters())
            slope=self.base_config['n_layer']*(2*self.base_config['n_embd']+1)
            expanded=dict(self.base_config,n_inner=self.base_config['n_inner']+round((target-original)/slope))
            wide=base_build(seed,False,config=expanded)
            with torch.no_grad():
                state=wide.lm.state_dict()
                for name,value in native.lm.state_dict().items():
                    dest=state[name]
                    if dest.shape==value.shape:dest.copy_(value)
                    elif '.mlp.c_fc.weight' in name:dest[:,:value.shape[1]].copy_(value)
                    elif '.mlp.c_fc.bias' in name:dest[:value.shape[0]].copy_(value)
                    elif '.mlp.c_proj.weight' in name:dest.zero_();dest[:value.shape[0]].copy_(value)
                    else:raise ValueError('Unexpected parameter matching shape')
            self.lm=wide.lm;self.memory=None;self.contact_head=None
            if abs(sum(p.numel() for p in self.lm.parameters())/target-1)>.001:raise ValueError('IgLM parameter matching failed')

    def forward(self,ids,mask,embeddings=None,antigen_mask=None,neighbors=None,edges=None,neighbor_valid=None):
        kwargs={};memory=None
        if self.branch:
            memory=self.memory(embeddings,neighbors,edges,neighbor_valid)
            kwargs=dict(encoder_hidden_states=memory,encoder_attention_mask=antigen_mask)
        output=self.lm(input_ids=ids,attention_mask=mask,output_hidden_states=self.branch,use_cache=False,**kwargs)
        contacts=None
        if self.branch:
            query=output.hidden_states[-1][:,2:-1]
            contacts=self.contact_head(query if self.contact_to_backbone else query.detach(),memory if self.contact_to_backbone else memory.detach())
        return output.logits,contacts


def balanced_contact_loss(logits,labels):
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels.float(),reduction='none')
    positive=labels.bool();negative=~positive
    if not positive.any():return logits.sum()*0
    return .5*losses[positive].mean()+.5*losses[negative].mean() if negative.any() else losses[positive].mean()


def save(path,model,metadata):
    from pathlib import Path
    path=Path(path);temp=path.with_suffix('.tmp.pt');torch.save(dict(state=model.state_dict(),base_config=model.base_config,**metadata),temp);temp.replace(path)
