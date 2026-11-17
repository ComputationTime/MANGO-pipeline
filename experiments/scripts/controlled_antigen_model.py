"""IgLM-native shared decoder and exactly shaped antigen-memory controls."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from pathlib import Path
import torch
import torch.nn.functional as F
from iglm_baseline import IgLMScratch,MODEL_CONFIG
from antigen_diagnostics import iglm_token_nll
from pairing_prefix_score_cache import SharedPrefixChainScorer
from transformers.cache_utils import EncoderDecoderCache

NATIVE_SPEC=dict(initialization='random_from_scratch',objective='full_chain_autoregressive_v1',species_token='[HUMAN]')


def build(seed,branch=False,frozen=False,source=None,config=None):
    from transformers import GPT2Config,GPT2LMHeadModel
    cfg=dict(MODEL_CONFIG if config is None else config)
    torch.manual_seed(seed)
    model=IgLMScratch(NATIVE_SPEC,cfg)
    base=source if source is not None else model.lm.state_dict()
    if branch:
        # The base RNG stream and base tensors are independent of branch size.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1000000)
            lm=GPT2LMHeadModel(GPT2Config(**dict(cfg,add_cross_attention=True)))
            projection=torch.nn.Linear(1280,cfg['n_embd'])
        missing,unexpected=lm.load_state_dict(base,strict=False)
        if unexpected or any('.crossattention.' not in k and '.ln_cross_attn.' not in k for k in missing):
            raise ValueError('Shared decoder state differs')
        for block in lm.transformer.h:
            torch.nn.init.zeros_(block.crossattention.c_proj.weight);torch.nn.init.zeros_(block.crossattention.c_proj.bias)
        model.lm=lm;model.antigen_projection=projection
        if frozen:
            for name,p in model.lm.named_parameters():p.requires_grad_('.crossattention.' in name or '.ln_cross_attn.' in name)
    elif source is not None:model.lm.load_state_dict(source,strict=True)
    model.branch=branch;model.frozen=frozen;model.decoder_config=cfg
    return model


def parameters(model):
    return list(model.lm.parameters())+(list(model.antigen_projection.parameters()) if model.branch else [])


def mode(model,training):
    model.lm.train(training)
    if model.branch:model.antigen_projection.train(training)


def device(model,target):
    model.lm.to(target)
    if model.branch:model.antigen_projection.to(target)


def memory(tensor,tokens=32,constant=False):
    if tensor.ndim==2:tensor=tensor.unsqueeze(0)
    if tensor.ndim!=3 or tensor.shape[-1]!=1280:raise ValueError('Expected production ESM2 antigen features')
    result=F.adaptive_avg_pool1d(tensor.transpose(1,2),tokens).transpose(1,2)
    return torch.zeros_like(result) if constant else result


def save(path,model,spec,metadata):
    weights={'lm':{k:v.detach().cpu() for k,v in model.lm.state_dict().items()}}
    if model.branch:weights['antigen_projection']={k:v.detach().cpu() for k,v in model.antigen_projection.state_dict().items()}
    value=dict(format='controlled_antigen_v2',spec=spec,decoder_config=model.decoder_config,weights=weights,**metadata)
    temp=Path(path).with_suffix('.tmp.pt');temp.parent.mkdir(parents=True,exist_ok=True);torch.save(value,temp);temp.replace(path)


def load(path,target='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);spec=checkpoint['spec']
    model=build(spec['seed'],spec['branch'],spec['frozen'],config=checkpoint['decoder_config'])
    for key,value in checkpoint['weights'].items():getattr(model,key).load_state_dict(value,strict=True)
    device(model,target);mode(model,False)
    return model,checkpoint


class ConditionalScorer(SharedPrefixChainScorer):
    """Native control tokens/EOS, with an antigen-aware reusable KV prompt."""
    def __init__(self,model,antigens):
        super().__init__(model,'iglm',antigens)

    @torch.inference_mode()
    def prompt(self,antigen_id,chain):
        if not self.model.branch:return super().prompt(antigen_id,chain)
        key=(antigen_id,chain)
        if key not in self.prompts:
            ids=torch.tensor([[self.vocab[f'[{chain.upper()}]'],self.vocab[self.model.spec['species_token']]]],device=self.device)
            hidden=self.model.antigen_projection(self.antigens[antigen_id].to(self.device))
            output=self.lm(input_ids=ids,attention_mask=torch.ones_like(ids),encoder_hidden_states=hidden,use_cache=True)
            self.prompts[key]=dict(length=2,first=output.logits[:,-1:],cache=output.past_key_values.to_legacy_cache(),
                memory=hidden,cross=isinstance(output.past_key_values,EncoderDecoderCache))
        return self.prompts[key]


@torch.inference_mode()
def parity(model,antigen,sequences,chain,tolerance=3e-6):
    mode(model,False)
    scorer=ConditionalScorer(model,{'ag':antigen})
    reference=[float(iglm_token_nll(model,chain,s,h_ag=antigen if model.branch else None).double().mean()) for s in sequences]
    errors=[]
    for size in (1,8,64):
        batch=[sequences[i%len(sequences)] for i in range(size)]
        first=scorer.score('ag',chain,batch)[0].cpu().tolist()
        scorer.prompts.clear();second=scorer.score('ag',chain,batch)[0].cpu().tolist()
        for i,s in enumerate(batch):
            error=abs(first[i]/(len(s)+1)-reference[i%len(sequences)])
            repeat=abs(first[i]-second[i])/(len(s)+1)
            if max(error,repeat)>tolerance:raise ValueError(f'Likelihood parity failure: scalar={error}, repeat={repeat}')
            errors.append(max(error,repeat))
    return dict(chain=chain,max_nll_error=max(errors),batch_sizes=[1,8,64])
