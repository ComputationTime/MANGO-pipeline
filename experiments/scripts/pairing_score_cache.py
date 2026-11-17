"""Batched exact target NLL with reusable antigen/selector/BOS KV caches."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from fixed_position_prefix import causal_context


class CachedChainScorer:
    def __init__(self, model, family, antigens):
        self.model = model
        self.family = family
        self.lm = model.lm
        self.device = next(self.lm.parameters()).device
        self.antigens = antigens
        self.prompts = {}
        self.vocab = model.vocab
        self.eos = self.vocab['[SEP]'] if family == 'iglm' else self.vocab['>']
        self.pad = self.vocab['[PAD]'] if family == 'iglm' else self.vocab['-']

    @torch.inference_mode()
    def prompt(self, antigen_id, chain):
        key = (None if self.family == 'iglm' else antigen_id, chain)
        if key in self.prompts:
            return self.prompts[key]
        if self.family == 'iglm':
            ids = torch.tensor([[self.vocab[f'[{chain.upper()}]'], self.vocab[self.model.spec['species_token']]]],
                               device=self.device)
            length = 2
            output = self.lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            memory = None
        else:
            antigen = self.antigens[antigen_id].to(self.device)
            prefix = self.model._prefix(antigen, chain)
            length = prefix.shape[1]
            context = self.model._antigen_context(antigen)
            memory = context.get('encoder_hidden_states')
            context['attention_mask'] = torch.ones(prefix.shape[:2],dtype=torch.long,device=self.device)
            if self.model.uses_fixed_positions:
                context.update(causal_context(antigen.shape[1],prefix))
            output = self.lm(inputs_embeds=prefix, use_cache=True, **context)
        result = dict(length=length, first=output.logits[:,-1:],
                      cache=output.past_key_values.to_legacy_cache(), memory=memory,
                      cross=isinstance(output.past_key_values,EncoderDecoderCache))
        self.prompts[key] = result
        return result

    def tokens(self, sequences):
        size = max(map(len,sequences))
        values = [[self.vocab[a] for a in sequence] for sequence in sequences]
        ids = torch.tensor([v+[self.pad]*(size-len(v)) for v in values],dtype=torch.long,device=self.device)
        labels = torch.tensor([v+[self.eos]+[-100]*(size-len(v)) for v in values],dtype=torch.long,device=self.device)
        return ids,labels

    @torch.inference_mode()
    def score(self, antigen_id, chain, sequences, cdr3_indices=None, encoded=None):
        prompt = self.prompt(antigen_id,chain)
        ids, labels = encoded if encoded is not None else self.tokens(sequences)
        batch,length = ids.shape
        expanded = tuple(tuple(t.expand(batch,-1,-1,-1).contiguous() for t in layer)
                         for layer in prompt['cache'])
        cache = (EncoderDecoderCache if prompt['cross'] else DynamicCache).from_legacy_cache(expanded)
        valid = torch.arange(length,device=self.device).unsqueeze(0) < torch.tensor(
            [len(s) for s in sequences],device=self.device).unsqueeze(1)
        mask = torch.cat([torch.ones((batch,prompt['length']),dtype=torch.bool,device=self.device),valid],dim=1)
        physical = torch.arange(prompt['length'],prompt['length']+length,device=self.device)
        positions = physical
        if self.family != 'iglm' and self.model.uses_fixed_positions:
            positions = torch.arange(2,2+length,device=self.device)
        context = dict(past_key_values=cache,use_cache=True,attention_mask=mask,
                       cache_position=physical,position_ids=positions.unsqueeze(0))
        if prompt['memory'] is not None:
            context['encoder_hidden_states'] = prompt['memory'].expand(batch,-1,-1).contiguous()
        output = self.lm(input_ids=ids, **context)
        logits = torch.cat([prompt['first'].expand(batch,-1,-1),output.logits],dim=1)
        losses = F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
        totals = losses.sum(dim=1,dtype=torch.float64)
        cdr = torch.zeros_like(losses,dtype=torch.bool)
        if cdr3_indices is not None:
            indices = [(i,j) for i,items in enumerate(cdr3_indices) for j in items]
            if indices:
                rows,cols = zip(*indices)
                cdr[list(rows),list(cols)] = True
        cdr_totals = (losses*cdr).sum(dim=1,dtype=torch.float64)
        return totals,cdr_totals,losses
