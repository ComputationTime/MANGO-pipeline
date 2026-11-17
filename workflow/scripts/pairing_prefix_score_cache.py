"""Reuse identical teacher-forced antibody prefixes within a scoring batch."""
import os

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from pairing_score_cache import CachedChainScorer


class SharedPrefixChainScorer(CachedChainScorer):
    """An exact computation cache; the antibody prefix remains an output target."""

    def _continue(self, prompt, ids, lengths, output_offset):
        batch, length = ids.shape
        expanded = tuple(tuple(t.expand(batch, -1, -1, -1).contiguous() for t in layer)
                         for layer in prompt['cache'])
        cache = (EncoderDecoderCache if prompt['cross'] else DynamicCache).from_legacy_cache(expanded)
        valid = torch.arange(length, device=self.device)[None, :] < torch.tensor(lengths, device=self.device)[:, None]
        attention = torch.cat([torch.ones((batch, prompt['length']), device=self.device, dtype=torch.bool), valid], dim=1)
        physical = torch.arange(prompt['length'], prompt['length']+length, device=self.device)
        positions = (torch.arange(2+output_offset, 2+output_offset+length, device=self.device)
                     if self.family != 'iglm' and self.model.uses_fixed_positions else physical)
        context = dict(past_key_values=cache, use_cache=True, attention_mask=attention,
                       cache_position=physical, position_ids=positions[None, :])
        if prompt['memory'] is not None:
            context['encoder_hidden_states'] = prompt['memory'].expand(batch, -1, -1).contiguous()
        return self.lm(input_ids=ids, **context)

    @torch.inference_mode()
    def score(self, antigen_id, chain, sequences, cdr3_indices=None, encoded=None):
        prompt = self.prompt(antigen_id, chain)
        ids, labels = encoded if encoded is not None else self.tokens(sequences)
        # Keep at least one input residue so EOS follows an ordinary continuation.
        shared = min(len(os.path.commonprefix(sequences)), min(map(len, sequences))-1)
        prefix_losses = None
        if shared >= 8:
            prefix_ids = ids[:1, :shared]
            output = self._continue(prompt, prefix_ids, [shared], 0)
            logits = torch.cat([prompt['first'], output.logits[:, :-1]], dim=1)
            prefix_losses = F.cross_entropy(logits.transpose(1, 2), prefix_ids, reduction='none')
            prompt = dict(prompt, length=prompt['length']+shared, first=output.logits[:, -1:],
                          cache=output.past_key_values.to_legacy_cache())
        else:
            shared = 0
        output = self._continue(prompt, ids[:, shared:], [len(s)-shared for s in sequences], shared)
        logits = torch.cat([prompt['first'].expand(len(sequences), -1, -1), output.logits], dim=1)
        losses = F.cross_entropy(logits.transpose(1, 2), labels[:, shared:], reduction='none')
        if prefix_losses is not None:
            losses = torch.cat([prefix_losses.expand(len(sequences), -1), losses], dim=1)
        cdr = torch.zeros_like(losses, dtype=torch.bool)
        if cdr3_indices is not None:
            indices = [(i,j) for i,items in enumerate(cdr3_indices) for j in items]
            if indices:
                rows, columns = zip(*indices); cdr[list(rows), list(columns)] = True
        return (losses.sum(dim=1, dtype=torch.float64),
                (losses*cdr).sum(dim=1, dtype=torch.float64), losses)
