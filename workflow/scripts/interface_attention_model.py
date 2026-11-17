"""Separate experimental decoder: predicted interface tags and attention bias.

No native interface information is accepted by forward(). Contact predictions
at position t describe the NEXT residue, using only the prefix through t.
The existing MANGO/interface-supervision implementations are imported read-only.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from controlled_antigen_model import build as base_build
from iglm_baseline import MODEL_CONFIG, VOCAB
from interface_supervision_model import GraphMemory, balanced_contact_loss


class InterfacePredictor(nn.Module):
    def __init__(self, hidden, width, tag_std):
        super().__init__()
        self.q = nn.Linear(hidden, width)
        self.k = nn.Linear(hidden, width)
        self.paratope = nn.Linear(hidden, 1)
        self.epitope = nn.Linear(hidden, 1)
        self.tag = nn.Parameter(torch.randn(hidden) * tag_std)

    def forward(self, query, memory):
        p = self.paratope(query).squeeze(-1)
        e = self.epitope(memory).squeeze(-1)
        pair = self.q(query) @ self.k(memory).transpose(-1, -2)
        pair = pair / math.sqrt(self.q.out_features) + p[..., None] + e[:, None, :]
        return pair, p, e


def interface_cross_attention(self, hidden_states, past_key_values=None,
                              encoder_hidden_states=None,
                              encoder_attention_mask=None, head_mask=None, **kwargs):
    """Q/K-only tags keep values fixed; all arms use the same SDPA implementation."""
    if past_key_values is not None:
        raise ValueError('This experiment uses full-prefix decoding, without a KV cache')
    if encoder_hidden_states is None:
        raise ValueError('Antigen memory is required')
    pair, p, e = self.interface_predictor(hidden_states, encoder_hidden_states)
    self.interface_outputs = (pair, p, e)
    # Both actual tags and contact logits remain correctly aligned for their
    # auxiliary loss. Only the routed signal is permuted by the intervention.
    permutation = self.interface_permutation
    routed_e = e if permutation is None else e.gather(1, permutation)
    routed_pair = pair if permutation is None else pair.gather(
        2, permutation[:, None, :].expand_as(pair))
    query_input = hidden_states
    q = self.q_attn(query_input)
    k, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
    if self.interface_mode == 'shared_tag' and not self.interface_disabled:
        # Add exactly the same learned hidden-space vector on the two sides,
        # weighted by predicted (never native) soft region memberships.
        q = self.q_attn(query_input + p.sigmoid()[..., None] * self.interface_predictor.tag)
        tagged_memory = encoder_hidden_states + routed_e.sigmoid()[..., None] * self.interface_predictor.tag
        k = self.c_attn(tagged_memory)[..., :self.split_size]
    q = q.view(*q.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(*k.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(*value.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
    mask = encoder_attention_mask
    if self.interface_mode == 'contact_bias' and not self.interface_disabled:
        bias = self.interface_strength * routed_pair.sigmoid()[:, None]
        mask = bias if mask is None else mask + bias
    scale = self.head_dim ** -0.5 if self.scale_attn_weights else 1.
    if self.scale_attn_by_inverse_layer_idx:
        scale /= self.layer_idx + 1
    attention = F.scaled_dot_product_attention(
        q, k, value, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0., scale=scale)
    if head_mask is not None:
        attention = attention * head_mask
    # Passive reconstruction for diagnostics; never switch attention backends.
    self.interface_attention = None
    if self.interface_capture:
        logits = q @ k.transpose(-1, -2) * scale
        if mask is not None:
            logits = logits + mask
        self.interface_attention = logits.softmax(-1).mean(1)
    attention = attention.transpose(1, 2).reshape(hidden_states.shape)
    return self.resid_dropout(self.c_proj(attention)), None


class InterfaceAttentionModel(nn.Module):
    def __init__(self, seed, arm, settings=None, base_config=None):
        super().__init__()
        self.arm = arm
        self.branch = arm != 'iglm'
        self.settings = dict(settings or {})
        self.base_config = dict(base_config or MODEL_CONFIG)
        self.vocab = {t: i for i, t in enumerate(VOCAB)}
        base = base_build(seed, True, config=self.base_config)
        self.lm = base.lm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 5000000)
            self.memory = GraphMemory(base.antigen_projection,
                                      self.settings.get('message_width', 64),
                                      self.settings.get('message_layers', 2))
            for block in self.lm.transformer.h:
                cross = block.crossattention
                cross.interface_predictor = InterfacePredictor(
                    self.base_config['n_embd'], self.settings.get('contact_width', 64),
                    self.settings.get('tag_initial_std', .1))
                cross.forward = MethodType(interface_cross_attention, cross)
        self.capture = False
        self.intervention = 'correct'
        if not self.branch:
            # Match all parameters available to the routing arms, including the
            # predictors that participate in their generation path.
            target = sum(p.numel() for p in self.parameters())
            native = base_build(seed, False, config=self.base_config)
            original = sum(p.numel() for p in native.lm.parameters())
            slope = self.base_config['n_layer'] * (2 * self.base_config['n_embd'] + 1)
            expanded = dict(self.base_config, n_inner=self.base_config['n_inner'] + round((target-original)/slope))
            wide = base_build(seed, False, config=expanded)
            with torch.no_grad():
                state = wide.lm.state_dict()
                for name, value in native.lm.state_dict().items():
                    dest = state[name]
                    if dest.shape == value.shape:
                        dest.copy_(value)
                    elif '.mlp.c_fc.weight' in name:
                        dest[:, :value.shape[1]].copy_(value)
                    elif '.mlp.c_fc.bias' in name:
                        dest[:value.shape[0]].copy_(value)
                    elif '.mlp.c_proj.weight' in name:
                        dest.zero_(); dest[:value.shape[0]].copy_(value)
                    else:
                        raise ValueError('Unexpected matching shape: ' + name)
            self.lm = wide.lm
            self.memory = None
            if abs(sum(p.numel() for p in self.parameters()) / target - 1) > .001:
                raise ValueError('IgLM parameter matching failed')
        self.auxiliary = []
        self.attention = None

    def forward(self, ids, mask, embeddings=None, antigen_mask=None,
                neighbors=None, edges=None, neighbor_valid=None):
        kwargs = {}
        self.auxiliary = []
        self.attention = None
        if self.branch:
            memory = self.memory(embeddings, neighbors, edges, neighbor_valid)
            # Fixed pseudorandom permutation per length: deterministic across
            # batch sizes, padding, arms and seeds, and never label-dependent.
            permutation = None
            if self.intervention == 'shuffled':
                permutation = torch.arange(memory.shape[1], device=ids.device).repeat(len(ids), 1)
                for i, length in enumerate(antigen_mask.sum(1).tolist()):
                    generator = torch.Generator(device='cpu').manual_seed(271828 + length)
                    permutation[i, :length] = torch.randperm(length, generator=generator).to(ids.device)
            for block in self.lm.transformer.h:
                cross = block.crossattention
                cross.interface_mode = self.arm
                cross.interface_strength = self.settings.get('bias_strength', 2.)
                cross.interface_disabled = self.intervention == 'off'
                cross.interface_capture = self.capture
                cross.interface_permutation = permutation
            kwargs = dict(encoder_hidden_states=memory, encoder_attention_mask=antigen_mask)
        output = self.lm(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs)
        contacts = None
        if self.branch:
            self.auxiliary = [b.crossattention.interface_outputs for b in self.lm.transformer.h]
            # Human token predicts first residue; residue t predicts residue t+1.
            contacts = torch.stack([x[0][:, 1:-2] for x in self.auxiliary]).mean(0)
            if self.capture:
                self.attention = torch.stack([
                    b.crossattention.interface_attention[:, 1:-2] for b in self.lm.transformer.h]).mean(0)
        return output.logits, contacts

    def supervised_contact_loss(self, index, truth):
        length, antigen_length = truth.shape
        parts = []
        for pair, p, e in self.auxiliary:
            parts.append(.5 * balanced_contact_loss(pair[index, 1:1+length, :antigen_length], truth)
                         + .25 * balanced_contact_loss(p[index, 1:1+length], truth.any(-1))
                         + .25 * balanced_contact_loss(e[index, :antigen_length], truth.any(0)))
        return torch.stack(parts).mean()
