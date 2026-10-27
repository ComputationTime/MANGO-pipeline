"""Position policy shared by fixed-prefix training, scoring and generation."""
import torch
from transformers import GPT2LMHeadModel


def position_ids(antigen_length, physical_positions):
    """Antigen starts at zero; selector/BOS/antibody start at zero/one/two."""
    return torch.where(physical_positions < antigen_length,
                       physical_positions, physical_positions - antigen_length)


def causal_context(antigen_length, inputs):
    positions = position_ids(antigen_length, torch.arange(inputs.shape[1], device=inputs.device))
    return dict(position_ids=positions.unsqueeze(0).expand(inputs.shape[0], -1),
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device))


class FixedPositionGPT2LMHeadModel(GPT2LMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                     attention_mask=None, inputs_embeds=None,
                                     cache_position=None, antigen_prefix_length=None, **kwargs):
        if antigen_prefix_length is None:
            raise ValueError('Fixed-position generation requires antigen_prefix_length')
        prepared = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        # KV-cache locations remain physical/monotonic. Only the learned position
        # embedding index resets after the antigen, including subsequent tokens.
        prepared['position_ids'] = position_ids(int(antigen_prefix_length), cache_position).unsqueeze(0)
        return prepared
