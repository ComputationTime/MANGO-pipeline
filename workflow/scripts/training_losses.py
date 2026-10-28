"""Target-token training objectives; model evaluation always uses ordinary NLL.

Focal modulation follows https://arxiv.org/abs/1708.02002, applied to the
correct next-token probability in the multiclass antibody vocabulary. No
binding-class labels or per-amino-acid class balancing weights are used.
"""
import math

import torch
import torch.nn.functional as F


def loss_configuration(value=None):
    value = dict(value or {})
    name = value.get('name', 'cross_entropy')
    if name == 'cross_entropy':
        if set(value) - {'name'}:
            raise ValueError('Cross-entropy does not accept focal parameters')
        return {'name': name}
    if name != 'focal' or set(value) - {'name', 'gamma'}:
        raise ValueError(f'Unknown training loss configuration: {value}')
    gamma = float(value.get('gamma', 2.0))
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('Focal gamma must be finite and nonnegative')
    return {'name': name, 'gamma': gamma}


def causal_token_losses(logits, labels, configuration=None):
    """Return (training objective, ordinary NLL) over scored next-token labels.

Ignore antigen/selector/BOS labels before reducing, retaining the GPT-2 causal
shift. Alpha is uniformly one; residues and EOS all receive the same policy.
The focusing weight participates in autograd, including its dependence on p_t.
"""
    configuration = loss_configuration(configuration)
    shifted_labels = labels[..., 1:]
    scored = shifted_labels != -100
    if not scored.any():
        raise ValueError('Training example has no scored target tokens')
    selected_logits = logits[..., :-1, :][scored].float()
    targets = shifted_labels[scored]
    token_nll = F.cross_entropy(selected_logits, targets, reduction='none')
    nll = token_nll.mean()
    if configuration['name'] == 'cross_entropy' or configuration['gamma'] == 0:
        return nll, nll
    # 1-exp(-CE) is accurate even for very easy tokens with small CE.
    modulation = (-torch.expm1(-token_nll)).pow(configuration['gamma'])
    return (modulation * token_nll).mean(), nll
