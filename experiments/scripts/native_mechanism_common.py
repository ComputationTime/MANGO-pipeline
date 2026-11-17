"""Specified mechanisms and descriptive uncertainty; no confirmation p-values."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import defaultdict
import numpy as np


def arms():
    result = {'source': dict(branch=False, representation='none', memory='none', frozen=False, objective='ce', source=None),
              'continuation': dict(branch=False, representation='none', memory='none', frozen=False, objective='ce', source='source'),
              'wide_source': dict(branch=False, representation='none', memory='none', frozen=False, objective='ce', source=None, wide=True),
              'wide_continuation': dict(branch=False, representation='none', memory='none', frozen=False, objective='ce', source='wide_source')}
    for update in ('frozen','joint'):
        for context in ('correct','shuffled','constant'):
            result[f'{update}_{context}'] = dict(branch=True, representation='esm2', memory=context,
                frozen=update=='frozen', objective='ce', source='source')
    for context in ('correct','shuffled'):
        result[f'pairing_{context}'] = dict(branch=True, representation='esm2', memory=context,
            frozen=False, objective='pairing', source='source')
        result[f'onehot_{context}'] = dict(branch=True, representation='one_hot', memory=context,
            frozen=False, objective='ce', source='source')
    return result


CONTRASTS = {
    'pairing_with_frozen_decoder': {'frozen_shuffled':1., 'frozen_correct':-1.},
    'pairing_with_joint_decoder': {'joint_shuffled':1., 'joint_correct':-1.},
    'freezing_by_pairing_interaction': {'frozen_shuffled':1.,'frozen_correct':-1.,'joint_shuffled':-1.,'joint_correct':1.},
    'content_with_frozen_decoder': {'frozen_constant':1.,'frozen_correct':-1.},
    'content_with_joint_decoder': {'joint_constant':1.,'joint_correct':-1.},
    'pairing_objective_interaction': {'pairing_shuffled':1.,'pairing_correct':-1.,'joint_shuffled':-1.,'joint_correct':1.},
    'pairing_objective_correct_input': {'joint_correct':1.,'pairing_correct':-1.},
    'onehot_pairing': {'onehot_shuffled':1.,'onehot_correct':-1.},
    'ESM2_vs_parameter_matched_IgLM': {'wide_continuation':1.,'joint_correct':-1.},
    'frozen_vs_native_continuation': {'continuation':1.,'frozen_correct':-1.},
}


def affinity_correlations(nll,affinity):
    """Average-rank Spearman and Pearson; no dependency on the analysis env."""
    import pandas as pd
    nll=np.asarray(nll,dtype=float);affinity=np.asarray(affinity,dtype=float)
    if nll.ndim!=1 or nll.shape!=affinity.shape or len(nll)<3:
        raise ValueError('Expected at least three paired affinity scores')
    if not np.isfinite(nll).all() or not np.isfinite(affinity).all():
        raise ValueError('Nonfinite affinity inputs')
    if np.ptp(nll)==0 or np.ptp(affinity)==0:
        return dict(spearman=None,pearson_perplexity=None)
    return dict(spearman=float(pd.Series(-nll).rank().corr(pd.Series(-affinity).rank())),
                pearson_perplexity=float(np.corrcoef(np.exp(nll),affinity)[0,1]))


def summarize(losses, counts, repeats=10000, seed=1729):
    """Biological bootstrap conditional on realized seeds, with separate seed SD.

    Arrays are paired [seed, biological component]. Do not resample the five
    seeds into a purported joint 95% interval: that procedure failed calibration.
    """
    losses=np.asarray(losses,dtype=float); counts=np.asarray(counts,dtype=float)
    if losses.ndim!=2 or losses.shape!=counts.shape or min(losses.shape)<1:
        raise ValueError('Expected nonempty matching seed-by-component arrays')
    if not np.isfinite(losses).all() or not np.isfinite(counts).all() or (counts<=0).any():
        raise ValueError('Invalid losses or token counts')
    points=losses.sum(1)/counts.sum(1)
    value=dict(mean=float(points.mean()), seed_points=points.tolist(), seeds=len(points),
               components=losses.shape[1], seed_sd=float(points.std(ddof=1)) if len(points)>1 else None,
               low=None, high=None, interval_scope='biological components conditional on realized seeds',
               inference='exploratory; no confirmation test or p-value')
    if losses.shape[1]<20:
        value['interval_unavailable_reason']='fewer_than_20_independent_components'
        return value
    rng=np.random.default_rng(seed)
    draws=np.empty(repeats)
    for start in range(0,repeats,200):
        ix=rng.integers(losses.shape[1],size=(min(200,repeats-start),losses.shape[1]))
        draws[start:start+len(ix)]=(losses[:,ix].sum(2)/counts[:,ix].sum(2)).mean(0)
    value['low'],value['high']=map(float,np.quantile(draws,[.025,.975]))
    return value


def paired_arrays(tables, contrast):
    """Strictly align every seed/arm/sequence mask before aggregation."""
    assembled=[]; denominators=[]; expected=None
    for seed in sorted(tables):
        maps=[]
        for arm,coefficient in contrast.items():
            if arm not in tables[seed]:raise ValueError('Missing paired arm: '+arm)
            values=[r for r in tables[seed][arm] if r['chain']=='heavy' and r['variant']=='correct']
            mapping={r['id']:r for r in values}
            if len(mapping)!=len(values):raise ValueError('Duplicate target')
            maps.append((mapping,coefficient))
        keys=set(maps[0][0])
        if any(set(m)!=keys for m,c in maps):raise ValueError('Unpaired evaluation membership')
        signature={k:(maps[0][0][k]['group'],int(maps[0][0][k]['ntokens']),maps[0][0][k]['sequence_sha256']) for k in keys}
        if expected is not None and signature!=expected:raise ValueError('Seed membership differs')
        expected=signature
        numerator=defaultdict(float);denominator=defaultdict(int)
        for key in sorted(keys):
            group,tokens,sequence=signature[key]
            denominator[group]+=tokens
            for mapping,coefficient in maps:
                r=mapping[key]
                if (r['group'],int(r['ntokens']),r['sequence_sha256'])!=(group,tokens,sequence):
                    raise ValueError('Group, target or token mismatch')
                numerator[group]+=coefficient*float(r['loss_sum'])
        groups=sorted(denominator)
        assembled.append([numerator[g] for g in groups]);denominators.append([denominator[g] for g in groups])
    return np.asarray(assembled),np.asarray(denominators)
