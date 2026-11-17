"""Predeclared contrasts and crossed uncertainty for decoder-matched ablations."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import defaultdict
import numpy as np

PRIMARY = {'pairing': [('correct',-1.),('shuffled',1.)],
           'content': [('correct',-1.),('constant',1.)],
           'branch': [('constant',-1.),('iglm',1.)],
           'overall': [('correct',-1.),('iglm',1.)]}
SECONDARY = {
    'pairing_frozen': [('frozen_correct',-1.),('frozen_shuffled',1.)],
    'pairing_joint': [('joint_correct',-1.),('joint_shuffled',1.)],
    'freezing_correct': [('frozen_correct',-1.),('joint_correct',1.)],
    'interaction': [('frozen_correct',-1.),('frozen_shuffled',1.),('joint_correct',1.),('joint_shuffled',-1.)]}


def crossed_interval(loss_sums, token_counts, repeats=10000, seed=1729):
    """Resample paired seed and biological-component axes independently.

    Arrays are [seed, component]. Paired contrast losses can be signed. Repeated
    seeds are optimization replicates, never counted as additional components.
    """
    losses=np.asarray(loss_sums,dtype=float);counts=np.asarray(token_counts,dtype=float)
    if losses.shape!=counts.shape or losses.ndim!=2 or min(losses.shape)<1:
        raise ValueError('Expected matched nonempty seed-by-component arrays')
    if not np.isfinite(losses).all() or not np.isfinite(counts).all() or (counts<=0).any():
        raise ValueError('Invalid paired losses/token counts')
    point=float(np.mean(losses.sum(1)/counts.sum(1)))
    if min(losses.shape)<2:return dict(estimate=point,low=None,high=None,pvalue=1.,seeds=losses.shape[0],groups=losses.shape[1],
                                      seed_points=(losses.sum(1)/counts.sum(1)).tolist())
    rng=np.random.default_rng(seed);draws=np.empty(repeats)
    for i in range(repeats):
        seeds=rng.integers(losses.shape[0],size=losses.shape[0])
        groups=rng.integers(losses.shape[1],size=losses.shape[1])
        selected=losses[np.ix_(seeds,groups)];denom=counts[np.ix_(seeds,groups)]
        draws[i]=np.mean(selected.sum(1)/denom.sum(1))
    lo,hi=np.quantile(draws,[.025,.975])
    p=(1+int(np.count_nonzero(np.abs(draws-point)>=abs(point))))/(repeats+1)
    return dict(estimate=point,low=float(lo),high=float(hi),pvalue=p,seeds=losses.shape[0],groups=losses.shape[1],
                seed_points=(losses.sum(1)/counts.sum(1)).tolist())


def score_arrays(tables, contrast, chain='heavy', region='full'):
    """Check exact row pairing before forming seed-by-component contrasts."""
    seeds=sorted(tables);groups=set();assembled=[];tokens=[]
    for seed in seeds:
        maps=[]
        for arm,coefficient in contrast:
            if arm not in tables[seed]:raise ValueError('Incomplete contrast arms')
            rows=[r for r in tables[seed][arm] if r['chain']==chain and r['variant']=='correct']
            mapping={(r['id'],r['chain']):r for r in rows}
            if len(mapping)!=len(rows):raise ValueError('Duplicate target scores')
            maps.append((mapping,coefficient))
        keys=set(maps[0][0])
        if any(set(m)!=keys for m,c in maps):raise ValueError('Unpaired comparison targets')
        sums=defaultdict(float);denoms=defaultdict(int)
        for key in keys:
            ref=maps[0][0][key];group=ref['group']
            loss_key='loss_sum' if region=='full' else region+'_sum'
            token_key='ntokens' if region=='full' else region+'_tokens'
            n=int(ref[token_key])
            if n==0:continue
            for mapping,coefficient in maps:
                row=mapping[key]
                if row['group']!=group or int(row[token_key])!=n:raise ValueError('Group or token mismatch')
                sums[group]+=coefficient*float(row[loss_key])
            denoms[group]+=n
        if assembled and set(sums)!=set(assembled[0]):raise ValueError('Seed membership differs')
        assembled.append(sums);tokens.append(denoms);groups.update(sums)
    groups=sorted(groups)
    return np.array([[v[g] for g in groups] for v in assembled]),np.array([[v[g] for g in groups] for v in tokens])
