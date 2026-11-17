"""Null-calibrate the seed-only special case of the crossed percentile interval.

No biological data, model outputs, or GPU are used. This is a necessary stress
test for confirmation inference, not a validation of affinity correlations.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, t

from affinity_study_common import digest, freeze, write_json, write_csv

ROOT=Path('artifacts/experiments/interval_calibration_v1')


def simulate(seed_count, distribution, repetitions, draws, rng):
    """All biological components share each seed's paired effect; E[effect]=0."""
    missed={'percentile':0,'seed_t_conditional_on_panel':0}
    positive={name:0 for name in missed}
    for start in range(0,repetitions,64):
        size=min(64,repetitions-start)
        if distribution=='normal':samples=rng.normal(size=(size,seed_count))
        elif distribution=='centered_exponential':samples=rng.exponential(size=(size,seed_count))-1
        else:raise ValueError('Unknown distribution')
        point=samples.mean(1)
        indices=rng.integers(seed_count,size=(size,draws,seed_count))
        bootstrap=samples[np.arange(size)[:,None,None],indices].mean(2)
        bounds=np.quantile(bootstrap,[.025,.975],axis=1)
        width=t.ppf(.975,seed_count-1)*samples.std(1,ddof=1)/math.sqrt(seed_count)
        intervals={'percentile':bounds,'seed_t_conditional_on_panel':np.array([point-width,point+width])}
        for name,(low,high) in intervals.items():
            missed[name]+=int(np.count_nonzero((low>0)|(high<0)))
            positive[name]+=int(np.count_nonzero(low>0))
    results=[]
    for method,number in missed.items():
        low=0. if not number else float(beta.ppf(.025,number,repetitions-number+1))
        high=1. if number==repetitions else float(beta.ppf(.975,number+1,repetitions-number))
        results.append(dict(seeds=seed_count,distribution=distribution,method=method,
            simulations=repetitions,noncoverage=number/repetitions,
            noncoverage_mc_low=low,noncoverage_mc_high=high,
            positive_lower_bound_rate=positive[method]/repetitions,
            fails_nominal_coverage_check=bool(low>.05)))
    return results


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed_counts=[3,5,10,20],distributions=['normal','centered_exponential'],
        simulations_per_scenario=4000,bootstrap_draws=1999,random_seed=172914,
        two_sided_nominal_alpha=.05,model='paired_effect[seed,component] = seed_effect; all component token counts identical',
        estimand='mean paired effect over independent seeds, with zero biological variance',
        comparison='Exact same seed percentile resampling as the seed-only special case of crossed_interval; Student t as a conditional-on-panel diagnostic reference.',
        rejection_rule='Necessary calibration fails if the 95% exact binomial lower bound for two-sided noncoverage exceeds 0.05.',
        limitations=['This test does not validate inference across biological components or assay correlations.',
            'Passing these finite simulations does not establish general coverage.',
            'Student t intervals assume a normal mean for exact small-sample calibration; they are not automatically a replacement method.',
            'No interval method is selected from model test outcomes.'],
        sources=['https://arxiv.org/abs/0712.1111','https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm'],
        implementation_sha256={str(Path(__file__)):digest(__file__),
            'experiments/scripts/controlled_antigen_common.py':digest('experiments/scripts/controlled_antigen_common.py')})
    freeze(ROOT/'protocol.json',protocol)
    started=time.monotonic();rng=np.random.default_rng(protocol['random_seed']);rows=[]
    write_json(ROOT/'state.json',dict(status='running',started_at=datetime.now(timezone.utc).isoformat()))
    for count in protocol['seed_counts']:
        for distribution in protocol['distributions']:
            rows.extend(simulate(count,distribution,protocol['simulations_per_scenario'],protocol['bootstrap_draws'],rng))
    write_csv(ROOT/'calibration.csv',rows)
    result=dict(complete=True,results=rows,seconds=time.monotonic()-started,
        scientific_models_trained=0,confirmation_method_approved=False,
        protocol_sha256=digest(ROOT/'protocol.json'),calibration_sha256=digest(ROOT/'calibration.csv'))
    write_json(ROOT/'results_report.json',result)
    write_json(ROOT/'state.json',dict(status='complete',seconds=result['seconds']))
    lines=['# Seed-interval null calibration','',
        'No biological labels, trained-model outcomes or GPU were used.',
        'This is a necessary seed-axis stress test, not approval of a confirmation method.','',
        '| Seeds | Distribution | Percentile noncoverage | Seed t reference noncoverage |',
        '| --- | --- | ---: | ---: |']
    for count in protocol['seed_counts']:
        for distribution in protocol['distributions']:
            chosen=[r for r in rows if r['seeds']==count and r['distribution']==distribution]
            values={r['method']:r['noncoverage'] for r in chosen}
            lines.append(f'| {count} | {distribution} | {values["percentile"]:.3f} | {values["seed_t_conditional_on_panel"]:.3f} |')
    lines+=['','Nominal two-sided noncoverage is 0.05. Monte Carlo intervals are in calibration.csv.',
        'The t reference does not address biological resampling or nonnormal small-sample assumptions.',
        'Existing exploratory reports and frozen model studies are unchanged.']
    (ROOT/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
