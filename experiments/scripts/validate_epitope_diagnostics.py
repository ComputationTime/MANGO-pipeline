"""Independent rank/statistic checks and exact RNase-A score decomposition."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
from pathlib import Path
import json
import statistics
import numpy as np
import pandas as pd
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_csv,write_json


def rho(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(pd.Series(a).rank(method='average').corr(pd.Series(b).rank(method='average')))


def main(root):
    root=Path(root);p=read_json(root/'protocol.json');cfg=p['settings'];report=read_json(root/'results_report.json')
    freeze(root/'validation_protocol.json',dict(validator_sha256=digest(__file__),purpose='Independent numeric verification and score decomposition; no model fitting or statistical tests.'))
    for field in ('inputs_sha256','implementation_sha256','checkpoints_sha256'):
        for name,value in p[field].items():
            if digest(name)!=value:raise ValueError('Frozen source changed: '+name)
    annotations=read_json(root/'regions.json')['sequences'];records={r['id']:r for r in read_csv(root/'native_records.csv')}
    contacts=read_json(root/'contacts.json')['records'];external=read_csv(root/'external_correlations.csv')
    expected={(r['arm'],int(r['seed']),r['target'],r['condition'],r['region']):float(r['rho']) if r['rho'] else None for r in external}
    differences=[];token_tables={};count=0
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_s{seed}';marker=read_json(folder/'complete.json')
            for name,value in marker['files_sha256'].items():
                if digest(folder/name)!=value:raise ValueError('Probe output changed')
            bins=defaultdict(list)
            for r in read_csv(folder/'external.csv'):bins[r['target'],r['condition']].append(r)
            for (target,condition),rows in bins.items():
                y=[-float(r['affinity']) for r in rows]
                for region in ('full','framework','cdr1','cdr2','cdr3','cdrs'):
                    if any(r[region]=='' for r in rows):continue
                    value=rho([-float(r[region]) for r in rows],y);stored=expected[arm,seed,target,condition,region]
                    if value is None:assert stored is None
                    else:differences.append(abs(value-stored))
                    count+=1
            raw=read_csv(folder/'native_tokens.csv');table=defaultdict(dict)
            for r in raw:table[r['id'],r['condition']][int(r['index'])]=float(r['nll'])
            token_tables[arm,seed]=table
            for r in read_csv(folder/'native.csv'):
                row=records[r['id']];labels=annotations[fingerprint(row['resolved_H_seq'])]['labels'];values=table[r['id'],r['condition']]
                assert len(values)==len(labels)
                for region in ('full','framework','cdr1','cdr2','cdr3','cdrs','paratope'):
                    if r.get(region,'')=='':continue
                    if region=='paratope':indices=contacts[r['id']]['contacts']['5.0']['paratope']
                    elif region=='full':indices=list(range(len(labels)))
                    elif region=='cdrs':indices=[i for i,l in enumerate(labels) if l.startswith('cdr')]
                    else:indices=[i for i,l in enumerate(labels) if l==region]
                    value=statistics.mean(values[i] for i in indices);differences.append(abs(value-float(r[region])))
    assert max(differences)<1e-12
    pair=read_json(Path(cfg['reference_root'])/'ranking_pair.json');a=records[pair['strong']['id']]['resolved_H_seq'];b=records[pair['weak']['id']]['resolved_H_seq']
    la=annotations[fingerprint(a)]['labels'];lb=annotations[fingerprint(b)]['labels'];assert len(a)==len(b) and la==lb
    parts=[]
    for family in ('IgLM','MANGO'):
        condition='correct' if family=='MANGO' else 'none'
        for seed in cfg['seeds']:
            gaps={}
            for suffix in ('CE','rank'):
                table=token_tables[family+'_'+suffix,seed]
                gaps[suffix]=np.array([table[pair['weak']['id'],condition][i]-table[pair['strong']['id'],condition][i] for i in range(len(la))])
            change=gaps['rank']-gaps['CE'];whole=float(change.mean())
            for region in ('framework','cdr1','cdr2','cdr3','eos'):
                contribution=float(sum(v for v,l in zip(change,la) if l==region)/len(change))
                parts.append(dict(family=family,seed=seed,region=region,contribution=contribution,total_gain=whole))
            for name,indices in [('mutated',[i for i in range(len(a)) if a[i]!=b[i]]),
                                 ('unchanged',[i for i in range(len(la)) if i==len(a) or a[i]==b[i]])]:
                parts.append(dict(family=family,seed=seed,region=name,contribution=float(change[indices].sum()/len(change)),total_gain=whole))
    write_csv(root/'ranking_gain_decomposition.csv',parts)
    summary=[]
    for family in ('IgLM','MANGO'):
        for region in ('framework','cdr1','cdr2','cdr3','eos','mutated','unchanged'):
            values=[r['contribution'] for r in parts if r['family']==family and r['region']==region]
            summary.append(dict(family=family,region=region,mean=statistics.mean(values),seed_sd=statistics.stdev(values) if len(values)>1 else None))
    gains=[r['total_gain'] for r in parts if r['family']=='MANGO' and r['region']=='cdr1']
    cdr1=next(r['mean'] for r in summary if r['family']=='MANGO' and r['region']=='cdr1')
    result=dict(complete=True,correlations_recomputed=count,max_discrepancy=max(differences),
                native_region_scores_reconstructed=True,source_checkpoint_hashes_verified=True,
                ranking_gain_decomposition=summary,MANGO_CDR1_share_of_mean_gain=cdr1/statistics.mean(gains),
                MANGO_training_pair_gain=statistics.mean(gains),
                mutated_sequence_positions=[i for i in range(len(a)) if a[i]!=b[i]],
                caveat='Autoregressive positional NLL depends on preceding residues. Contributions are not causal energetic effects of residues.',
                report_sha256=digest(root/'results_report.json'))
    write_json(root/'independent_validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/epitope_diagnostics_v1');a=p.parse_args();main(Path(a.root))
