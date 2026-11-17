"""Independent exposure, AP and factorial checks after the fixed study finishes."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import statistics
import numpy as np
import pandas as pd
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_json


def reference_ap(truth,score):
    """Each positive contributes precision at its own threshold; handles ties."""
    truth=np.asarray(truth,bool);score=np.asarray(score,float)
    if not truth.any():return None
    return float(np.mean([truth[score>=threshold].sum()/np.count_nonzero(score>=threshold) for threshold in score[truth]]))


def random_ap_expectation(n,p):
    harmonic=sum(1/i for i in range(1,n+1))/n
    return 1. if n==1 else harmonic+(p-1)/(n-1)*(1-harmonic)


def main(root):
    root=Path(root);p=read_json(root/'protocol.json');cfg=p['settings'];result=read_json(root/'results_report.json')
    freeze(root/'validation_protocol.json',dict(validator_sha256=digest(__file__),scope='Independent verification, not extra model/metric selection.'))
    for key in ('inputs_sha256','implementation_sha256'):
        for path,value in p[key].items():
            if digest(path)!=value:raise ValueError('Frozen source changed: '+path)
    rows=read_csv(root/'records.csv');byid={r['id']:r for r in rows};pair=read_json(root/'ranking_pair.json')
    for key in ('ab_ag_cluster','biological_group'):
        assert not ({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='test'})
    assert all(r['source_split']=='train' for r in rows if r['split']=='train')
    prepared=read_json(root/'prepared.json');canonical={r['id']:r for r in rows+read_csv(root/'antigens.csv')}
    for r in rows:
        source=canonical[prepared['inputs'][fingerprint(r['resolved_ag_seq'])]['canonical_record']]
        assert source['resolved_ag_seq']==r['resolved_ag_seq'] and source['split']==r['split']
    table={(r['arm'],int(r['seed']),r['variant']):r for r in read_csv(root/'per_seed.csv')}
    diffs=[];objective_error=0.;ap_count=0;nulls=[]
    for seed in cfg['seeds']:
        exposure=[];tokens=0
        for phase in ('pretrain','adapt'):
            for batch in read_json(root/'schedules'/f'{phase}_{seed}.json')['items']:
                examples=[(r['id'],r['chain']) for r in batch]
                if phase=='adapt':examples += [(pair[k]['id'],'heavy') for k in ('strong','weak')]
                assert all(byid[rid]['split']=='train' for rid,chain in examples)
                exposure.extend(examples);tokens+=sum(len(byid[rid]['resolved_H_seq' if c=='heavy' else 'resolved_L_seq'])+1 for rid,c in examples)
        for arm in cfg['arms']:
            folder=root/'runs'/f'{arm}_s{seed}';fit=read_json(folder/'complete.json');evaluation=read_json(folder/'evaluation.json')
            assert fit['exposure_sha256']==fingerprint(json.dumps(exposure));assert fit['target_tokens']==tokens
            assert digest(folder/'final.pt')==fit['checkpoint_sha256']==evaluation['checkpoint_sha256']
            curve=read_csv(folder/'training_curve.csv');assert len(curve)==cfg['pretraining_updates']+cfg['adaptation_updates']
            for r in curve:
                actual=float(r['objective']);wanted=float(r['nll'])+(cfg['contact_weight']*float(r['contact_loss']) if arm!='iglm_ce' else 0.)
                objective_error=max(objective_error,abs(actual-wanted));assert abs(actual-wanted)<2e-6
            external=read_csv(folder/'external.csv');rhos=[]
            for target in ('HER2','PD1','TIGIT'):
                selected=[r for r in external if r['target']==target]
                rho=float(pd.Series([-float(r['nll']) for r in selected]).rank().corr(pd.Series([-float(r['affinity']) for r in selected]).rank()))
                diffs.append(abs(rho-float(table[arm,seed,'correct']['affinity_'+target])));rhos.append(rho)
            diffs.append(abs(statistics.mean(rhos)-float(table[arm,seed,'correct']['affinity_macro'])))
            if arm=='iglm_ce':continue
            groups=defaultdict(list)
            for row in read_csv(folder/'native.csv'):
                path=folder/'contacts'/f'{fingerprint(row["id"])}_{row["variant"]}.npz';payload=np.load(path);truth=payload['truth'];score=payload['logits']
                assert np.isfinite(score).all();per_row=[reference_ap(a,b) for a,b in zip(truth,score) if a.any()]
                ap=float(np.mean(per_row));ap_count+=len(per_row);groups[row['variant'],row['group']].append(ap)
                if arm==cfg['arms'][0] and seed==cfg['seeds'][0] and row['variant']=='correct':
                    null=float(np.mean([random_ap_expectation(len(a),int(a.sum())) for a in truth if a.any()]))
                    nulls.append((row['group'],null))
            for variant in {k[0] for k in groups}:
                value=float(np.mean([np.mean(v) for (va,g),v in groups.items() if va==variant]))
                diffs.append(abs(value-float(table[arm,seed,variant]['contact_ap'])))
    primary=next(r for r in result['contrasts'] if r['contrast']=='interaction' and r['metric']=='contact_ap')
    interaction=[float(table['geom_contact',s,'correct']['contact_ap'])-float(table['geom_ce',s,'correct']['contact_ap'])
                 -float(table['seq_contact',s,'correct']['contact_ap'])+float(table['seq_ce',s,'correct']['contact_ap']) for s in cfg['seeds']]
    diffs.append(abs(statistics.mean(interaction)-primary['mean']))
    if len(interaction)>1:diffs.append(abs(statistics.stdev(interaction)-primary['seed_sd']))
    assert max(diffs)<1e-12
    grouped=defaultdict(list)
    for group,value in nulls:grouped[group].append(value)
    write_json(root/'independent_validation.json',dict(complete=True,models=len(cfg['arms'])*len(cfg['seeds']),
        individual_residue_APs_recomputed=ap_count,max_numeric_discrepancy=max(diffs),max_logged_objective_roundoff=objective_error,
        source_hashes_split_boundaries_and_canonicality_verified=True,exact_training_exposure_verified=True,
        expected_random_ranking_AP=float(np.mean([np.mean(v) for v in grouped.values()])),
        random_AP_scope='Exact uniform random ranking expectation per positive native query row, group-balanced; descriptive reference, not a p-value.',
        report_sha256=digest(root/'results_report.json')))
    print(json.dumps(read_json(root/'independent_validation.json'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/interface_supervision_v1');a=p.parse_args();main(Path(a.root))
