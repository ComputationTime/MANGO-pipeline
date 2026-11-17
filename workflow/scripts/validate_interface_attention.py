"""Independent ranks/AP, exposure, output provenance and preservation checks."""
import argparse
from collections import defaultdict
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from affinity_study_common import digest, fingerprint, read_csv, read_json, write_json


def reference_ap(truth, scores):
    truth=np.asarray(truth,bool)
    return float(np.mean([truth[scores>=threshold].sum()/np.count_nonzero(scores>=threshold)
                          for threshold in scores[truth]]))


def main(root):
    root=Path(root);protocol=read_json(root/'protocol.json');cfg=protocol['settings']
    for field in ('inputs_sha256','implementation_sha256'):
        for path,expected in protocol[field].items():
            if digest(path)!=expected:raise ValueError('Frozen source changed: '+path)
    preserved=read_json(root/'preserved_previous.json')['files_sha256']
    for path,expected in preserved.items():
        if digest(path)!=expected:raise ValueError('Previous MANGO source/output changed: '+path)
    prepared=read_json(root/'prepared.json')
    for field in ('inputs','labels'):
        for info in prepared[field].values():
            if digest(info['path'])!=info['sha256']:raise ValueError('Reused input/label changed')
    rows=read_csv(root/'records.csv');byid={r['id']:r for r in rows}
    for key in ('ab_ag_cluster','biological_group'):
        if {r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']=='test'}:
            raise ValueError('Protected group crossed splits')
    if not all(r['source_split']=='train' for r in rows if r['split']=='train'):
        raise ValueError('Previous holdout entered training')
    pair=read_json(root/'ranking_pair.json')
    table={(r['arm'],int(r['seed']),r['variant']):r for r in read_csv(root/'per_seed.csv')}
    discrepancies=[];ap_count=0;objective_error=0.;models=0
    for seed in cfg['seeds']:
        exposure=[];tokens=0
        for phase in ('pretrain','adapt'):
            for batch in read_json(root/'schedules'/f'{phase}_{seed}.json')['items']:
                examples=[(r['id'],r['chain']) for r in batch]
                if phase=='adapt':examples += [(pair[k]['id'],'heavy') for k in ('strong','weak')]
                if not all(byid[rid]['split']=='train' for rid,chain in examples):raise ValueError('Holdout exposure')
                exposure.extend(examples)
                tokens+=sum(len(byid[rid]['resolved_H_seq' if c=='heavy' else 'resolved_L_seq'])+1 for rid,c in examples)
        for arm in cfg['arms']:
            models+=1;folder=root/'runs'/f'{arm}_s{seed}'
            fit=read_json(folder/'complete.json');done=read_json(folder/'evaluation.json')
            if fit['exposure_sha256']!=fingerprint(json.dumps(exposure)) or fit['target_tokens']!=tokens:
                raise ValueError('Exposure mismatch')
            if digest(folder/'final.pt')!=fit['checkpoint_sha256'] or fit['checkpoint_sha256']!=done['checkpoint_sha256']:
                raise ValueError('Checkpoint mismatch')
            curve=read_csv(folder/'training_curve.csv')
            if len(curve)!=cfg['pretraining_updates']+cfg['adaptation_updates']:raise ValueError('Update mismatch')
            for row in curve:
                wanted=float(row['nll'])+(cfg['contact_weight']*float(row['contact_loss']) if arm!='iglm' else 0.)
                objective_error=max(objective_error,abs(float(row['objective'])-wanted))
            if objective_error>2e-6:raise ValueError('Wrong objective')
            external=read_csv(folder/'external.csv');native=read_csv(folder/'native.csv')
            variants=sorted({r['variant'] for r in external})
            for variant in variants:
                correlations=[]
                for target in ('HER2','PD1','TIGIT'):
                    subset=[r for r in external if r['variant']==variant and r['target']==target]
                    x=pd.Series([-float(r['nll']) for r in subset]).rank()
                    y=pd.Series([-float(r['affinity']) for r in subset]).rank()
                    value=float(x.corr(y));correlations.append(value)
                    discrepancies.append(abs(value-float(table[arm,seed,variant]['affinity_'+target])))
                discrepancies.append(abs(statistics.mean(correlations)-float(table[arm,seed,variant]['affinity_macro'])))
            if arm=='control':
                # A sham intervention must have exactly zero effect when
                # predicted signals are not routed into the generator.
                for metric in ('affinity_macro','heavy_nll','attention_ap','contact_ap'):
                    values=[float(table[arm,seed,v][metric]) for v in variants]
                    if max(values)!=min(values):raise ValueError('Control changed under sham intervention')
            if arm=='iglm':continue
            groups=defaultdict(list)
            for row in native:
                variant=row['variant'];path=folder/'contacts'/f'{fingerprint(row["id"])}_{variant}.npz'
                payload=np.load(path);truth=payload['truth'];pred=payload['logits'];att=payload['attention']
                contact=[];attention=[];enrichment=[]
                for y,c,a in zip(truth,pred,att):
                    if not y.any():continue
                    contact.append(reference_ap(y,c));attention.append(reference_ap(y,a));ap_count+=2
                    enrichment.append(float(sum(float(a[j]) for j in np.flatnonzero(y))/(np.count_nonzero(y)/len(y))))
                for metric,value in [('contact_ap',statistics.mean(contact)),('attention_ap',statistics.mean(attention)),
                                     ('attention_enrichment',statistics.mean(enrichment))]:
                    groups[variant,metric,row['group']].append(value)
            for variant in variants:
                for metric in ('contact_ap','attention_ap','attention_enrichment'):
                    value=statistics.mean(statistics.mean(v) for (va,m,g),v in groups.items() if va==variant and m==metric)
                    discrepancies.append(abs(value-float(table[arm,seed,variant][metric])))
    result=read_json(root/'results_report.json')
    for row in result['contrasts']:
        points=row['seed_points']
        discrepancies.append(abs(statistics.mean(points)-row['mean']))
        if len(points)>1:discrepancies.append(abs(statistics.stdev(points)-row['seed_sd']))
    if max(discrepancies)>3e-6:raise ValueError('Independent metric discrepancy: '+str(max(discrepancies)))
    write_json(root/'independent_validation.json',dict(
        complete=True,models=models,individual_residue_APs_recomputed=ap_count,
        max_numeric_discrepancy=max(discrepancies),max_logged_objective_roundoff=objective_error,
        previous_sources_and_checkpoints_preserved=True,preserved_file_count=len(preserved),
        split_and_exposure_checks_passed=True,control_sham_interventions_exactly_zero=True,
        report_sha256=digest(root/'results_report.json')))
    print(json.dumps(read_json(root/'independent_validation.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(Path(parser.parse_args().root))
