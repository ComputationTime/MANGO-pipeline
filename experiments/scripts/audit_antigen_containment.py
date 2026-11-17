"""Chain-aware external-antigen containment audit, without model training."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import shutil

from affinity_study_common import digest, fingerprint, freeze, read_json, write_csv, write_json
from benchmark_homology_backend import call
from prepare_affinity_study import fasta

ROOT=Path('artifacts/experiments/antigen_containment_audit_v1')
REFERENCE=Path('artifacts/experiments/affinity_short_v3a')


def run_search(binary,folder,queries,targets,mode,timeout):
    folder.mkdir(parents=True,exist_ok=True)
    fasta(folder/'query.fa',queries);fasta(folder/'target.fa',targets)
    output=folder/'hits.tsv'
    args=[binary,'easy-search',str(folder/'query.fa'),str(folder/'target.fa'),str(output),str(folder/'tmp'),
        '--min-seq-id','0.2','-c','0.6','--cov-mode',str(mode),'-e','0.001','-s','7.5',
        '--threads','4','--max-seqs','100000','--max-accept','1','--alignment-mode','3',
        '--format-output','query,target,fident,qcov,tcov,evalue']
    result=call(args,folder/'search.log',timeout)
    if result['status']=='complete':
        with output.open() as handle:rows=list(csv.reader(handle,delimiter='\t'))
        for row in rows:
            if len(row)!=6 or float(row[2])<.2 or float(row[5])>.001:raise ValueError('Invalid antigen alignment')
            qcov,tcov=map(float,row[3:5])
            if (mode==0 and min(qcov,tcov)<.6) or (mode==1 and tcov<.6) or (mode==2 and qcov<.6):
                raise ValueError('Antigen coverage threshold failed')
        result.update(hits=len(rows),query_ids=sorted({r[0] for r in rows}),hits_sha256=digest(output))
    return result


def controls(binary):
    rng=random.Random(1729);sequence=''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(240))
    fragment=sequence[70:160];unrelated=''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(240))
    cases=[('long_query',sequence,fragment,{0:False,1:True,2:False}),
        ('short_query',fragment,sequence,{0:False,1:False,2:True}),
        ('unrelated',unrelated,fragment,{0:False,1:False,2:False})]
    results=[]
    for name,query,target,expected in cases:
        for mode,positive in expected.items():
            stage=run_search(binary,ROOT/'controls'/f'{name}_{mode}',[query],[target],mode,30)
            results.append(dict(case=name,mode=mode,expected_positive=positive,stage=stage,
                passed=stage['status']=='complete' and bool(stage['hits'])==positive))
    return results


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg=read_json(REFERENCE/'protocol.json')['settings'];binary=cfg['mmseqs']
    source=Path(cfg['source_records']);external=Path(cfg['external_root'])/'antigens.csv'
    protocol=dict(reference_protocol_sha256=digest(REFERENCE/'protocol.json'),
        input_sha256={str(p):digest(p) for p in (source,external)},implementation_sha256={str(Path(__file__)):digest(__file__)},
        binary_sha256=digest(binary),filters=dict(identity=.2,coverage=.6,evalue=.001,modes=[1,2]),
        maximum_antigen_length=2048,per_search_cpu_seconds=180,
        question='Does directional, chain-aware homology recover protected antigen domains missed by bidirectional whole-antigen coverage?',
        scope='Antigen-only protection audit and capacity bounds; not complete antibody-lineage screening or final split allocation.',
        gpu_used=False,model_training=False)
    freeze(ROOT/'protocol.json',protocol)
    saved=ROOT/'frozen_source'/Path(__file__).name
    saved.parent.mkdir(exist_ok=True)
    if not saved.exists():shutil.copyfile(__file__,saved)
    write_json(ROOT/'state.json',dict(status='running'))
    test=controls(binary)
    if not all(r['passed'] for r in test):
        write_json(ROOT/'results_report.json',dict(complete=False,controls=test,reason='containment_control_failure'))
        write_json(ROOT/'state.json',dict(status='failed'));return
    rows=[];membership=defaultdict(set)
    with source.open() as handle:
        for row in csv.DictReader(handle):
            if len(row['resolved_ag_seq'])>2048 or row.get('has_buzz')=='True' or 'alpha' in row.get('source','').lower() or any(s.startswith('flab_') for s in row['source'].lower().split(';')):continue
            rows.append(row)
            for chain in row['resolved_ag_seq'].split(','):
                if chain:membership[fingerprint(chain)].add(row['resolved_ag_seq'])
    targets={part for row in csv.DictReader(external.open()) for part in row['resolved_ag_seq'].split(',') if part}
    queries={part for row in rows for part in row['resolved_ag_seq'].split(',') if part}
    stages={str(mode):run_search(binary,ROOT/'searches'/f'coverage_{mode}',queries,targets,mode,180) for mode in (1,2)}
    result=dict(complete=all(s['status']=='complete' for s in stages.values()),controls=test,stages=stages,
        source_records=len(rows),query_chains=len(queries),target_chains=len(targets),full_audit=False,
        training_eligibility='unknown',model_training=False,gpu_used=False)
    if result['complete']:
        blocked={ag for s in stages.values() for key in s['query_ids'] for ag in membership[key]}
        external_exact={row['resolved_ag_seq'] for row in csv.DictReader(external.open())}
        blocked|=external_exact
        excluded=[dict(id=row['id'],source_split=row['split'],source_group=row['ab_ag_cluster'],reason='external_antigen_directional_homology')
            for row in rows if row['resolved_ag_seq'] in blocked]
        write_csv(ROOT/'antigen_exclusions.csv',excluded)
        blocked_groups={row['ab_ag_cluster'] for row in rows if row['resolved_ag_seq'] in blocked}
        bounds={}
        for policy in ('matching_antigen_rows','entire_source_groups_containing_match'):
            retained=[row for row in rows if (row['resolved_ag_seq'] not in blocked if policy=='matching_antigen_rows' else row['ab_ag_cluster'] not in blocked_groups)]
            bounds[policy]={split:dict(records=sum(row['split']==split for row in retained),
                original_groups=len({row['ab_ag_cluster'] for row in retained if row['split']==split}),
                antigens=len({row['resolved_ag_seq'] for row in retained if row['split']==split})) for split in ('train','val','test')}
        result.update(excluded_records=len(excluded),excluded_by_split=dict(Counter(row['source_split'] for row in excluded)),
            matching_source_groups=len(blocked_groups),retained_upper_bounds=bounds,
            exclusion_sha256=digest(ROOT/'antigen_exclusions.csv'),
            limitations=['Antibody near-lineage filtering and final component allocation remain undone.',
                'Original group counts are not independent biological-component counts.',
                'Whole-group exclusion is a separate conservative bound, not an automatically adopted new cohort.'])
    write_json(ROOT/'results_report.json',result)
    write_json(ROOT/'state.json',dict(status='complete' if result['complete'] else 'incomplete'))
    (ROOT/'findings.md').write_text('# Antigen containment audit\n\n'+
        f'Execution complete: {result["complete"]}.\n'+
        'No GPU or model training was used. This is not a completed cohort eligibility audit.\n'+
        f'Antigen-based excluded records: {result.get("excluded_records","unavailable")}.\n')


if __name__=='__main__':main()
