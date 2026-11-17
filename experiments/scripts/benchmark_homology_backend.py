"""Validate GPU homology search before changing the protected-cohort auditor.

This tests search software, not an antibody model. Source protocols are read-only.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
from pathlib import Path
import random
import os
import signal
import subprocess
import shutil
import time

from affinity_study_common import read_json,write_json,digest,freeze,fingerprint

ROOT=Path('artifacts/experiments/homology_backend_probe_v1')
BINARY='artifacts/envs/cath-split/bin/mmseqs'
SOURCE=Path('artifacts/experiments/affinity_short_v3/searches/external_antibodies')
GOLD=Path('artifacts/experiments/controlled_antigen_v2/searches/all_antibody_lineages')


def fasta(path):
    values={};name=None
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):name=line[1:].strip();values[name]=''
            elif line.strip():values[name]+=line.strip()
    return values


def call(command,log,timeout):
    started=time.monotonic()
    with Path(log).open('w') as handle:
        child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            return dict(status='incomplete',reason='time_limit',seconds=time.monotonic()-started,command=command)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM);child.wait();raise
    return dict(status='complete' if code==0 else 'failed',returncode=code,seconds=time.monotonic()-started,command=command)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    protocol=dict(kind='homology_backend_validation',binary=BINARY,binary_sha256=digest(BINARY),
        binary_version=subprocess.check_output([BINARY,'version'],text=True).strip(),
        source_sha256={str(p):digest(p) for p in (SOURCE/'query.fa',SOURCE/'target.fa',GOLD/'query.fa',GOLD/'target.fa',GOLD/'hits.tsv')},
        implementation_sha256={str(p):digest(p) for p in (Path(__file__),Path('experiments/workflow/homology_backend.smk'))},
        filters=dict(identity=.9,bidirectional_coverage=.8,evalue=.001,max_seqs=100000),
        acceptance='All CPU-positive query IDs recovered; all CPU graph edges recovered; GPU-reported hits obey the same biological thresholds.',
        note='Sampled backend validation cannot prove equality of heuristic searches on every possible input.')
    freeze(ROOT/'protocol.json',protocol)
    for name in protocol['implementation_sha256']:
        destination=ROOT/'frozen_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(name,destination)
    queries=fasta(SOURCE/'query.fa');targets=fasta(SOURCE/'target.fa')
    prior=Path('artifacts/experiments/affinity_improvement_v1/searches/external_antibodies/hits.tsv')
    with prior.open() as handle:known={line.split('\t')[0] for line in handle}&set(queries)
    selected={name:queries[name] for name in sorted(known)[:64]}
    unknown=sorted(set(queries)-known,key=lambda n:fingerprint('1729:'+n))[:64]
    selected.update({name:queries[name] for name in unknown})
    rng=random.Random(1729);controls={}
    for i,key in enumerate(sorted(targets)[:16]):
        seq=targets[key];exact='exact_'+str(i);near='near_'+str(i);negative='shuffle_'+str(i)
        selected[exact]=seq;position=len(seq)//2
        selected[near]=seq[:position]+('A' if seq[position]!='A' else 'G')+seq[position+1:]
        residues=list(seq);rng.shuffle(residues);selected[negative]=''.join(residues)
        controls[exact]='exact_positive';controls[near]='single_substitution_positive';controls[negative]='composition_shuffle_control'
    path=ROOT/'queries.fa';path.write_text(''.join(f'>{k}\n{s}\n' for k,s in sorted(selected.items())))
    freeze(ROOT/'selection.json',dict(query_sha256=digest(path),queries=len(selected),known_natural_positives=len(known&set(selected)),
        controls=controls,selection='64 known-positive +64 new hash-selected +16 exact/near/shuffle controls'))
    for name,path in [('external',SOURCE/'target.fa'),('graph',GOLD/'target.fa')]:
        database=ROOT/(name+'_db');padded=ROOT/(name+'_padded')
        a=call([BINARY,'createdb',str(path),str(database),'--shuffle','0'],ROOT/'logs'/f'createdb_{name}.log',120)
        if a['status']!='complete':write_json(ROOT/'prepared.json',a);return
        b=call([BINARY,'makepaddedseqdb',str(database),str(padded)],ROOT/'logs'/f'pad_{name}.log',120)
        if b['status']!='complete':write_json(ROOT/'prepared.json',b);return
    write_json(ROOT/'prepared.json',dict(status='complete',selection_sha256=digest(ROOT/'selection.json')))


def search(kind):
    if read_json(ROOT/'prepared.json')['status']!='complete':raise ValueError('Backend preparation did not complete')
    protocol=read_json(ROOT/'protocol.json')
    if digest(BINARY)!=protocol['binary_sha256']:raise ValueError('MMseqs binary changed')
    graph=kind=='gpu_graph';gpu=kind!='cpu';query=GOLD/'query.fa' if graph else ROOT/'queries.fa'
    target=ROOT/('graph_padded' if graph else 'external_padded') if gpu else SOURCE/'target.fa'
    command=[BINARY,'easy-search',str(query),str(target),str(ROOT/(kind+'.tsv')),str(ROOT/('tmp_'+kind)),
        '--min-seq-id','0.9','-c','0.8','--cov-mode','0','-e','0.001','--threads','4','--max-seqs','100000',
        '--alignment-mode','3','--format-output','query,target,fident,qcov,tcov,evalue']
    if gpu:command+=['--gpu','1']
    else:command+=['-s','7.5']
    if not graph:command+=['--max-accept','1']
    result=call(command,ROOT/'logs'/(kind+'.log'),300)
    if result['status']=='complete':result['hits_sha256']=digest(ROOT/(kind+'.tsv'))
    write_json(ROOT/(kind+'.json'),result)


def hits(path):
    values=[]
    with Path(path).open() as handle:
        for row in csv.reader(handle,delimiter='\t'):
            if len(row)!=6:raise ValueError('Invalid alignment output')
            if float(row[2])<.9 or min(float(row[3]),float(row[4]))<.8 or float(row[5])>.001:raise ValueError('Hit violates biological thresholds')
            values.append(row)
    return values


def report():
    states={name:read_json(ROOT/(name+'.json')) for name in ('cpu','gpu','gpu_graph') if (ROOT/(name+'.json')).exists()}
    result=dict(complete=len(states)==3 and all(v['status']=='complete' for v in states.values()),stages=states,validated=False,
        training_launched=False,biological_thresholds_relaxed=False)
    if result['complete']:
        cpu={r[0] for r in hits(ROOT/'cpu.tsv')};gpu={r[0] for r in hits(ROOT/'gpu.tsv')}
        gold={(r[0],r[1]) for r in hits(GOLD/'hits.tsv')};graph={(r[0],r[1]) for r in hits(ROOT/'gpu_graph.tsv')}
        controls=read_json(ROOT/'selection.json')['controls'];positive={k for k,v in controls.items() if v.endswith('positive')}
        missing=sorted(cpu-gpu);missing_graph=sorted(gold-graph)
        result.update(validated=not missing and not missing_graph and positive<=gpu,
            cpu_positive_queries=len(cpu),gpu_positive_queries=len(gpu),cpu_positives_missing_on_gpu=missing,
            cpu_graph_edges=len(gold),gpu_graph_edges=len(graph),missing_graph_edges=len(missing_graph),
            missing_positive_controls=sorted(positive-gpu),gpu_additional_positive_queries=len(gpu-cpu),
            boolean_speed_ratio=states['cpu']['seconds']/states['gpu']['seconds'])
    write_json(ROOT/'results_report.json',result)
    write_json(ROOT/'state.json',dict(status='complete' if result['complete'] else 'incomplete',stages=states))
    (ROOT/'findings.md').write_text(f'# Homology backend probe\n\nExecution complete: {result["complete"]}.\nGPU backend validated on sampled inputs: {result["validated"]}.\n\nNo antibody model was trained. No protection threshold was relaxed.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu','gpu','gpu_graph','report'])
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='report':report()
    else:search(args.stage)
