"""Bounded CPU witness precheck; unresolved sequences are never called safe."""
from collections import defaultdict
from pathlib import Path
import csv
import heapq
import shutil
import sys

from affinity_study_common import digest, fingerprint, freeze, read_json, write_json
from benchmark_homology_backend import fasta, call
from witness_homology_search import verify_candidates, checked_hits

ROOT=Path('artifacts/experiments/cpu_witness_probe_v1')
SOURCE=Path('artifacts/experiments/affinity_short_v3/searches/external_antibodies')
BINARY='artifacts/envs/cath-split/bin/mmseqs'


def candidates(queries,targets,per_query=16):
    if per_query<1:raise ValueError('Candidate limit must be positive')
    def bucket(seq):return len(seq)//16,seq[:6],seq[-4:]
    groups=defaultdict(list);exact={seq:key for key,seq in targets.items()}
    for key,seq in targets.items():groups[bucket(seq)].append(key)
    representatives={key:heapq.nsmallest(per_query,ids,key=lambda n:fingerprint(targets[n])) for key,ids in groups.items()}
    for key,seq in sorted(queries.items()):
        if seq in exact:
            yield key,exact[seq]
        else:
            for target in representatives.get(bucket(seq),[]):yield key,target


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    validation=Path('artifacts/experiments/witness_revalidation_v1/results_report.json')
    if not read_json(validation)['validated']:raise ValueError('Sparse realignment has not passed validation')
    protocol=dict(source_sha256={str(p):digest(p) for p in (SOURCE/'query.fa',SOURCE/'target.fa')},
        implementation_sha256={str(Path(__file__)):digest(__file__),
            'workflow/scripts/witness_homology_search.py':digest('workflow/scripts/witness_homology_search.py')},
        binary_sha256=digest(BINARY),validation_sha256=digest(validation),
        candidates_per_query=16,selection='Exact sequence when available; otherwise fixed hash representatives in matching length/16, prefix6, suffix4 buckets.',
        cutoff=dict(identity=.9,both_coverage=.8,evalue=.001,full_database_evalues=True),
        cpu_time_limit_seconds=180,full_audit=False,unresolved_policy='Unknown; must receive complete original search before any eligibility decision.',
        gpu_used=False,model_training=False)
    freeze(ROOT/'protocol.json',protocol)
    for name in protocol['implementation_sha256']:
        destination=ROOT/'frozen_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(name,destination)
    write_json(ROOT/'state.json',dict(status='running'))
    queries=fasta(SOURCE/'query.fa');targets=fasta(SOURCE/'target.fa');count=0;proposed=set()
    with (ROOT/'candidates.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
        for query,target in candidates(queries,targets,protocol['candidates_per_query']):
            writer.writerow([query,target]);count+=1;proposed.add(query)
    stage=call([sys.executable,str(Path(__file__)),'--worker'],ROOT/'worker.log',protocol['cpu_time_limit_seconds'])
    result=dict(complete=stage['status']=='complete',stage=stage,queries=len(queries),full_target_sequences=len(targets),
        candidate_pairs=count,queries_with_candidates=len(proposed),full_audit=False,training_eligibility='unknown',
        gpu_used=False,model_training=False)
    if result['complete']:
        rows=checked_hits(ROOT/'alignment/verified.tsv');found={row[0] for row in rows}
        unresolved=set(queries)-found
        (ROOT/'unresolved_query_ids.txt').write_text(''.join(q+'\n' for q in sorted(unresolved)))
        result.update(verified_positive_queries=len(found),unresolved_queries=len(unresolved),
            positive_fraction=len(found)/len(queries),verified_sha256=digest(ROOT/'alignment/verified.tsv'),
            unresolved_sha256=digest(ROOT/'unresolved_query_ids.txt'))
    write_json(ROOT/'results_report.json',result)
    write_json(ROOT/'state.json',dict(status='complete' if result['complete'] else 'incomplete'))
    (ROOT/'findings.md').write_text('# CPU witness precheck\n\n'+
        f'Execution complete: {result["complete"]}.\n'+
        'Every positive is independently realigned against the complete target DB.\n'+
        'Unresolved sequences remain unknown, not eligible. This is not a completed protection audit.\n')


if __name__=='__main__':
    if '--worker' in sys.argv:
        verify_candidates(dict(mmseqs=BINARY),ROOT/'alignment',SOURCE/'query.fa',SOURCE/'target.fa',ROOT/'candidates.tsv')
    else:main()
