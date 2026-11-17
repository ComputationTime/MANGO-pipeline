"""Small-target candidate search with full-database revalidation and fallback.

Candidate witnesses are never themselves an exclusion decision. Their final
alignment/E-value uses the complete target database. Unresolved queries still
receive the existing full search. Full lineage graphs bypass this shortcut.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import Counter, defaultdict
import csv
import heapq
from pathlib import Path
import subprocess

from affinity_study_common import digest, fingerprint, freeze, write_json
from prepare_affinity_study import fasta
from accelerated_affinity_search import search as full_search


def representatives(queries, targets, limit=512):
    if limit<1:raise ValueError('Representative count must be positive')
    def bucket(seq):return (len(seq)//16,seq[:6],seq[-4:])
    counts=Counter(map(bucket,queries));groups=defaultdict(list)
    for seq in set(targets)-{''}:groups[bucket(seq)].append(seq)
    keys=sorted(groups,key=lambda k:(-counts[k],k))
    for key in keys:groups[key].sort(key=fingerprint)
    selected=[];index=0;preferred=[k for k in keys if counts[k]]
    preferred_limit=min(3*limit//4,sum(len(groups[k]) for k in preferred))
    while len(selected)<preferred_limit:
        for key in preferred:
            if index<len(groups[key]):selected.append(groups[key][index])
            if len(selected)==preferred_limit:break
        index+=1
    selected=set(selected)
    selected.update(heapq.nsmallest(limit-len(selected),set(targets)-selected-{''},key=fingerprint))
    return selected


def lookup(path):
    values={}
    with Path(path).open() as handle:
        for row in csv.reader(handle,delimiter='\t'):
            if len(row)!=3 or row[1] in values:raise ValueError('Invalid or duplicate MMseqs lookup')
            values[row[1]]=int(row[0])
    return values


def prefilter_rows(candidates, queries, targets):
    pairs=set()
    for row in candidates:
        if row[0] not in queries or row[1] not in targets:raise ValueError('Candidate identifier outside the full databases')
        pairs.add((queries[row[0]],targets[row[1]]))
    # Protein Smith-Waterman computes the actual score; these are candidate IDs.
    return [f'{q}\t{t}\t0\t0\n' for q,t in sorted(pairs)]


def checked_hits(path):
    with Path(path).open() as handle:rows=list(csv.reader(handle,delimiter='\t'))
    for row in rows:
        if len(row)!=6 or float(row[2])<.9 or min(float(row[3]),float(row[4]))<.8 or float(row[5])>.001:
            raise ValueError('Revalidated witness fails a frozen protection threshold')
    return rows


def verify_candidates(settings, directory, query_fasta, target_fasta, candidate_file):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    freeze(directory/'verification_manifest.json',dict(query_sha256=digest(query_fasta),
        full_target_sha256=digest(target_fasta),candidates_sha256=digest(candidate_file),
        binary_sha256=digest(settings['mmseqs']),evalue_database='complete target FASTA, not representative subset'))
    with Path(candidate_file).open() as handle:candidates=list(csv.reader(handle,delimiter='\t'))
    output=directory/'verified.tsv'
    if not candidates:output.write_text('');return []
    binary=settings['mmseqs'];qdb=directory/'query_db';tdb=directory/'full_target_db'
    with (directory/'verify.log').open('w') as log:
        def run(args):subprocess.run([binary,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
        run(['createdb',query_fasta,qdb,'--shuffle','0'])
        run(['createdb',target_fasta,tdb,'--shuffle','0'])
        query_ids=lookup(str(qdb)+'.lookup');target_ids=lookup(str(tdb)+'.lookup')
        prefilter=directory/'candidate_pairs.tsv'
        prefilter.write_text(''.join(prefilter_rows(candidates,query_ids,target_ids)))
        run(['tsv2db',prefilter,directory/'candidate_db','--output-dbtype','7'])
        run(['align',qdb,tdb,directory/'candidate_db',directory/'alignment_db',
            '--min-seq-id','0.9','-c','0.8','--cov-mode','0','-e','0.001',
            '--threads','4','--alignment-mode','3'])
        run(['convertalis',qdb,tdb,directory/'alignment_db',output,'--threads','4',
            '--format-output','query,target,fident,qcov,tcov,evalue'])
    verified=checked_hits(output)
    supplied={(row[0],row[1]) for row in candidates}
    if any((row[0],row[1]) not in supplied for row in verified):raise ValueError('Unexpected revalidation alignment')
    freeze(directory/'complete.json',dict(verified_queries=len({row[0] for row in verified}),
        verified_sha256=digest(output),manifest_sha256=digest(directory/'verification_manifest.json')))
    return verified


def search(settings,root,name,queries,targets,antibody):
    queries,targets=set(queries)-{''},set(targets)-{''}
    if not antibody or name=='all_lineages' or not queries or not targets:
        return full_search(settings,root,name,queries,targets,antibody)
    directory=root/'searches'/name;directory.mkdir(parents=True,exist_ok=True)
    fasta(directory/'query.fa',queries);fasta(directory/'target.fa',targets)
    chosen=representatives(queries,targets,settings.get('witness_targets',512))
    freeze(directory/'tier_manifest.json',dict(query_sha256=digest(directory/'query.fa'),
        target_sha256=digest(directory/'target.fa'),representative_ids=sorted(map(fingerprint,chosen)),
        policy='Only full-database revalidated witnesses remove a query; every unresolved query receives the original full search.'))
    witness_name=name+'_witness'
    full_search(settings,root,witness_name,queries,chosen,True)
    verified=verify_candidates(settings,directory/'revalidation',directory/'query.fa',directory/'target.fa',
        root/'searches'/witness_name/'hits.tsv')
    found={row[0] for row in verified};remaining={q for q in queries if fingerprint(q) not in found}
    result=set(found);combined=list(verified)
    if remaining:
        suffix=name+'_remaining';result|=full_search(settings,root,suffix,remaining,targets,True)
        combined.extend(checked_hits(root/'searches'/suffix/'hits.tsv'))
    with (directory/'hits.tsv').open('w') as handle:
        csv.writer(handle,delimiter='\t',lineterminator='\n').writerows(combined)
    freeze(directory/'complete.json',dict(query_count=len(queries),representative_targets=len(chosen),
        full_database_verified_positive_queries=len(found),unresolved_queries_sent_to_full_search=len(remaining),
        positive_queries=len(result),hits_sha256=digest(directory/'hits.tsv')))
    return result
