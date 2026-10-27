"""Extract IMGT VH/VL targets from CATH-eligible AbRank records using ANARCI.

Run in workflow/envs/analysis_anarci.yaml. No antibody-derived value is used
for antigen embedding or biological conditioning.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil

from prepare_cath_sequences import clean_sequence, sequence_id
from prepare_abrank_transfer import cath_annotations


def number(abrank, sabdab, cohort, reference, output, threads=8):
    from anarci import anarci
    cohort=Path(cohort)
    families,confident,_,native=cath_annotations(cohort,Path(reference))
    protected=list(csv.DictReader(open(sabdab)))
    blocked=set()
    for row in protected:
        if row['split']!='test': continue
        blocked.update(families['sabdab2',row['id']])
        for chain in row['antigen_chains'].split(','):
            if len(chain)==1:
                blocked.update(native[row['pdb_id'][-4:].lower()+chain])
    sequences={}
    count=0
    for i,row in enumerate(csv.DictReader(open(abrank),delimiter='\t')):
        key=('abrank',f'abrank_{i:07d}')
        if 'AG' not in confident[key] or families[key]&blocked: continue
        count+=1
        for column,chain in [('Ab_heavy_chain_seq','heavy'),('Ab_light_chain_seq','light')]:
            try: seq=clean_sequence(row[column])
            except ValueError: continue
            sid=sequence_id(seq)
            if sid in sequences and sequences[sid]!=(seq,chain):
                raise ValueError('Same sequence has incompatible chain annotations')
            sequences[sid]=(seq,chain)
    print(f'{count} CATH-eligible records; {len(sequences)} distinct antibody chains',flush=True)
    items=sorted(sequences.items());result={}
    executable=shutil.which('hmmscan')
    if not executable: raise RuntimeError('ANARCI environment must provide hmmscan')
    for start in range(0,len(items),100):
        batch=items[start:start+100]
        numbered,details,_=anarci([(key,s[0]) for key,s in batch],scheme='imgt',
            allow={'H','K','L'},hmmerpath=str(Path(executable).parent),ncpu=threads,
            allowed_species=['human','mouse','rat','rabbit','rhesus','pig','alpaca'])
        for (key,(seq,expected)),domains,info in zip(batch,numbered,details):
            matches=[]
            for domain,detail in zip(domains or [],info or []):
                kind=detail['chain_type']
                if (expected=='heavy' and kind=='H') or (expected=='light' and kind in {'K','L'}):
                    numbering,begin,end=domain
                    target=''.join(aa for _,aa in numbering if aa!='-')
                    matches.append(dict(sequence=target,chain=expected,anarci_type=kind,start=begin,end=end))
            result[key]=dict(input_sequence=seq,expected_chain=expected,domains=matches,
                             status='valid' if len(matches)==1 else 'unassigned_or_multiple')
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
        print(f'Numbered {min(start+100,len(items))}/{len(items)}',flush=True)
    manifest=dict(abrank_sha256=hashlib.sha256(Path(abrank).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                  scheme='imgt',allowed_species=['human','mouse','rat','rabbit','rhesus','pig','alpaca'],
                  counts=dict(collections.Counter(r['status'] for r in result.values())))
    Path(output).with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['abrank','sabdab','cohort','reference','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--threads',type=int,default=8)
    number(**vars(parser.parse_args()))
