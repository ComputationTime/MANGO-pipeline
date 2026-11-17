"""Check fixed-light controls and sequence variation in the frozen affinity panel."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_json


def main(root):
    root=Path(root);p=read_json(root/'protocol.json');binding=Path(p['reference_settings']['binding_root'])
    sources=[root/'external_selection.csv',root/'regions.json',binding/'cohort.sqlite',Path(__file__)]
    freeze(root/'benchmark_audit_protocol.json',dict(scope='Metadata/sequence audit only; no subset selection or fitting.',
           inputs_sha256={str(s):digest(s) for s in sources}))
    selected=read_csv(root/'external_selection.csv');regions=read_json(root/'regions.json')['sequences'];bins=defaultdict(list)
    with sqlite3.connect('file:'+str((binding/'cohort.sqlite').resolve())+'?mode=ro',uri=True) as db:
        for row in selected:
            h,l=db.execute('SELECT heavy,light FROM queries WHERE id=?',(row['query_id'],)).fetchone()
            expected=db.execute('SELECT sequence FROM targets WHERE id=?',(h,)).fetchone()[0]
            if row['heavy_sequence']!=expected:raise ValueError('Selected sequence disagrees with canonical heavy chain')
            bins[row['target']].append(dict(row,light_id=l))
    result={}
    for target,rows in bins.items():
        unique={region:set() for region in ('framework','cdr1','cdr2','cdr3')};lengths=Counter()
        for row in rows:
            seq=row['heavy_sequence'];labels=regions[fingerprint(seq)]['labels'][:-1];lengths[len(seq)]+=1
            for region in unique:unique[region].add(''.join(a for a,label in zip(seq,labels) if label==region))
        counts=Counter(r['light_id'] for r in rows)
        result[target]=dict(rows=len(rows),unique_heavy=len({r['heavy_sequence'] for r in rows}),
                           unique_light=len(counts),most_common_light_rows=counts.most_common(1)[0][1],
                           heavy_lengths=dict(lengths),unique_region_sequences={k:len(v) for k,v in unique.items()})
    write_json(root/'benchmark_sequence_audit.json',dict(complete=True,targets=result,
        caution='Regional autoregressive NLL can vary after upstream CDR changes even if the scored framework sequence is identical. A regional score is not a local energetic measurement.'))
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/epitope_diagnostics_v1');a=p.parse_args();main(a.root)
