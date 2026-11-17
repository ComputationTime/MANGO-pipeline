"""Read-only bounds on additional native data, before any homology search."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path

from affinity_study_common import digest, freeze, write_csv
from prepare_abrank_transfer import UnionFind


def main():
    root=Path('artifacts/experiments/native_cath_inventory_v1');root.mkdir(parents=True,exist_ok=True)
    source=Path('artifacts/data/sabdab2_v0.1.0/standardized/standardized.csv')
    prior=[Path('artifacts/data/sabdab2_v0.1.0/processed/records.csv'),
        Path('artifacts/experiments/asd_cath_transfer_v1/cohort/sabdab2_records.csv')]
    daily=Path('artifacts/data/cath/reference/cath-b-newest-all.gz')
    rows=list(csv.DictReader(source.open()));by_id={r['id']:r for r in rows}
    protected={r['ab_ag_cluster'] for r in rows if r['split']!='train'}
    for path in prior:
        for row in csv.DictReader(path.open()):
            if row['split']!='train' and row['id'] in by_id:protected.add(by_id[row['id']]['ab_ag_cluster'])
    families=defaultdict(set)
    with gzip.open(daily,'rt') as handle:
        for line in handle:
            fields=line.split()
            if len(fields)>=3:families[fields[0][:5]].add(fields[2])
    mapped=[];counts=Counter();uf=UnionFind();aa=set('ACDEFGHIKLMNPQRSTVWY')
    for row in rows:
        heavy=row['resolved_H_seq'];ag=row['resolved_ag_seq'];chains=row['antigen_chains'].split(',')
        if not heavy or not set(heavy)<=aa or not ag or any(not s or not set(s)<=aa for s in ag.split(',')):
            counts['invalid_or_noncanonical_sequence']+=1;continue
        if not all('protein' in t.lower() for t in row['antigen_types'].split(',')):
            counts['nonprotein_antigen']+=1;continue
        if len(ag)>2048 or not Path(row['pdb_path']).is_file():
            counts['length_or_missing_structure']+=1;continue
        if len(chains)!=len(ag.split(',')):
            counts['chain_count_mismatch']+=1;continue
        pdb=row['pdb_id'][-4:].lower()
        assigned=[families[pdb+chain] if len(chain)==1 else set() for chain in chains]
        if not all(assigned):counts['incomplete_direct_cath_annotation']+=1;continue
        union=set().union(*assigned);key='row:'+row['id']
        uf.union(key,'source:'+row['ab_ag_cluster'])
        uf.union(key,'antigen:'+ag)
        for family in union:uf.union(key,'family:'+family)
        for field in ('resolved_H_seq','resolved_L_seq'):
            if row[field]:uf.union(key,'antibody:'+row[field])
        mapped.append(dict(id=row['id'],source_group=row['ab_ag_cluster'],
            source_split=row['split'],protected_group=row['ab_ag_cluster'] in protected,
            cath_families=';'.join(sorted(union)),antigen_length=len(ag),heavy_length=len(heavy)))
    groups=defaultdict(list)
    for row in mapped:groups[uf.find('row:'+row['id'])].append(row)
    summaries=Counter();membership=[];sizes=[]
    for index,group in enumerate(sorted(groups.values(),key=lambda g:min(r['id'] for r in g))):
        reserved=any(r['protected_group'] for r in group);label='reserved' if reserved else 'candidate_train'
        summaries[label+'_rows']+=len(group);summaries[label+'_components']+=1;sizes.append(len(group))
        for row in group:membership.append(dict(row,component=index,component_reserved=reserved))
    write_csv(root/'membership.csv',membership)
    result=dict(source_sha256={str(p):digest(p) for p in [source,daily,*prior,Path(__file__)]},
        original_rows=len(rows),protected_source_groups=len(protected),excluded=dict(counts),
        direct_cath_eligible_rows=len(mapped),bounds=dict(summaries),component_sizes=sorted(sizes,reverse=True),
        membership_sha256=digest(root/'membership.csv'),model_training_launched=False,
        limitations=['Inventory bounds only; no new split assignments.',
            'Direct PDB-chain CATH annotations only; unassigned chains quarantined.',
            'External antigen/antibody homology and non-exact antibody links not yet screened.',
            'Components can only merge or lose records after the remaining protection audit; counts are upper bounds.'])
    freeze(root/'results_report.json',result)
    print({k:result[k] for k in ('original_rows','direct_cath_eligible_rows','excluded','bounds')})


if __name__=='__main__':main()
