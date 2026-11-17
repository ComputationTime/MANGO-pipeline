"""Frozen short-test arms, bounded sampling and within-assay affinity curation."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import defaultdict
import math
import random

from affinity_study_common import fingerprint,row_key
from chain_examples import example_chains

ARMS={
 'U_native':dict(branch=False,shuffled=False,objective='ce',pool='mean',test=3,native=True),
 'U':dict(branch=False,shuffled=False,objective='ce',pool='mean',test=3),
 'C':dict(branch=True,shuffled=False,objective='ce',pool='mean',test=3),
 'S':dict(branch=True,shuffled=True,objective='ce',pool='mean',test=3),
 'P':dict(branch=True,shuffled=False,objective='pairing',pool='mean',test=3),
 'PS':dict(branch=True,shuffled=True,objective='pairing',pool='mean',test=3),
 'C_CDR':dict(branch=True,shuffled=False,objective='cdr',pool='mean',test=4),
 'U_CDR':dict(branch=False,shuffled=False,objective='cdr',pool='mean',test=4),
 'L':dict(branch=True,shuffled=False,objective='ce',pool='learned',test=5),
 'LS':dict(branch=True,shuffled=True,objective='ce',pool='learned',test=5),
 'U_L':dict(branch=False,shuffled=False,objective='ce',pool='learned',test=5),
 'C_Rank':dict(branch=True,shuffled=False,objective='ranking',pool='mean',test=6),
 'U_Rank':dict(branch=False,shuffled=False,objective='ranking',pool='mean',test=6),
 'C_AffCE':dict(branch=True,shuffled=False,objective='affinity_ce',pool='mean',test=6),
 'U_AffCE':dict(branch=False,shuffled=False,objective='affinity_ce',pool='mean',test=6)}


def balanced_select(rows,limit):
    """Round-robin components, then antigen identities, without duplicating rows."""
    bins=defaultdict(lambda:defaultdict(list))
    for row in sorted(rows,key=row_key):bins[row['biological_group']][row['resolved_ag_seq']].append(row)
    groups=sorted(bins,key=fingerprint);chosen=[];cursor=defaultdict(int)
    while groups and len(chosen)<limit:
        for group in list(groups):
            antigens=sorted(bins[group],key=fingerprint)
            if not antigens:groups.remove(group);continue
            antigen=antigens[cursor[group]%len(antigens)];cursor[group]+=1
            chosen.append(bins[group][antigen].pop(0))
            if not bins[group][antigen]:del bins[group][antigen]
            if len(chosen)==limit:break
    return chosen


def schedule(rows,updates,seed):
    from asd_scaling_studies import maximal_record_shuffle
    training=[r for r in rows if r['split']=='train'];by_group=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    known=defaultdict(set)
    for row in training:
        for chain in example_chains(row):
            by_group[chain][row['biological_group']][row['resolved_ag_seq']].append(row)
            known[chain,row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq']].add(row['resolved_ag_seq'])
    rng=random.Random(seed);chains=sorted(by_group);groups={c:sorted(by_group[c]) for c in chains};items=[];uses=defaultdict(int)
    for index in range(updates):
        chain=chains[index%len(chains)];turn=uses[chain];uses[chain]+=1
        if turn%len(groups[chain])==0:rng.shuffle(groups[chain])
        group=groups[chain][turn%len(groups[chain])];bucket=by_group[chain][group]
        antigen=sorted(bucket)[(turn//len(groups[chain]))%len(bucket)];row=rng.choice(bucket[antigen])
        items.append(dict(step=index,id=row['id'],chain=chain,resolved_ag_seq=row['resolved_ag_seq'],split='train'))
    by_id={r['id']:r for r in training};donors={}
    for chain in ('heavy','light'):
        members=[dict(x,id=str(x['step']),record_id=x['id']) for x in items if x['chain']==chain]
        if not members:continue
        donors.update({int(k):v['record_id'] for k,v in maximal_record_shuffle(members,seed).items()})
    for item in items:
        row=by_id[item['id']];item['shuffled_id']=donors[item['step']]
        if row['resolved_ag_seq']==by_id[item['shuffled_id']]['resolved_ag_seq']:raise ValueError('Schedule cannot be completely deranged')
        excluded=known[item['chain'],row['resolved_H_seq' if item['chain']=='heavy' else 'resolved_L_seq']]
        for kind,positive in (('correct',row),('shuffled',by_id[item['shuffled_id']])):
            candidates=[r for r in training if r['resolved_ag_seq']!=positive['resolved_ag_seq'] and r['resolved_ag_seq'] not in excluded]
            if not candidates:raise ValueError('No eligible putative negative antigen')
            item['decoy_'+kind]=rng.choice(candidates)['id']
        del item['resolved_ag_seq'];del item['split']
    return items


def affinity_pairs(measurements,records):
    """Conservative matched Kd preferences; refuse ambiguous assay provenance."""
    by_id={r['id']:r for r in records};bins=defaultdict(list);retained=[]
    required={'id','target_id','parent_id','assay_id','assay_provenance_url','measurement_type',
              'construct_format','operator','log10_kd_nm','log10_kd_sd','replicate_count'}
    for value in measurements:
        if not required<=set(value):raise ValueError('Affinity table lacks assay/lineage/uncertainty provenance')
        if value['id'] not in by_id:continue
        if value['measurement_type']!='Kd' or value['operator']!='=':continue
        if not value['assay_provenance_url'].startswith('https://') or any(not value[k] for k in ('target_id','parent_id','assay_id','construct_format')):raise ValueError('Invalid affinity provenance')
        mean=float(value['log10_kd_nm']);sd=float(value['log10_kd_sd'])
        if not math.isfinite(mean) or not math.isfinite(sd) or sd<0:raise ValueError('Invalid affinity uncertainty')
        if int(value['replicate_count'])<2:continue
        r=by_id[value['id']]
        if 'heavy' not in example_chains(r):continue
        item=dict(value,split=r['split'],biological_group=r['biological_group'])
        retained.append(item)
        key=(r['split'],r['resolved_ag_seq'],value['target_id'],value['parent_id'],value['assay_id'],value['assay_provenance_url'],value['construct_format'])
        bins[key].append(item)
    if len({r['id'] for r in retained})!=len(retained):raise ValueError('Multiple measurements per canonical record require curator aggregation')
    pairs=[]
    for key,members in sorted(bins.items()):
        members.sort(key=lambda r:(float(r['log10_kd_nm']),r['id']))
        for strong in members:
            weak=next((w for w in members if float(w['log10_kd_nm'])-float(strong['log10_kd_nm'])>=1.
                and float(w['log10_kd_nm'])-float(strong['log10_kd_nm'])>2*(float(w['log10_kd_sd'])+float(strong['log10_kd_sd']))),None)
            if weak:pairs.append(dict(split=key[0],strong=strong['id'],weak=weak['id'],assay_id=key[4]))
    return retained,pairs
