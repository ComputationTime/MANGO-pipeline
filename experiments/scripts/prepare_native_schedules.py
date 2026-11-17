"""Freeze target schedules and label-blind affinity selection before fitting."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from pathlib import Path
import sqlite3
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_csv,write_json
from short_affinity_common import schedule


def main(root):
    root=Path(root);cfg=read_json(root/'training_protocol.json')['settings'];rows=read_csv(root/'records.csv')
    for seed in cfg['seeds']:
        # Derange separately within each stage, preserving its exact marginals.
        items=schedule(rows,cfg['pretraining_updates'],seed)+schedule(rows,cfg['adaptation_updates'],seed+100000)
        for index,item in enumerate(items):item['step']=index
        freeze(root/'schedules'/f'{seed}.json',dict(records_sha256=digest(root/'records.csv'),items=items))
    binding=Path(cfg['binding_root']);queries={r['id']:r for r in read_csv(binding/'heldout_antibodies.csv')}
    targets={r['id']:r['family'] for r in read_csv(binding/'antigens.csv')}
    with sqlite3.connect(f'file:{(binding/"measurements.sqlite").resolve()}?mode=ro',uri=True) as db:
        pairs=db.execute('SELECT query_id,antigen_id,reliable_sum/reliable_count FROM pairs WHERE reliable_count>0').fetchall()
    selected=[]
    for aid,target in targets.items():
        subset=sorted((p for p in pairs if p[1]==aid),key=lambda p:fingerprint(p[0]))[:cfg['diagnostic_pairs_per_target']]
        selected.extend(dict(query_id=q,antigen_id=a,target=target,affinity=v,heavy_sequence=queries[q]['heavy_sequence']) for q,a,v in subset)
    write_csv(root/'affinity_selection.csv',selected)
    write_json(root/'schedules_complete.json',dict(complete=True,selection='query hash without affinity thresholds',
        records_sha256=digest(root/'records.csv'),selection_sha256=digest(root/'affinity_selection.csv'),
        schedules_sha256={str(s):digest(root/'schedules'/f'{s}.json') for s in cfg['seeds']},
        source_sha256={n:digest(binding/n) for n in ('heldout_antibodies.csv','antigens.csv','measurements.sqlite')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(**vars(p.parse_args()))
