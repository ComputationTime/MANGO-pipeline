"""Freeze the factorial study; keep model inputs and label files separate."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_csv,write_json
from affinity_ranking_pilot import verify,native_schedule
from interface_supervision_common import antigen_input,contact_labels,graph


def main(config,settings):
    cfg=json.loads(settings);root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    ref=Path(cfg['reference_root']);old=verify(ref);binding=Path(old['binding_root'])
    rows=read_csv(ref/'records.csv');pair=read_json(ref/'ranking_pair.json');pairids={pair[k]['id'] for k in ('strong','weak')}
    if cfg['smoke']:
        rows=[r for r in rows if r['id'] in pairids]+[r for r in rows if r['split']=='train' and r['id'] not in pairids][:2]+[r for r in rows if r['split']=='test'][:2]
    agrows=read_csv(binding/'antigen_records.csv');structure_manifest=read_json(binding/'structures_manifest.json')
    if structure_manifest['antibody_inputs'] or structure_manifest['affinity_inputs']:raise ValueError('External structure input used antibody/affinity information')
    for row in agrows:
        marker=read_json(Path(row['pdb_path']).parent/'complete.json')
        if marker['structure_sha256']!=digest(row['pdb_path']) or marker['structure_manifest_sha256']!=digest(binding/'structures_manifest.json'):
            raise ValueError('Antigen-only prediction provenance changed')
    sources=[config,str(ref/'records.csv'),str(ref/'training_protocol.json'),str(ref/'cohort_audit.json'),str(ref/'ranking_pair.json'),
             str(ref/'external_selection.csv'),str(ref/'embeddings_complete.json'),str(binding/'structures_manifest.json'),
             str(Path(cfg['diagnostic_root'])/'regions.json')]+[r['pdb_path'] for r in rows+agrows]
    implementation=[str(Path(__file__)),'workflow/scripts/interface_supervision_common.py','workflow/scripts/interface_supervision_model.py',
                    'workflow/scripts/interface_supervision_worker.py','workflow/scripts/report_interface_supervision.py','workflow/interface_supervision.smk',
                    'workflow/scripts/affinity_study_common.py','workflow/scripts/affinity_ranking_pilot.py',
                    'workflow/scripts/controlled_antigen_model.py','workflow/scripts/iglm_baseline.py','workflow/scripts/chain_examples.py']
    freeze(root/'protocol.json',dict(settings=cfg,reference_settings=old,inputs_sha256={p:digest(p) for p in sources},
        implementation_sha256={p:digest(p) for p in implementation},
        primary='Held-out heavy-chain within-residue antigen-contact average precision, averaged within biological group and then across groups.',
        contrasts=['geometry effect without contact loss','contact-loss effect without geometry','geometry-by-contact-loss interaction'],
        secondary='Native heavy/CDR NLL, fixed-panel affinity Spearman from negative NLL, and geometry-permutation sensitivity.',
        representation='All MANGO arms keep one ESM2 token per antigen residue. Identical two-layer graph encoders use sequence neighbors or antigen CA-distance neighbors.',
        structural_input='Canonical antigen-only conformation per exact antigen sequence, chosen by record ID; no antibody-derived selection or coordinates.',
        supervision='Native complex contact maps are labels only; prediction of contacts uses states after consuming each generated residue.',
        probe_control='Both CE controls train the identical contact head on detached decoder/memory features. Only contact arms backpropagate this loss into the generator.',
        exposure='Native CE schedule plus identical RNase-A pair CE exposure; no affinity ranking term in this study.',
        uncertainty='Paired optimization-seed sample SD only; no biological CI or significance claim with this small, previously inspected study.',
        evaluation='External predicted antigen-alone Chai structures differ from experimental native structures; affinity panels were previously inspected.',
        limitations=['Full-residue memory is common to all four factorial arms; its effect relative to the earlier 32-token model is not isolated.',
                     '99 training records from 33 groups and 16 held-out groups are exploratory, not a powered generalization study.']))
    write_csv(root/'records.csv',rows);write_json(root/'ranking_pair.json',pair)
    ext=read_csv(ref/'external_selection.csv')
    if cfg['smoke']:ext=[r for t in ('HER2','PD1','TIGIT') for r in [x for x in ext if x['target']==t][:8]]
    write_csv(root/'external_selection.csv',ext);write_csv(root/'antigens.csv',agrows)
    for key in ('ab_ag_cluster','biological_group'):
        if {r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']=='test'}:raise ValueError('Group crosses splits')
    if any(r['source_split']!='train' for r in rows if r['split']=='train'):raise ValueError('Prior holdout entered training')
    byag=defaultdict(list)
    for row in rows+agrows:byag[fingerprint(row['resolved_ag_seq'])].append(row)
    (root/'inputs').mkdir(exist_ok=True);(root/'labels').mkdir(exist_ok=True);inputs={};labels={}
    for key,members in sorted(byag.items()):
        row=min(members,key=lambda r:r['id']);raw=antigen_input(row);payload={k:v for k,v in raw.items() if k!='ca_fallbacks'}
        for kind in ('sequence','geometry'):
            indices,edges,valid=graph(raw['coords'],raw['chains'],raw['positions'],kind,cfg['neighbors'])
            payload[kind+'_neighbors']=indices;payload[kind+'_edges']=edges;payload[kind+'_valid']=valid
        permuted=raw['coords'].copy();rng=np.random.default_rng(1729)
        for chain in np.unique(raw['chains']):
            idx=np.flatnonzero(raw['chains']==chain);permuted[idx]=permuted[rng.permutation(idx)]
        indices,edges,valid=graph(permuted,raw['chains'],raw['positions'],'geometry',cfg['neighbors'])
        payload.update(scrambled_neighbors=indices,scrambled_edges=edges,scrambled_valid=valid)
        path=root/'inputs'/f'{key}.npz';np.savez_compressed(path,**payload)
        inputs[key]=dict(path=str(path),sha256=digest(path),canonical_record=row['id'],residues=len(raw['coords']),ca_fallbacks=raw['ca_fallbacks'])
    for i,row in enumerate(rows,1):
        lab=contact_labels(row,cfg['contact_cutoff']);path=root/'labels'/f'{fingerprint(row["id"])}.npz';np.savez_compressed(path,**lab)
        labels[row['id']]=dict(path=str(path),sha256=digest(path),heavy_contacts=int(lab['heavy'].sum()),light_contacts=int(lab['light'].sum()))
        if i%25==0:print('Contact labels',i,'/',len(rows),flush=True)
    for seed in cfg['seeds']:
        for phase,updates in [('pretrain',cfg['pretraining_updates']),('adapt',cfg['adaptation_updates'])]:
            write_json(root/'schedules'/f'{phase}_{seed}.json',dict(items=native_schedule(rows,updates,cfg['native_batch_size'],seed+(0 if phase=='pretrain' else 100000))))
    names=['records.csv','ranking_pair.json','external_selection.csv','antigens.csv']+[str(p.relative_to(root)) for p in (root/'schedules').glob('*.json')]
    write_json(root/'prepared.json',dict(complete=True,inputs=inputs,labels=labels,records=len(rows),
        counts={s:dict(records=sum(r['split']==s for r in rows),groups=len({r['biological_group'] for r in rows if r['split']==s})) for s in ('train','test')},
        files_sha256={n:digest(root/n) for n in names},antibody_information_in_conditioning=False))
    print('Prepared',len(rows),'records;',len(inputs),'canonical antigen graphs.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--settings-json',required=True);a=p.parse_args();main(a.config,a.settings_json)
