"""Freeze checkpoints, annotate IMGT regions and structural contacts for analysis."""
import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
import sys
from affinity_study_common import digest,fingerprint,freeze,read_csv,read_json,write_csv,write_json
from affinity_ranking_pilot import verify


def main(config,settings_json):
    cfg=json.loads(settings_json);root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    reference=Path(cfg['reference_root']);old=verify(reference)
    rows=read_csv(reference/'records.csv');external=read_csv(reference/'external_selection.csv')
    pair=read_json(reference/'ranking_pair.json');pairids={pair[k]['id'] for k in ('strong','weak')}
    native=[r for r in rows if (r['split']=='test' and 'heavy' in r['example_target_chains'].split(';')) or r['id'] in pairids]
    if cfg['smoke']:
        native=[r for r in native if r['id'] in pairids]+[r for r in native if r['split']=='test'][:2]
        external=[r for t in ('HER2','PD1','TIGIT') for r in [x for x in external if x['target']==t][:8]]
    for row in native:
        candidates=[r for r in rows if r['split']=='test' and r['biological_group']!=row['biological_group'] and r['resolved_ag_seq']!=row['resolved_ag_seq']]
        donor=min(candidates,key=lambda r:(abs(len(row['resolved_ag_seq'])-len(r['resolved_ag_seq'])),r['id']))
        row['wrong_antigen_id']=donor['id'];row['wrong_antigen_sequence']=donor['resolved_ag_seq']
    checkpoints={}
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=reference/'runs'/f'{arm}_s{seed}';path=folder/'final.pt'
            if digest(path)!=read_json(folder/'complete.json')['checkpoint_sha256']:raise ValueError('Reference checkpoint changed')
            checkpoints[str(path)]=digest(path)
    sources=[config,str(reference/'training_protocol.json'),str(reference/'cohort_audit.json'),str(reference/'records.csv'),
             str(reference/'external_selection.csv'),str(reference/'ranking_pair.json'),str(reference/'embeddings_complete.json')]
    sources += [r['pdb_path'] for r in native]
    implementation=[str(Path(__file__)),'workflow/scripts/epitope_diagnostics_common.py',
                    'workflow/scripts/epitope_diagnostics_worker.py','workflow/scripts/report_epitope_diagnostics.py',
                    'workflow/epitope_diagnostics.smk']
    freeze(root/'protocol.json',dict(settings=cfg,reference_settings=old,checkpoints_sha256=checkpoints,
        inputs_sha256={p:digest(p) for p in sources},implementation_sha256={p:digest(p) for p in implementation},
        purpose='Inference-only test of antigen dependence, score-region dilution and epitope-localized sensitivity.',
        primary=['Affinity rho under correct versus absent antigen; rank agreement between these scores.',
                 'Affinity rho of CDR3 versus framework scores and wrong-minus-correct specificity residual.',
                 'RNase-A ranking gain under correct, wrong, mean, zero and absent antigen.',
                 'Antigen token occlusion sensitivity versus structural epitope fraction on held-out native complexes.'],
        contacts='Heavy-chain/antigen non-hydrogen atom distances, 5A primary; 4A/8A sensitivity. Contacts are not energetic hot spots.',
        causal_design='Occlude every one of the 32 memory positions in a fixed order, replacing it with the mean memory token. No structural label chooses an input, crop, mask or checkpoint.',
        attention='Descriptive cross-attention at positions predicting heavy residues, averaged over layers/heads. Causal claims require the independent token interventions.',
        limitations=['Existing test panels were previously inspected; all findings are exploratory.',
                     'One affinity-training parent; optimization-seed SD is not a biological confidence interval.',
                     'Post-encoder token interventions cannot remove contextualized epitope information elsewhere in ESM2 features.',
                     'No antigen-only generative workflow uses antibody-derived conditioning or oracle epitope input.']))
    write_csv(root/'native_records.csv',native);write_csv(root/'external_selection.csv',external)
    sequences=sorted({r['resolved_H_seq'] for r in native}|{r['heavy_sequence'] for r in external})
    from anarci import anarci
    from number_diagnostic_regions import region
    hmmscan=shutil.which('hmmscan') or str(Path(sys.executable).parent/'hmmscan');annotations={}
    for start in range(0,len(sequences),100):
        batch=[(fingerprint(s),s) for s in sequences[start:start+100]]
        numbered,details,_=anarci(batch,scheme='imgt',allow={'H'},hmmerpath=str(Path(hmmscan).parent),ncpu=4)
        for (key,sequence),domains,meta in zip(batch,numbered,details):
            labels=['unassigned']*len(sequence);valid=[]
            for domain,info in zip(domains or [],meta or []):
                if info['chain_type']!='H':continue
                numbering,begin,end=domain;residues=[(int(pos[0]),aa) for pos,aa in numbering if aa!='-']
                text=''.join(aa for pos,aa in residues)
                if sequence[int(begin):int(begin)+len(text)]!=text:raise ValueError('ANARCI sequence alignment mismatch')
                valid.append((int(begin),residues))
            if len(valid)==1:
                begin,residues=valid[0];labels[begin:begin+len(residues)]=[region(pos) for pos,aa in residues]
            annotations[key]=dict(sequence=sequence,labels=labels+['eos'],status='numbered' if len(valid)==1 else 'unassigned')
        print('Numbered',min(start+100,len(sequences)),'/',len(sequences),flush=True)
    for row in external:
        item=annotations[fingerprint(row['heavy_sequence'])]
        indices=[i for i,label in enumerate(item['labels'][:-1]) if label=='cdr3']
        if indices!=json.loads(row['cdr3_indices']):raise ValueError('New CDR3 annotation disagrees with frozen external annotation')
    write_json(root/'regions.json',dict(sequences=annotations,status_counts=dict(Counter(r['status'] for r in annotations.values()))))
    from epitope_diagnostics_common import contact_annotations
    geometry={};failures=[]
    for row in native:
        try:geometry[row['id']]=contact_annotations(row,cfg['contact_cutoffs'])
        except Exception as error:failures.append(dict(id=row['id'],reason=type(error).__name__+': '+str(error)))
    write_json(root/'contacts.json',dict(records=geometry,failures=failures))
    names=['native_records.csv','external_selection.csv','regions.json','contacts.json']
    write_json(root/'prepared.json',dict(complete=True,native_records=len(native),external_rows=len(external),
        numbered_sequences=len(annotations),geometry_records=len(geometry),geometry_failures=failures,
        files_sha256={n:digest(root/n) for n in names}))
    print('Prepared',len(native),'native examples;',len(geometry),'structure mappings;',len(failures),'mapping failures.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/epitope_diagnostics.yaml');p.add_argument('--settings-json',required=True)
    a=p.parse_args();main(a.config,a.settings_json)
