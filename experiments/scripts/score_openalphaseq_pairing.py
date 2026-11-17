"""Stream completed IMGT domains through all frozen 16k models on one GPU."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch
import yaml

from antigen_diagnostics import atomic_json, mango_token_nll, iglm_token_nll
from device_common import get_device
from model_common import MangoModel, comparison_cohort, load_embedding
from iglm_baseline import IgLMScratch
from pairing_prefix_score_cache import SharedPrefixChainScorer
from prepare_asd_transfer import sha256


def target_id(chain,sequence):
    return hashlib.sha256(f'{chain}:{sequence}'.encode()).hexdigest()


def load_models(study,antigens,device):
    expected=comparison_cohort(study['training_records'])
    scorers,provenance={},{}
    for name,spec in study['models'].items():
        directory=Path(spec['directory'])
        marker=json.loads((directory/'train_complete.json').read_text())
        if marker['cohort_sha256']!=expected or marker['checkpoint_sha256']!=sha256(marker['checkpoint']):
            raise ValueError('Checkpoint does not match the protected 16k training cohort')
        checkpoint=torch.load(marker['checkpoint'],map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=0:
            raise ValueError('The pairing comparison requires one-epoch 16k checkpoints')
        if spec['family']=='iglm':
            metadata=checkpoint['metadata']
            if metadata.get('pretrained_weights_loaded') or metadata.get('antigen_conditioning'):
                raise ValueError('Expected the antigen-free scratch-trained IgLM control')
            model=IgLMScratch(dict(initialization='random_from_scratch',objective=metadata['objective'],
                                  species_token=metadata['species_token']),metadata['model_config'])
            model.lm.load_state_dict(checkpoint['lm']);model.lm.to(device).eval()
        else:
            model=MangoModel.from_checkpoint(checkpoint).to(device).eval()
        scorers[name]=SharedPrefixChainScorer(model,spec['family'],antigens)
        provenance[name]=dict(checkpoint=marker['checkpoint'],checkpoint_sha256=marker['checkpoint_sha256'],
                              cohort_sha256=expected,architecture=checkpoint.get('architecture',spec['family']))
        print('Loaded',name,flush=True)
    return scorers,provenance


def score(config,limit=None,upgrade_score_cache=False):
    study=yaml.safe_load(Path(config).read_text());root=Path(study['experiment_root'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device=get_device('OpenAlphaSeq 16k all-antigen pairing scores')
    antigen_rows=list(csv.DictReader((root/'antigens.csv').open()))
    antigen_rows.sort(key=lambda r:r['id'])
    antigen_ids=[r['id'] for r in antigen_rows]
    store=root/'openalphaseq/embeddings/antigen/esm2'
    antigens={r['id']:load_embedding(str(store/r['split']/f'{r["id"]}.pt')).to(device) for r in antigen_rows}
    scorers,provenance=load_models(study,antigens,device)
    inputs=dict(configuration=study,checkpoints=provenance,antigens_sha256=sha256(root/'antigens.csv'),
                raw_manifest_sha256=sha256(root/'raw_manifest.json'),antigen_order=antigen_ids,
                scoring_implementation={p:sha256(p) for p in [__file__,'experiments/scripts/pairing_score_cache.py',
                                                            'experiments/scripts/pairing_prefix_score_cache.py']},
                precision='float32_highest',protocol='shared_target_prefix_teacher_forced_residues_and_eos_v2')
    manifest_path=root/'scoring_manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text())!=inputs:
        previous=json.loads(manifest_path.read_text())
        old_implementation=previous['scoring_implementation']
        # The original cache has already passed scalar-production parity. This
        # narrowly scoped migration changes computation reuse, preserving the
        # checkpoint, sequences, precision, target objective and saved scores.
        expected_legacy='d50f40ab2f14e9ef0d0410dac4e414f9251aaeef628c8c2a917fbfa29318707c'
        comparable=lambda d:{k:v for k,v in d.items() if k not in ('protocol','scoring_implementation')}
        if (not upgrade_score_cache or comparable(previous)!=comparable(inputs) or
            previous['protocol']!='cached_teacher_forced_residues_and_eos_v1' or
            old_implementation.get(str(Path(__file__).resolve()))!=expected_legacy or
            old_implementation.get('experiments/scripts/pairing_score_cache.py')!=sha256('experiments/scripts/pairing_score_cache.py')):
            raise ValueError('Scoring inputs/implementation changed; use a new score cache')
        atomic_json(root/'scoring_manifest_v1.json',previous)
        atomic_json(root/'scoring_cache_upgrade.json',dict(
            previous_manifest_sha256=sha256(manifest_path),
            protocol_from=previous['protocol'],protocol_to=inputs['protocol'],
            reason='Reuse identical teacher-forced target prefixes within sequence-sorted batches; no biological input change',
            preserved_scores=True,precision=inputs['precision'],
            validation='All model/chain paths are rechecked against scalar production scores before writing new batches'))
    atomic_json(manifest_path,inputs)
    db=sqlite3.connect(root/'scores.sqlite',timeout=60)
    db.executescript('''PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS targets (id TEXT PRIMARY KEY,chain TEXT,sequence TEXT,cdr3_indices TEXT,ntokens INTEGER,ncdr3 INTEGER);
    CREATE TABLE IF NOT EXISTS scores (model TEXT,target_id TEXT,loss_sum BLOB,cdr3_sum BLOB,PRIMARY KEY(model,target_id));
    CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY,value TEXT);''')
    known={name:{r[0] for r in db.execute('SELECT target_id FROM scores WHERE model=?',(name,))} for name in scorers}
    old=db.execute("SELECT value FROM progress WHERE key='cursor'").fetchone();cursor=int(old[0]) if old else 0
    started=time.monotonic();last_data=time.monotonic();parity_done=False;new_targets=0
    while True:
        domains_path=root/'domains.sqlite'
        if not domains_path.is_file():
            time.sleep(2);continue
        with sqlite3.connect(f'file:{domains_path.resolve()}?mode=ro',uri=True,timeout=60) as numberdb:
            rows=numberdb.execute('SELECT rowid,component_id,sequence,status,chain,target,cdr3_indices FROM domains WHERE rowid>? ORDER BY rowid LIMIT 5000',(cursor,)).fetchall()
        if not rows:
            if (root/'numbering_complete.json').is_file():break
            if time.monotonic()-last_data>600:raise RuntimeError('ANARCI made no progress for ten minutes; inspect numbering.log')
            time.sleep(2);continue
        last_data=time.monotonic()
        targets={}
        for rowid,key,raw,status,chain,sequence,cdr3 in rows:
            if status!='valid' or len(raw)>170 or not (60<=len(sequence)<=170):continue
            tid=target_id(chain,sequence)
            targets[tid]=(chain,sequence,json.loads(cdr3))
        db.executemany('INSERT OR IGNORE INTO targets VALUES (?,?,?,?,?,?)',
            [(tid,c,s,json.dumps(ind),len(s)+1,len(ind)) for tid,(c,s,ind) in targets.items()])
        for chain in ('heavy','light'):
            items=sorted([(key,*value) for key,value in targets.items() if value[0]==chain],key=lambda v:(v[2],v[0]))
            for name,scorer in scorers.items():
                missing=[v for v in items if v[0] not in known[name]]
                for begin in range(0,len(missing),study['batch_size']):
                    batch=missing[begin:begin+study['batch_size']]
                    sequences=[v[2] for v in batch];cdrs=[v[3] for v in batch]
                    encoded=scorer.tokens(sequences)
                    loss=np.zeros((len(batch),len(antigen_ids)),dtype=np.float64);cdr=np.zeros_like(loss)
                    active=antigen_ids[:1] if scorer.family=='iglm' else antigen_ids
                    for j,antigen_id in enumerate(active):
                        total,cdr_total,_=scorer.score(antigen_id,chain,sequences,cdrs,encoded=encoded)
                        loss[:,j]=total.cpu().numpy();cdr[:,j]=cdr_total.cpu().numpy()
                    if scorer.family=='iglm':loss[:,:]=loss[:,:1];cdr[:,:]=cdr[:,:1]
                    if not np.isfinite(loss).all() or not np.isfinite(cdr).all():raise ValueError('Non-finite pairing score')
                    if not parity_done and begin==0 and missing:
                        maximum=0.
                        for j,aid in enumerate(active):
                            for i in range(min(2,len(batch))):
                                with torch.no_grad():
                                    reference=(iglm_token_nll(scorer.model,chain,sequences[i]) if scorer.family=='iglm' else
                                               mango_token_nll(scorer.model,antigens[aid],chain,sequences[i]))
                                delta=abs(loss[i,j]/(len(sequences[i])+1)-reference.double().mean().item())
                                maximum=max(maximum,delta)
                        if maximum>3e-6:raise ValueError(f'Cached scorer differs from production NLL: {name} {maximum}')
                        print('Production-score parity',name,chain,maximum,flush=True)
                    db.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?)',
                        [(name,v[0],loss[i].tobytes(),cdr[i].tobytes()) for i,v in enumerate(batch)])
                    db.commit();known[name].update(v[0] for v in batch)
        parity_done=True
        cursor=rows[-1][0]
        db.execute("INSERT OR REPLACE INTO progress VALUES ('cursor',?)",(str(cursor),));db.commit()
        new_targets+=len(targets)
        progress=dict(complete=False,annotated_components_consumed=cursor,
            unique_scored_targets={k:len(v) for k,v in known.items()},wall_seconds=time.monotonic()-started,
            total_new_targets_this_invocation=new_targets,antigens=len(antigen_ids))
        atomic_json(root/'scoring_progress.json',progress)
        print(json.dumps(progress),flush=True)
        if limit and new_targets>=limit:
            print('Stopped after requested benchmark limit',flush=True);db.close();return progress
    count=db.execute('SELECT COUNT(*) FROM targets').fetchone()[0]
    if any(len(v)!=count for v in known.values()):raise ValueError('Models scored different target sets')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    ready=dict(complete=True,targets=count,models=list(scorers),antigens=len(antigen_ids),
               scoring_manifest_sha256=sha256(manifest_path),wall_seconds=time.monotonic()-started)
    atomic_json(root/'scoring_complete.json',ready);print(json.dumps(ready,indent=2),flush=True)
    return ready


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='experiments/config/openalphaseq_pairing.yaml');p.add_argument('--limit',type=int)
    p.add_argument('--upgrade-score-cache',action='store_true',help='Migrate the verified v1 cache to shared target-prefix computation')
    score(**vars(p.parse_args()))
