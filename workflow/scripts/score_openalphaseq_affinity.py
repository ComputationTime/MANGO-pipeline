"""Score only measured, training-disjoint AlphaSeq chain/antigen pairs, resumably."""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from antigen_diagnostics import iglm_token_nll, mango_token_nll
from device_common import get_device
from finalize_openalphaseq_pairing import digest, write_json
from iglm_baseline import IgLMScratch
from model_common import MangoModel, comparison_cohort, load_embedding
from pairing_prefix_score_cache import SharedPrefixChainScorer


def main(study_path, model_name, limit_batches=None):
    study = json.loads(Path(study_path).read_text())
    root = Path(study['experiment_root']); spec = study['models'][model_name]
    directory = Path(spec['directory']); checkpoint_path = directory / 'checkpoints/best.pt'
    expected = comparison_cohort(study['training_records'])
    evaluation = json.loads((directory / 'eval.json').read_text())
    if evaluation['cohort_sha256'] != expected:
        raise ValueError('Model evaluation uses a different training cohort')
    audit = json.loads((root / 'cohort_audit.json').read_text())
    if audit['training_records_sha256'] != digest(study['training_records']):
        raise ValueError('The AlphaSeq novelty audit uses different training records')
    complete = json.loads((root / 'measurements_complete.json').read_text())
    if (complete['cohort_sha256'] != digest(root / 'cohort.sqlite') or
            complete['measurements_sha256'] != digest(root / 'measurements.sqlite')):
        raise ValueError('The frozen AlphaSeq cohort or measurements changed')
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    device = get_device(f'AlphaSeq measured-pair scoring: {model_name}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata'] if spec['family'] == 'iglm' else checkpoint
    if metadata['cohort_sha256'] != expected or checkpoint['epoch'] != evaluation['checkpoint_epoch']:
        raise ValueError('Checkpoint/evaluation cohort or selected epoch differs')
    if spec['family'] == 'iglm':
        if metadata.get('pretrained_weights_loaded') or metadata.get('antigen_conditioning'):
            raise ValueError('Expected scratch-trained, antigen-free IgLM')
        model = IgLMScratch(dict(initialization='random_from_scratch', objective=metadata['objective'],
                               species_token=metadata['species_token']), metadata['model_config'])
        model.lm.load_state_dict(checkpoint['lm'], strict=True); model.lm.to(device).eval()
    else:
        model = MangoModel.from_checkpoint(checkpoint).to(device).eval()
    del checkpoint
    scoring = root / 'scoring'; scoring.mkdir(exist_ok=True)
    embedding_dir = root / 'embeddings' / model_name
    marker = embedding_dir / '.batch_complete.json'
    if spec['family'] != 'iglm':
        embeddings = json.loads(marker.read_text())
        if embeddings['records_sha256'] != digest(root / 'antigen_records.csv'):
            raise ValueError('Antigen embedding records changed')
        for aid, expected_hash in embeddings['files_sha256'].items():
            if digest(embedding_dir / 'test' / f'{aid}.pt') != expected_hash:
                raise ValueError('A cached antigen embedding changed')
    manifest = dict(model=model_name, checkpoint_sha256=digest(checkpoint_path),
        checkpoint_path=str(checkpoint_path), checkpoint_epoch=evaluation['checkpoint_epoch'],
        cohort_sha256=expected, measurements_sha256=complete['measurements_sha256'],
        novelty_audit_sha256=digest(root / 'cohort_audit.json'),
        embeddings_marker_sha256=None if spec['family'] == 'iglm' else digest(marker),
        precision='float32_highest', loss='variable_domain_residues_plus_eos',
        implementation_sha256={p:digest(p) for p in [__file__, 'workflow/scripts/pairing_score_cache.py',
                                                    'workflow/scripts/pairing_prefix_score_cache.py']})
    manifest_path = scoring / f'{model_name}.manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Scoring provenance changed; use a fresh score cache')
    write_json(manifest_path, manifest)
    inputs = sqlite3.connect(f'file:{(root / "measurements.sqlite").resolve()}?mode=ro', uri=True)
    db = sqlite3.connect(scoring / f'{model_name}.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS scores (target_id TEXT,antigen_id TEXT,loss_sum REAL,ntokens INTEGER,
                                          PRIMARY KEY(antigen_id,target_id));''')
    scorer = SharedPrefixChainScorer(model, spec['family'], {})
    if spec['family'] == 'iglm':
        groups = [('', chain) for chain in ('heavy', 'light')]
        statement = 'SELECT DISTINCT target_id,sequence,ntokens FROM required WHERE chain=? ORDER BY sequence,target_id'
        expected_count = inputs.execute('SELECT COUNT(DISTINCT target_id) FROM required').fetchone()[0]
    else:
        groups = inputs.execute('SELECT DISTINCT antigen_id,chain FROM required ORDER BY antigen_id,chain').fetchall()
        statement = 'SELECT target_id,sequence,ntokens FROM required WHERE chain=? AND antigen_id=? ORDER BY sequence,target_id'
        expected_count = complete['required_chain_antigen_scores']
    batches = 0; started = time.monotonic(); parity = []
    for aid, chain in groups:
        if spec['family'] != 'iglm':
            scorer.prompts.clear()
            scorer.antigens = {aid:load_embedding(str(embedding_dir / 'test' / f'{aid}.pt'))}
        known = {r[0] for r in db.execute('SELECT target_id FROM scores WHERE antigen_id=?', (aid,))}
        arguments = (chain,) if spec['family'] == 'iglm' else (chain, aid)
        missing = [r for r in inputs.execute(statement, arguments) if r[0] not in known]
        for begin in range(0, len(missing), study['batch_size']):
            batch = missing[begin:begin+study['batch_size']]
            sequences = [r[1] for r in batch]
            totals, _, _ = scorer.score(aid, chain, sequences)
            values = totals.cpu().numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Invalid sequence likelihood')
            if begin == 0:
                with torch.no_grad():
                    reference = (iglm_token_nll(model, chain, sequences[0]) if spec['family'] == 'iglm' else
                                 mango_token_nll(model, scorer.antigens[aid].to(device), chain, sequences[0]))
                delta = abs(values[0] / batch[0][2] - reference.double().mean().item())
                if delta > 3e-6:
                    raise ValueError(f'Batched/scalar production NLL differs: {delta}')
                parity.append(dict(antigen=aid, chain=chain, absolute_nll_error=delta))
            db.executemany('INSERT INTO scores VALUES (?,?,?,?)',
                [(row[0],aid,float(value),row[2]) for row,value in zip(batch, values)])
            db.commit(); batches += 1
            if batches % 100 == 0:
                count = db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
                progress = dict(model=model_name, complete=False, scored=count, expected=expected_count,
                                wall_seconds=time.monotonic()-started)
                write_json(scoring / f'{model_name}.progress.json', progress)
                print(json.dumps(progress), flush=True)
            if limit_batches and batches >= limit_batches:
                db.close(); inputs.close(); return
    count = db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
    if count != expected_count:
        raise ValueError(f'Incomplete score coverage: {count}/{expected_count}')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close(); inputs.close()
    write_json(scoring / f'{model_name}.complete.json', dict(complete=True, model=model_name,
        scores=count, expected=expected_count, wall_seconds=time.monotonic()-started,
        score_database_sha256=digest(scoring / f'{model_name}.sqlite'),
        manifest_sha256=digest(manifest_path), scalar_parity=parity))
    print(f'{model_name}: all {count:,} measured comparisons scored', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path', default='artifacts/experiments/openalphaseq_affinity_full_v1/study.json')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--limit-batches', type=int)
    main(**vars(parser.parse_args()))
