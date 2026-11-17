"""Reuse exact-sequence affinity artifacts only after verifying their provenance."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from finalize_openalphaseq_pairing import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def checked_copy(source, destination, expected):
    if digest(source) != expected:
        raise ValueError(f'Cached artifact changed: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f'Local artifact differs: {destination}')
    else:
        shutil.copyfile(source, destination)


def structures(study):
    root, cache = Path(study['experiment_root']), Path(study['cache_root'])
    manifest_path = cache / 'structures_manifest.json'
    manifest = read(manifest_path)
    if (manifest['antibody_inputs'] or manifest['affinity_inputs'] or
            manifest['method'] != study['structure'] or
            manifest['records_sha256'] != digest(cache / 'antigen_records.csv') or
            manifest['implementation_sha256'] != digest('workflow/scripts/fold_openalphaseq_antigens.py')):
        raise ValueError('Antigen-only structure provenance differs')
    source_records = {r['id']:r for r in csv.DictReader((cache / 'antigen_records.csv').open())}
    for row in csv.DictReader((root / 'antigen_records.csv').open()):
        aid = row['id']; source = cache / 'structures' / aid
        if not (source / 'complete.json').exists():
            continue  # The ordinary production folder will compute missing structures.
        status = read(source / 'complete.json')
        if (not status['complete'] or status['structure_manifest_sha256'] != digest(manifest_path) or
                source_records[aid]['resolved_ag_seq'] != row['resolved_ag_seq'] or
                (source / 'antigen.fasta').read_text() != f'>protein|name=antigen\n{row["resolved_ag_seq"]}\n'):
            raise ValueError('Cached structure sequence/provenance differs')
        destination = root / 'structures' / aid
        checked_copy(source / 'antigen.cif', destination / 'antigen.cif', status['structure_sha256'])
        checked_copy(source / 'antigen.fasta', destination / 'antigen.fasta', status['sequence_sha256'])
    return True


def embeddings(study, model_name):
    root, cache = Path(study['experiment_root']), Path(study['cache_root'])
    source = cache / 'embeddings' / model_name
    marker = source / '.batch_complete.json'
    if not marker.exists():
        return False
    details = read(marker)
    old_spec = read(cache / 'study.json')['models'][model_name]['spec']
    if (old_spec != study['models'][model_name]['spec'] or
            details['records_sha256'] != digest(cache / 'antigen_records.csv') or
            details['implementation_sha256'] != digest('workflow/scripts/openalphaseq_affinity_embeddings.py')):
        raise ValueError('Cached embedding provenance differs')
    # Structural embeddings must be generated against the verified local structures.
    if model_name in {'pyrosetta_pre', 'esm3_structure', 'esmif', 'proteinmpnn'}:
        return False
    output = root / 'embeddings' / model_name
    rows = list(csv.DictReader((root / 'antigen_records.csv').open()))
    original = {r['id']:r for r in csv.DictReader((cache / 'antigen_records.csv').open())}
    for row in rows:
        aid = row['id']
        if original[aid]['resolved_ag_seq'] != row['resolved_ag_seq']:
            raise ValueError('Embedding antigen sequence differs')
        checked_copy(source / 'test' / f'{aid}.pt', output / 'test' / f'{aid}.pt', details['files_sha256'][aid])
    details.update(records_sha256=digest(root / 'antigen_records.csv'),
                   files_sha256={r['id']:details['files_sha256'][r['id']] for r in rows},
                   reused_from=str(marker), reused_marker_sha256=digest(marker))
    write_json(output / '.batch_complete.json', details)
    return True


def scores(study, model_name):
    root, cache = Path(study['experiment_root']), Path(study['cache_root'])
    spec = study['models'][model_name]
    checkpoint = Path(spec['directory']) / 'checkpoints/best.pt'
    if digest(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    source = cache / 'scoring'; marker = source / f'{model_name}.complete.json'
    if not marker.exists():
        return False
    old = read(marker); old_manifest_path = source / f'{model_name}.manifest.json'
    old_manifest = read(old_manifest_path)
    evaluation = read(Path(spec['directory']) / 'eval.json')
    if (not old['complete'] or digest(source / f'{model_name}.sqlite') != old['score_database_sha256'] or
            digest(old_manifest_path) != old['manifest_sha256'] or
            old_manifest['checkpoint_sha256'] != spec['checkpoint_sha256'] or
            old_manifest['checkpoint_epoch'] != spec['checkpoint_epoch'] or
            old_manifest['cohort_sha256'] != evaluation['cohort_sha256'] or
            read(cache / 'cohort_audit.json')['training_records_sha256'] != digest(study['training_records']) or
            old_manifest['precision'] != 'float32_highest' or
            old_manifest['loss'] != 'variable_domain_residues_plus_eos'):
        raise ValueError('Cached likelihood provenance differs')
    for path, expected in old_manifest['implementation_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Likelihood implementation changed: {path}')
    if not old['scalar_parity'] or any(r['absolute_nll_error'] > 3e-6 for r in old['scalar_parity']):
        raise ValueError('Cached likelihoods lack scalar parity checks')
    if spec['family'] != 'iglm':
        source_embedding = cache / 'embeddings' / model_name / '.batch_complete.json'
        if digest(source_embedding) != old_manifest['embeddings_marker_sha256']:
            raise ValueError('Source score embeddings changed')
        before = read(source_embedding)['files_sha256']
        after = read(root / 'embeddings' / model_name / '.batch_complete.json')['files_sha256']
        if any(before[aid] != value for aid,value in after.items()):
            raise ValueError('Score reuse requires identical antigen embeddings')
    with sqlite3.connect(f'file:{(root / "measurements.sqlite").resolve()}?mode=ro', uri=True) as db:
        required = db.execute('SELECT DISTINCT target_id,antigen_id,ntokens FROM required').fetchall()
    if spec['family'] == 'iglm':
        required = sorted({(tid, '', n) for tid,aid,n in required})
    with sqlite3.connect(f'file:{(source / (model_name + ".sqlite")).resolve()}?mode=ro', uri=True) as db:
        values = []
        for tid, aid, ntokens in required:
            row = db.execute('SELECT * FROM scores WHERE target_id=? AND antigen_id=?', (tid,aid)).fetchone()
            if row is None:
                return False
            if row[3] != ntokens:
                raise ValueError('Cached score token count differs')
            values.append(row)
    output = root / 'scoring'; output.mkdir(exist_ok=True)
    destination = output / f'{model_name}.sqlite'
    with sqlite3.connect(destination) as db:
        db.execute('CREATE TABLE IF NOT EXISTS scores (target_id TEXT,antigen_id TEXT,loss_sum REAL,ntokens INTEGER,PRIMARY KEY(antigen_id,target_id))')
        db.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?)', values)
        if db.execute('SELECT COUNT(*) FROM scores').fetchone()[0] != len(required):
            raise ValueError('Unexpected score rows in local cache')
    manifest = dict(old_manifest, checkpoint_path=str(checkpoint),
        measurements_sha256=read(root / 'measurements_complete.json')['measurements_sha256'],
        novelty_audit_sha256=digest(root / 'cohort_audit.json'),
        embeddings_marker_sha256=None if spec['family'] == 'iglm' else digest(root / 'embeddings' / model_name / '.batch_complete.json'),
        reused_from=str(source / f'{model_name}.sqlite'), reused_complete_sha256=digest(marker))
    write_json(output / f'{model_name}.manifest.json', manifest)
    write_json(output / f'{model_name}.complete.json', dict(complete=True, model=model_name,
        scores=len(values), expected=len(required), score_database_sha256=digest(destination),
        manifest_sha256=digest(output / f'{model_name}.manifest.json'),
        scalar_parity=old['scalar_parity'], reused=True))
    print(f'{model_name}: reused {len(values):,} exact cached scores', flush=True)
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['structures', 'embeddings', 'scores'])
    p.add_argument('--study-path', required=True)
    p.add_argument('--model-name')
    p.add_argument('--fallback-script')
    args = p.parse_args(); study = read(args.study_path)
    result = structures(study) if args.kind == 'structures' else globals()[args.kind](study, args.model_name)
    if not result and args.fallback_script:
        raise SystemExit(subprocess.call([sys.executable, args.fallback_script,
            '--study-path', args.study_path, '--model-name', args.model_name]))
    raise SystemExit(0 if result else 10)
