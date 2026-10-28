"""Execute one isolated rule in the audited AbRank transfer experiment."""
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_study(study, phase, records):
    root = Path(study['experiment_root'])
    root.mkdir(parents=True, exist_ok=True)
    audit_path = Path(study['cohort_dir']) / 'split_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'split_complete':
        raise ValueError('Audited cohort preparation must finish before training')
    if sha256(records) != audit['sha256'][str(records)]:
        raise ValueError('Prepared records differ from split audit')
    manifest = {'configuration': study, 'split_audit_sha256': sha256(audit_path)}
    path = root / 'experiment_manifest.json'
    with (root / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Experiment inputs changed. Use a new experiment_root to preserve this experiment.')
        else:
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return audit


def embed_esm2(study, phase, records, marker):
    if phase == 'asd':
        from asd_transfer_embeddings import embed_esm2 as compact_esm2
        return compact_esm2(study, phase, records, marker)
    import torch
    import embed_common as ec
    import embed_antigen_esm2 as esm_module
    root = Path(study['experiment_root'])
    output = root / phase / 'embeddings' / 'antigen' / 'esm2'
    rows = list(csv.DictReader(open(records)))
    old_rows = {r['id']: r for r in csv.DictReader(open(study['original_sabdab_records']))}
    runtime = None
    computed = reused = 0
    for index, row in enumerate(rows):
        destination = output / row['split'] / f"{row['id']}.pt"
        sequences = list(ec.antigen_sequences(row, 'resolved').items())
        antigen_hash = hashlib.sha256(json.dumps(sequences).encode()).hexdigest()
        expected_length = sum(len(seq) for _, seq in sequences) + len(sequences) - 1
        if phase == 'sabdab2':
            original = old_rows[row['id']]
            if any(original[k] != row[k] for k in ['resolved_ag_seq', 'antigen_chains']):
                raise ValueError('SAbDab2 embedding input changed')
            source = Path(study['original_esm2_dir']) / original['split'] / f"{row['id']}.pt"
            payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
            if payload.get('id') != row['id'] or payload.get('model_name') != f"esm2_{study['esm2_size']}_UR50D" or payload.get('shape') != [expected_length, 1280]:
                raise ValueError(f'Invalid existing frozen ESM2 cache: {source}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() and destination.resolve() != source.resolve():
                raise ValueError('Destination points at a different ESM2 source')
            if not destination.exists():
                destination.symlink_to(source.resolve())
            reused += 1
        else:
            cache = root / 'cache' / 'esm2' / f'{antigen_hash}.pt'
            if not cache.is_file():
                if runtime is None:
                    runtime = esm_module.load_runtime(study['esm2_size'])
                esm_module.esm2(records, row['id'], str(cache), 'resolved', study['esm2_size'], 'esm2', row=row, runtime=runtime)
                computed += 1
            payload = torch.load(cache, map_location='cpu', weights_only=False)
            if payload.get('model_name') != f"esm2_{study['esm2_size']}_UR50D" or payload.get('shape') != [expected_length, 1280]:
                raise ValueError('Invalid sequence-deduplicated ESM2 cache')
            payload.update(id=row['id'], split=row['split'], antigen_input_sha256=antigen_hash)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            reused += 1
        if (index+1) % 100 == 0 or index+1 == len(rows):
            print(f'{phase} ESM2 {index+1}/{len(rows)}; unique sequences embedded={computed}', flush=True)
    Path(marker).write_text(json.dumps(dict(status='complete', phase=phase, records=len(rows),
                                          computed_unique_sequences=computed, reused=reused,
                                          records_sha256=sha256(records)), indent=2)+'\n')


def embed_controls(study, phase, tag, records, marker):
    if phase == 'asd':
        from asd_transfer_embeddings import embed_controls as compact_controls
        return compact_controls(study, phase, tag, records, marker)
    root = Path(study['experiment_root']) / phase
    source = root / 'embeddings' / 'antigen' / 'esm2'
    output = root / 'embeddings' / 'antigen' / tag
    baseline_eval = root / 'runs' / 'esm2' / 'eval.json'
    if tag == 'control_pooled_esm2':
        from pool_antigen_embeddings import build_pooled_embeddings
        build_pooled_embeddings(str(records), str(source), str(baseline_eval), str(output),
                                str(marker), tag, 'esm2', ['train', 'val', 'test'])
    elif tag == 'control_shuffled':
        from shuffle_antigen_embeddings import build_shuffled_embeddings
        build_shuffled_embeddings(str(records), str(source), str(baseline_eval), str(output),
                                  str(marker), str(output/'mapping.csv'), tag, 'esm2', 0,
                                  ['train', 'val', 'test'])
        # A record permutation can retain the same antigen in an imbalanced
        # dataset. Report its actual biological pairing instead of assuming
        # that every different donor ID represents a different antigen.
        import torch
        by_id = {r['id']: r for r in csv.DictReader(open(records))}
        pairing = {}
        for r in csv.DictReader((output/'mapping.csv').open()):
            correct = by_id[r['target_id']]['resolved_ag_seq'] == by_id[r['donor_id']]['resolved_ag_seq']
            counts = pairing.setdefault(r['split'], {'records': 0, 'same_antigen_sequence': 0})
            counts['records'] += 1
            counts['same_antigen_sequence'] += int(correct)
            path = output/r['split']/f"{r['target_id']}.pt"
            payload = torch.load(path, map_location='cpu', weights_only=False)
            payload['pairing_correct'] = correct
            torch.save(payload, path)
        (output/'biological_pairing_audit.json').write_text(json.dumps(pairing, indent=2)+'\n')
    else:
        from embed_batch import batch_embed
        spec = dict(mode='constant' if 'constant' in tag else 'random', width=1280,
                    seed=0, shape_mode='fixed' if tag.endswith('_l1') else 'match_source',
                    tokens=1, source_embedder='esm2', source_model_name=f"esm2_{study['esm2_size']}_UR50D",
                    source_dir=str(source))
        batch_embed(str(records), str(output), str(marker), 'antigen', tag, 'antigen_control',
                    spec, 'resolved', ['train', 'val', 'test'],
                    'workflow/scripts/embed_antigen_control.py')


def run(smk):
    study = dict(smk.params.study)
    phase = smk.wildcards.phase
    records = str(smk.input.records)
    audit = validate_study(study, phase, records)
    job = smk.params.job
    if job == 'esm2':
        return embed_esm2(study, phase, records, str(smk.output.marker))
    tag = smk.wildcards.tag
    if job == 'controls':
        return embed_controls(study, phase, tag, records, str(smk.output.marker))
    import torch
    torch.set_num_threads(int(smk.threads))
    root = Path(study['experiment_root'])
    embedding_tag = 'esm2' if tag == 'control_cross_attention_esm2' else tag
    emb_dir = root/phase/'embeddings'
    model_cfg = dict(study['model'])
    if tag == 'control_cross_attention_esm2':
        model_cfg['architecture'] = 'antigen_chain_cross_attention_gpt2_v1'
    if job == 'train':
        from train_mango import train
        initial = str(smk.input.initialization[0]) if smk.input.initialization else None
        if initial:
            import model_common as mc
            checkpoint = torch.load(initial, map_location='cpu', weights_only=False)
            source_dataset = study.get('pretraining_dataset', 'abrank')
            expected = mc.comparison_cohort(Path(study['cohort_dir'])/f'{source_dataset}_records.csv')
            if checkpoint.get('cohort_sha256') != expected:
                raise ValueError('Fine-tuning checkpoint did not train on the audited pretraining cohort')
            del checkpoint
        experiment_hash = hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest()[:8]
        return train(records, str(emb_dir), tag, ['train'], ['val'], model_cfg,
            str(smk.output.ckpt), str(smk.output.latest), str(smk.output.model_config),
            str(smk.output.metrics), str(smk.output.curve), str(smk.output.plot),
            str(smk.output.run_config), f'{phase}__{tag}__{experiment_hash}', experiment_hash,
            study, embedding_tag=embedding_tag, initial_checkpoint=initial)
    if job == 'evaluate':
        from evaluate_mango import evaluate
        subsets = None
        if phase == 'sabdab2':
            ids = [r['id'] for r in csv.DictReader(open(smk.input.coverage))
                   if r['cath_assigned_all_antigen_chains'] == 'True']
            if len(ids) != audit['primary_cath_assigned_test_records']:
                raise ValueError('Primary evaluation coverage differs from split audit')
            subsets = {'test_cath_assigned': {'source_split': 'test', 'ids': ids}}
        return evaluate(records, str(emb_dir), tag, str(smk.input.ckpt),
                        str(smk.input.model_config), ['train', 'val', 'test'],
                        str(smk.output.evaluation), evaluation_subsets=subsets)
    raise ValueError(f'Unknown transfer job: {job}')


if __name__ == '__main__':
    smk = globals().get('snakemake')
    if smk is None:
        raise RuntimeError('Run through workflow/abrank_transfer.smk')
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            run(smk)
        except Exception:
            traceback.print_exc()
            raise
