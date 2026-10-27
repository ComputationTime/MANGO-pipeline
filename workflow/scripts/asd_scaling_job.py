"""ASD-only training-size experiment using existing model/embedding contracts."""
from collections import Counter, defaultdict
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_asd_scaling import paths, record_derangement
from asd_scaling_studies import maximal_record_shuffle
from antigen_diagnostics import atomic_json, iglm_token_nll, mango_token_nll, read_csv, sha256


def validate(study, size):
    import yaml
    root = Path(study['experiment_root'])
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    if study != manifest['configuration']:
        raise ValueError('Scaling study configuration changed')
    base = yaml.safe_load(Path(study['prototype_config']).read_text())
    if base != manifest['prototype_configuration']:
        raise ValueError('Reference prototype configuration changed')
    location = paths(study, size)
    audit = json.loads(Path(location['audit']).read_text())
    if sha256(location['records']) != audit['sha256'][location['records']]:
        raise ValueError('Scaling records differ from audited cohort')
    return base, location


def embeddings(study, size):
    import torch
    from asd_transfer_embeddings import embed_esm2, write_index
    base, location = validate(study, size)
    root = Path(study['experiment_root']) / f'n{size}'
    new_tensors = 0
    if not (Path(location['embeddings']) / 'antigen/esm2/complete.json').is_file():
        import embed_common as ec
        rows = read_csv(location['records'])
        keys = {hashlib.sha256(json.dumps(list(ec.antigen_sequences(r, 'resolved').items())).encode()).hexdigest()
                for r in rows}
        new_tensors = sum(not (Path(base['esm2_cache_dir']) / f'{key}.pt').is_file() for key in keys)
        embed_esm2(dict(base, experiment_root=str(root)), 'asd', location['records'],
                   str(Path(location['embeddings']) / 'antigen/esm2/complete.json'))
    source = Path(location['embeddings']) / 'antigen/esm2'
    config = json.loads((source / 'store.json').read_text())
    if config['records_sha256'] != sha256(location['records']):
        raise ValueError('ESM2 store does not match scaling cohort')
    with sqlite3.connect(f'file:{config["index"]}?mode=ro', uri=True) as db:
        assignments = {r[1]: r for r in db.execute('SELECT split,id,antigen_key,length FROM records')}
    rows = read_csv(location['records'])
    train = [r for r in rows if r['split'] == 'train']
    shuffle = maximal_record_shuffle if study.get('allow_partial_training_shuffle') else record_derangement
    donors = shuffle(train, study['seed'])
    unchanged = sum(r['resolved_ag_seq'] == donors[r['id']]['resolved_ag_seq'] for r in train)
    target = root / 'shuffled/antigen/esm2_shuffled'
    target.mkdir(parents=True, exist_ok=True)
    shuffled = []
    with (target / 'mapping.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'id', 'donor_id', 'same_antigen_sequence'])
        for row in rows:
            donor = donors[row['id']] if row['split'] == 'train' else row
            _, _, key, length = assignments[donor['id']]
            shuffled.append((row['split'], row['id'], key, length))
            writer.writerow([row['split'], row['id'], donor['id'], row['resolved_ag_seq'] == donor['resolved_ag_seq']])
    correct_counts = Counter(assignments[r['id']][2] for r in train)
    shuffled_counts = Counter(key for split, _, key, _ in shuffled if split == 'train')
    if correct_counts != shuffled_counts:
        raise ValueError('Shuffled training changed antigen record marginals')
    write_index(target / 'index.sqlite', shuffled)
    atomic_json(target / 'store.json', dict(config, mode='shuffled', index=str((target / 'index.sqlite').resolve())))
    marker = dict(status='complete', size=size, new_esm2_tensors=new_tensors,
                  records_sha256=sha256(location['records']), shuffled_training_records=len(train),
                  shuffled_antigen_marginals_preserved=True, heldout_embeddings_correct=True,
                  shuffled_correct_antigen_records=unchanged,
                  shuffled_correct_antigen_fraction=unchanged / len(train),
                  correct_store_sha256=sha256(source / 'store.json'), shuffled_store_sha256=sha256(target / 'store.json'),
                  shuffled_mapping_sha256=sha256(target / 'mapping.csv'))
    atomic_json(root / 'embeddings_complete.json', marker)
    print(json.dumps(marker, indent=2), flush=True)


def train(study, run):
    import torch
    import iglm_baseline as iglm
    import model_common as mc
    from train_mango import train as mango_train
    torch.set_num_threads(2)
    spec = study['runs'][run]
    base, location = validate(study, spec['size'])
    directory = Path(study['experiment_root']) / 'runs' / run
    directory.mkdir(parents=True, exist_ok=True)
    cfg = dict(base['model'], epochs=spec['epochs'])
    identity = hashlib.sha256(json.dumps(dict(study=study, run=run), sort_keys=True).encode()).hexdigest()[:8]
    if spec.get('reuse'):
        source = (Path(study['iglm_root']) / 'asd' if spec['family'] == 'iglm' else
                  Path(study['prototype_root']) / 'asd/cross_entropy/cross_attention')
        for name in ('checkpoints/best.pt', 'checkpoints/latest.pt', 'model_config.json', 'config.json',
                     'metrics.jsonl', 'training_curve.csv', 'training_curve.png', 'training_runtime.json'):
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if destination.resolve() != (source / name).resolve():
                    raise ValueError('Scaling baseline points to a different source')
            elif destination.exists():
                raise ValueError('Expected a reference to the original baseline')
            else:
                destination.symlink_to((source / name).resolve())
        print(f'Reusing audited {source}', flush=True)
    elif spec['family'] == 'iglm':
        model_cfg = dict(cfg, train_splits=['train'], val_splits=['val'])
        iglm.train(location['records'], model_cfg,
                   dict(initialization='random_from_scratch', objective='full_chain_autoregressive_v1', species_token='[HUMAN]'),
                   directory, identity, study)
    else:
        architectures = {'mango': mc.MangoModel.ARCHITECTURE,
                         'mango_fixed': mc.MangoModel.FIXED_POSITION_ARCHITECTURE,
                         'cross_attention': mc.MangoModel.CROSS_ATTENTION_ARCHITECTURE}
        cfg.update(architecture=architectures[spec['family']], training_loss={'name': 'cross_entropy'})
        tag = 'esm2_shuffled' if spec.get('shuffled') else 'esm2'
        emb_dir = (str(Path(study['experiment_root']) / f'n{spec["size"]}/shuffled')
                   if spec.get('shuffled') else location['embeddings'])
        outputs = [str(directory / p) for p in ('checkpoints/best.pt', 'checkpoints/latest.pt', 'model_config.json',
                   'metrics.jsonl', 'training_curve.csv', 'training_curve.png', 'config.json')]
        mango_train(location['records'], emb_dir, tag, ['train'], ['val'], cfg, *outputs,
                    run, identity, study, embedding_tag=tag)
    chosen = directory / ('checkpoints/latest.pt' if spec['epochs'] == 2 else 'checkpoints/best.pt')
    checkpoint = torch.load(chosen, map_location='cpu', weights_only=False)
    metadata = checkpoint.get('metadata', checkpoint)
    if metadata['cohort_sha256'] != mc.comparison_cohort(location['records']):
        raise ValueError('Trained baseline cohort differs')
    if checkpoint['epoch'] != spec['epochs'] - 1:
        raise ValueError('Scaling evaluation checkpoint does not match its epoch budget')
    if json.loads((directory / 'training_runtime.json').read_text())['status'] != 'complete':
        raise ValueError('Scaling training is incomplete')
    history = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines() if x]
    if len(history) != spec['epochs']:
        raise ValueError('Wrong number of scaling epochs')
    atomic_json(directory / 'train_complete.json', dict(status='complete', run=run, spec=spec,
                cohort_sha256=metadata['cohort_sha256'], records_sha256=sha256(location['records']),
                checkpoint=str(chosen), checkpoint_sha256=sha256(chosen), evaluated_epoch=checkpoint['epoch'],
                checkpoint_policy='last_epoch_compute_control' if spec['epochs'] == 2 else 'validation_nll',
                reused=bool(spec.get('reuse'))))


def evaluate(study, run):
    import torch
    import iglm_baseline as iglm
    import model_common as mc
    from chain_examples import example_chains
    from device_common import get_device
    torch.set_num_threads(2)
    started = time.monotonic()
    device = get_device(f'ASD scaling evaluation {run}')
    spec = study['runs'][run]
    _, location = validate(study, spec['size'])
    directory = Path(study['experiment_root']) / 'runs' / run
    training = json.loads((directory / 'train_complete.json').read_text())
    if training['checkpoint_sha256'] != sha256(training['checkpoint']):
        raise ValueError('Scaling checkpoint changed after training')
    checkpoint = torch.load(training['checkpoint'], map_location='cpu', weights_only=False)
    if spec['family'] == 'iglm':
        metadata = checkpoint['metadata']
        if metadata.get('pretrained_weights_loaded') or metadata.get('antigen_conditioning'):
            raise ValueError('Scaling IgLM must be a scratch-trained antigen-free baseline')
        model = iglm.IgLMScratch(dict(initialization='random_from_scratch', objective=metadata['objective'],
                                     species_token=metadata['species_token']), metadata['model_config'])
        model.lm.load_state_dict(checkpoint['lm'])
        model.lm.to(device).eval()
    else:
        metadata = checkpoint
        model = mc.MangoModel.from_checkpoint(checkpoint).to(device).eval()
    if metadata['cohort_sha256'] != mc.comparison_cohort(location['records']):
        raise ValueError('Scaling evaluation cohort differs from training')
    rows = [r for r in read_csv(location['records']) if r['split'] in {'val', 'test'}]
    evaluation_rows = read_csv(Path(study['experiment_root']) / 'evaluation_records.csv')
    if rows != evaluation_rows:
        raise ValueError('Training size changed held-out evaluation membership')
    donors = [record_derangement(rows, seed) for seed in (study['seed'], study['seed'] + 1)]
    variants = ['correct'] if spec['family'] == 'iglm' else ['correct', 'wrong_0', 'wrong_1', 'no_context']
    totals = defaultdict(lambda: [0., 0, 0])
    with (directory / 'token_scores.jsonl.tmp').open('w') as handle, torch.no_grad():
        for row in rows:
            path = Path(location['embeddings']) / 'antigen/esm2' / row['split'] / f'{row["id"]}.pt'
            correct = mc.load_embedding(str(path)).to(device) if spec['family'] != 'iglm' else None
            for variant in variants:
                donor = donors[int(variant[-1])][row['id']] if variant.startswith('wrong_') else None
                h_ag = correct
                if donor:
                    path = Path(location['embeddings']) / 'antigen/esm2' / donor['split'] / f'{donor["id"]}.pt'
                    h_ag = mc.load_embedding(str(path)).to(device)
                    if not model.uses_cross_attention and h_ag.shape[1] != correct.shape[1]:
                        h_ag = torch.nn.functional.interpolate(h_ag.transpose(1, 2), size=correct.shape[1],
                            mode='linear', align_corners=False).transpose(1, 2)
                for chain in example_chains(row):
                    sequence = row['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']
                    losses = (iglm_token_nll(model, chain, sequence) if spec['family'] == 'iglm' else
                              mango_token_nll(model, h_ag, chain, sequence, mode=variant))
                    if len(losses) != len(sequence) + 1 or not torch.isfinite(losses).all():
                        raise ValueError('Invalid ASD scaling token losses')
                    tokens = losses.cpu().tolist()
                    handle.write(json.dumps(dict(model=run, variant=variant, id=row['id'], split=row['split'],
                        chain=chain, cluster=row['ab_ag_cluster'], antigen_length=len(row['resolved_ag_seq']),
                        sequence=sequence, donor_id=donor['id'] if donor else None,
                        token_nll=tokens), separators=(',', ':')) + '\n')
                    tally = totals[row['split'], variant]
                    tally[0] += sum(tokens)
                    tally[1] += len(tokens)
                    tally[2] += 1
    (directory / 'token_scores.jsonl.tmp').replace(directory / 'token_scores.jsonl')
    summaries = [dict(split=split, variant=variant, nll=total / tokens, n_tokens=tokens, n_examples=examples)
                 for (split, variant), (total, tokens, examples) in totals.items()]
    measured_val = next(x['nll'] for x in summaries if x['split'] == 'val' and x['variant'] == 'correct')
    if abs(measured_val - checkpoint['val_loss']) > 2e-6:
        raise ValueError('Per-token ASD scorer does not reproduce checkpoint validation NLL')
    if spec.get('reuse'):
        official = json.loads((directory / 'model_config.json').resolve().with_name('eval.json').read_text())
        for split in ('val', 'test'):
            measured = next(x for x in summaries if x['split'] == split and x['variant'] == 'correct')
            expected = official['splits'][split]
            if any(measured[k] != expected[k] for k in ('n_tokens', 'n_examples')) or abs(measured['nll'] - expected['nll']) > 2e-6:
                raise ValueError('Reused baseline scores differ from original ASD evaluation')
    atomic_json(directory / 'eval.json', dict(status='complete', run=run, spec=spec, training=training,
        evaluation_loss='cross_entropy', loss_reduction='target_token_mean_including_eos',
        heldout_sha256=sha256(Path(study['experiment_root']) / 'evaluation_records.csv'),
        token_scores_sha256=sha256(directory / 'token_scores.jsonl'),
        test_clusters=len({r['ab_ag_cluster'] for r in rows if r['split'] == 'test'}),
        wall_seconds=time.monotonic() - started, scores=summaries))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            study = dict(smk.params.study)
            job = smk.params.job
            if job == 'embeddings':
                embeddings(study, int(smk.wildcards.size))
            elif job == 'train':
                train(study, str(smk.wildcards.run))
            else:
                evaluate(study, str(smk.wildcards.run))
        except Exception:
            traceback.print_exc()
            raise
