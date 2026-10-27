"""Small controlled training experiments; experimental checkpoints stay isolated.

No production architecture or default is changed. All arms use the pinned pilot
targets, frozen ESM2, Adam/CE, one epoch per stage and validation NLL. Antibody
targets/region annotations never enter the antigen representation.
"""
import contextlib
import csv
import json
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antigen_diagnostics import (atomic_json, balanced_records, embedding_path, iglm_token_nll,
    mango_token_nll, memory, read_csv, sha256, wrong_antigen_map)
from model_common import MangoModel, comparison_cohort
from iglm_baseline import IgLMScratch, MODEL_CONFIG

FORMAT = 'antigen_diagnostic_checkpoint_v1'


def parameter_list(model, kind):
    return list(model.parameters()) if kind == 'mango' else (
        list(model.lm.parameters()) + (list(model.antigen_projection.parameters()) if kind == 'iglm_cross' else []))


def build_model(spec, source=None):
    """Zero-output cross attention preserves the exact IgLM function initially."""
    from transformers import GPT2Config, GPT2LMHeadModel
    if spec['kind'] == 'mango':
        architecture = (MangoModel.CROSS_ATTENTION_ARCHITECTURE if spec['architecture'] == 'cross_attention'
                        else MangoModel.ARCHITECTURE)
        model = MangoModel(1280, architecture=architecture)
    else:
        native = dict(initialization='random_from_scratch', objective='full_chain_autoregressive_v1', species_token='[HUMAN]')
        model = IgLMScratch(native)
        if spec['kind'] == 'iglm_cross':
            cfg = dict(MODEL_CONFIG, add_cross_attention=True)
            model.lm = GPT2LMHeadModel(GPT2Config(**cfg))
            model.antigen_projection = torch.nn.Linear(1280, MODEL_CONFIG['n_embd'])
    if source is not None:
        if source.get('format') == FORMAT:
            weights = source['weights']
            for key, value in weights.items():
                getattr(model, key).load_state_dict(value, strict=True)
        elif spec['kind'] == 'mango':
            if source.get('architecture') != model.architecture:
                raise ValueError('Diagnostic MANGO source architecture differs')
            for key in ('lm', 'chain_embedding', 'antigen_projection'):
                getattr(model, key).load_state_dict(source[key], strict=True)
        else:
            metadata = source['metadata']
            if metadata['model_config'] != MODEL_CONFIG or metadata['public_pretrained_weights_loaded']:
                raise ValueError('Diagnostic IgLM must use the audited scratch-trained architecture')
            missing, unexpected = model.lm.load_state_dict(source['lm'], strict=False)
            if unexpected or any('.crossattention.' not in k and '.ln_cross_attn.' not in k for k in missing):
                raise ValueError('Warm-start changed existing IgLM decoder parameters')
    if spec['kind'] == 'iglm_cross':
        if source is None or source.get('format') != FORMAT:
            for block in model.lm.transformer.h:
                torch.nn.init.zeros_(block.crossattention.c_proj.weight)
                torch.nn.init.zeros_(block.crossattention.c_proj.bias)
        if spec.get('freeze_decoder'):
            for name, parameter in model.lm.named_parameters():
                parameter.requires_grad_('.crossattention.' in name or '.ln_cross_attn.' in name)
    return model


def to_device(model, kind, device):
    if kind == 'mango':
        model.to(device)
    else:
        model.lm.to(device)
        if kind == 'iglm_cross':
            model.antigen_projection.to(device)


def set_training(model, kind, training):
    if kind == 'mango':
        model.train(training)
    else:
        model.lm.train(training)
        if kind == 'iglm_cross':
            model.antigen_projection.train(training)


def token_losses(model, spec, example, h_ag, variant='correct'):
    if spec['kind'] == 'iglm':
        return iglm_token_nll(model, example.chain, example.sequence)
    if spec['mode'] == 'pooled':
        h_ag = h_ag.mean(dim=1, keepdim=True)
    if spec['kind'] == 'iglm_cross':
        return iglm_token_nll(model, example.chain, example.sequence,
                              h_ag=None if variant == 'no_context' else h_ag)
    return mango_token_nll(model, h_ag, example.chain, example.sequence,
                           mode=variant, rebase=spec['mode'] == 'rebased',
                           standard_causal_mask=spec['mode'] == 'explicit_causal_prefix')


def stage(study, arm, phase, source_path=None):
    from device_common import get_device
    from train_mango import _rows
    from training_order import epoch_training_rows
    from training_curve import write_training_plot

    torch.set_num_threads(2)
    device = get_device(f'diagnostic training {arm}/{phase}')
    if study['epochs_per_stage'] != 1:
        raise ValueError('Diagnostic experiments require a one-epoch stage budget')
    dataset = 'asd' if phase == 'asd' else 'sabdab2'
    records_path = Path(study['cohort_dir']) / f'{dataset}_records.csv'
    cohort = comparison_cohort(records_path)
    audit = json.loads((Path(study['cohort_dir']) / 'split_audit.json').read_text())
    if sha256(records_path) != audit['sha256'][str(records_path)]:
        raise ValueError('Training records differ from audited prototype')
    spec = dict(study['arms'][arm])
    torch.manual_seed(study['seed'])
    source = torch.load(source_path, map_location='cpu', weights_only=False) if source_path else None
    if source is not None:
        source_meta = source.get('metadata', source)
        expected_path = Path(study['cohort_dir']) / ('asd_records.csv' if spec['source'] == 'prototype_asd' else 'sabdab2_records.csv')
        if source_meta['cohort_sha256'] != comparison_cohort(expected_path):
            raise ValueError('Training initialization cohort differs from the predeclared route')
        if source.get('format') == FORMAT and source_meta['arm'] != arm:
            raise ValueError('Fine-tuning must load the matching experimental ASD arm')
    model = build_model(spec, source)
    to_device(model, spec['kind'], device)
    del source
    parameters = [p for p in parameter_list(model, spec['kind']) if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(study['lr']))
    rows = read_csv(records_path)
    by_id = {r['id']: r for r in rows}
    donors = wrong_antigen_map(rows, 1000 + study['seed'])
    training = _rows(records_path, ['train'], ['heavy', 'light'])
    validation = _rows(records_path, ['val'], ['heavy', 'light'])
    directory = Path(study['experiment_root']) / arm / phase
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(format=FORMAT, arm=arm, phase=phase, spec=spec, cohort_sha256=cohort,
        records_sha256=sha256(records_path), seed=study['seed'], epochs=1, lr=study['lr'],
        training_loss='cross_entropy', loss_reduction='target_token_mean_including_eos',
        target_chains=['heavy', 'light'], target_contract='imgt_variable_vh_vl_v1',
        antigen_representation='frozen_ESM2_t33_650M',
        source_checkpoint=str(source_path) if source_path else None,
        source_checkpoint_sha256=sha256(source_path) if source_path else None,
        optimizer_loaded=False, trainable_parameters=sum(p.numel() for p in parameters),
        decoder_config=model.lm.config.to_dict(), training_examples=len(training), validation_examples=len(validation),
        order_epoch=1 if spec['source'] != 'prototype_asd' else 0)
    atomic_json(directory / 'metadata.json', metadata)
    if spec.get('shuffle_training_antigens'):
        atomic_json(directory / 'training_antigen_donors.json', {r.id: donors[r.id]['id'] for r in training})

    def antigen(example, training_mode=False, wrong=False):
        if spec['kind'] == 'iglm':
            return None
        shuffled = wrong or (training_mode and spec.get('shuffle_training_antigens'))
        row = donors[example.id] if shuffled else by_id[example.id]
        h_ag = memory(embedding_path(study, row, dataset)).to(device)
        if shuffled:
            expected = memory(embedding_path(study, by_id[example.id], dataset)).shape[1]
            if h_ag.shape[1] != expected:
                # Keep attention/dropout tensor shapes identical in the paired
                # frozen-adapter training arms; change only antigen content.
                h_ag = torch.nn.functional.interpolate(h_ag.transpose(1, 2), size=expected,
                                                       mode='linear', align_corners=False).transpose(1, 2)
        return h_ag

    def score(examples):
        set_training(model, spec['kind'], False)
        total = count = 0
        with torch.no_grad():
            for example in examples:
                losses = token_losses(model, spec, example, antigen(example))
                total += float(losses.sum())
                count += len(losses)
        return total / count

    # The frozen and joint adapter arms must begin at exactly the native IgLM
    # function, even with real antigen memory present.
    initial_validation = score(validation)
    if spec['kind'] == 'iglm_cross':
        baseline = json.loads((Path(study['iglm_root']) / 'sabdab2_finetuned/eval.json').read_text())
        if abs(initial_validation - baseline['splits']['val']['nll']) > 2e-6:
            raise ValueError('Zero-output antigen branch does not exactly preserve initial IgLM NLL')
    history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    ordered = epoch_training_rows(training, seed=study['seed'], epoch=metadata['order_epoch'], shuffle=True)
    with (directory / 'training_curve.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['iteration', 'epoch', 'phase', 'loss', 'record_id', 'chain_type'])
        writer.writeheader()
        set_training(model, spec['kind'], True)
        total = count = 0
        for i, example in enumerate(ordered, start=1):
            optimizer.zero_grad()
            losses = token_losses(model, spec, example, antigen(example, training_mode=True))
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite diagnostic training loss')
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            point = dict(iteration=i, epoch=0, phase='train', loss=value, record_id=example.id, chain_type=example.chain)
            writer.writerow(point)
            handle.flush()
            history.append(point)
            total += value * len(losses)
            count += len(losses)
            if i % 250 == 0:
                write_training_plot(history, directory / 'training_curve.png', f'{arm}/{phase}')
                print(f'{arm}/{phase}: {i}/{len(ordered)}', flush=True)
        validation_nll = score(validation)
        for label, value in [('train_epoch', total / count), ('validation', validation_nll)]:
            point = dict(iteration=len(ordered), epoch=0, phase=label, loss=value, record_id='', chain_type='')
            writer.writerow(point)
            history.append(point)
        handle.flush()
        write_training_plot(history, directory / 'training_curve.png', f'{arm}/{phase}')
    weights = {'lm': model.lm.state_dict()}
    if spec['kind'] == 'mango':
        weights['chain_embedding'] = model.chain_embedding.state_dict()
    if spec['kind'] in {'mango', 'iglm_cross'}:
        weights['antigen_projection'] = model.antigen_projection.state_dict()
    checkpoint = dict(format=FORMAT, metadata=metadata, weights=weights, epoch=0, val_loss=validation_nll)
    torch.save(checkpoint, directory / 'best.pt.tmp')
    (directory / 'best.pt.tmp').replace(directory / 'best.pt')
    atomic_json(directory / 'metrics.json', dict(initial_val_nll=initial_validation, val_nll=validation_nll,
                 train_nll=total / count, epoch=0, iterations=len(ordered)))
    atomic_json(directory / 'training_runtime.json', dict(status='complete', wall_seconds=time.monotonic() - started,
                peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))
    print(f'{arm}/{phase}: validation {initial_validation:.5f} → {validation_nll:.5f}', flush=True)
    return


def evaluate_stage(study, arm):
    from device_common import get_device
    from train_mango import _rows
    torch.set_num_threads(2)
    device = get_device(f'diagnostic evaluation {arm}')
    dataset, phase = 'sabdab2', 'native'
    spec = dict(study['arms'][arm])
    records_path = Path(study['cohort_dir']) / 'sabdab2_records.csv'
    cohort = comparison_cohort(records_path)
    directory = Path(study['experiment_root']) / arm / phase
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    if metadata['cohort_sha256'] != cohort or metadata['spec'] != spec:
        raise ValueError('Experimental evaluation cohort or architecture differs')
    model = build_model(spec, checkpoint)
    to_device(model, spec['kind'], device)
    rows = read_csv(records_path)
    by_id = {r['id']: r for r in rows}
    donors = wrong_antigen_map(rows, 1000 + study['seed'])

    def antigen(example, wrong=False):
        if spec['kind'] == 'iglm':
            return None
        row = donors[example.id] if wrong else by_id[example.id]
        return memory(embedding_path(study, row, dataset)).to(device)

    probe_records = balanced_records([r for r in rows if r['split'] == 'train'], study['train_probe_records'], study['seed'])
    selected_ids = {r['id'] for r in probe_records}
    examples = _rows(records_path, ['train', 'val', 'test'], ['heavy', 'light'])
    examples = [r for r in examples if r.split != 'train' or r.id in selected_ids]
    set_training(model, spec['kind'], False)
    variants = ['correct'] if spec['kind'] == 'iglm' else ['correct', 'wrong_0', 'no_context']
    with (directory / 'token_scores.jsonl').open('w') as handle, torch.no_grad():
        for example in examples:
            row = by_id[example.id]
            for variant in variants:
                h_ag = antigen(example, wrong=variant == 'wrong_0')
                if variant == 'wrong_0' and spec.get('architecture') == 'mango':
                    expected_length = memory(embedding_path(study, row, dataset)).shape[1]
                    h_ag = torch.nn.functional.interpolate(h_ag.transpose(1, 2), size=expected_length,
                                                           mode='linear', align_corners=False).transpose(1, 2)
                losses = token_losses(model, spec, example, h_ag, variant)
                handle.write(json.dumps(dict(model=arm, variant=variant, id=example.id, split=example.split,
                    chain=example.chain, cluster=row['ab_ag_cluster'], antigen_length=len(row['resolved_ag_seq']),
                    sequence=example.sequence, token_nll=losses.cpu().tolist()), separators=(',', ':')) + '\n')
    atomic_json(directory / 'complete.json', dict(status='complete', arm=arm, phase=phase,
                 checkpoint_sha256=sha256(directory / 'best.pt'), scores_sha256=sha256(directory / 'token_scores.jsonl'),
                 cohort_sha256=cohort, metrics=json.loads((directory / 'metrics.json').read_text())))


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            if smk.params.job == 'train':
                initial = str(smk.input.initialization[0]) if smk.input.initialization else None
                stage(dict(smk.params.study), str(smk.wildcards.arm), str(smk.wildcards.phase), initial)
            else:
                evaluate_stage(dict(smk.params.study), str(smk.wildcards.arm))
        except Exception:
            traceback.print_exc()
            raise
