"""Run one stage of the audited ESM2 objective/architecture/transfer study."""
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abrank_transfer_job import validate_study, embed_esm2


def run(smk):
    study = dict(smk.params.study)
    records = str(smk.input.records)
    job = smk.params.job
    phase = smk.wildcards.dataset if job == 'embeddings' else smk.wildcards.phase
    audit = validate_study(study, phase, records)
    if job == 'embeddings':
        embedding_study = dict(study, experiment_root=study['embedding_root'])
        return embed_esm2(embedding_study, phase, records, str(smk.output.marker))
    import torch
    import model_common as mc
    from training_losses import loss_configuration
    torch.set_num_threads(int(smk.threads))
    objective, architecture = smk.wildcards.objective, smk.wildcards.architecture
    model_cfg = dict(study['model'], architecture=study['architectures'][architecture],
                     training_loss=loss_configuration(study['objectives'][objective]))
    dataset = 'asd' if phase == 'asd' else 'sabdab2'
    emb_dir = str(Path(study['embedding_root']) / dataset / 'embeddings')
    marker = json.loads(Path(smk.input.embeddings).read_text())
    records_sha = hashlib.sha256(Path(records).read_bytes()).hexdigest()
    if marker.get('records_sha256') != records_sha:
        raise ValueError('Frozen ESM2 marker does not match the phase cohort')
    tag = 'esm2'  # Every run consumes the actual frozen ESM2 representation.
    if job == 'train':
        from train_mango import train
        initial = str(smk.input.initialization[0]) if smk.input.initialization else None
        if initial:
            checkpoint = torch.load(initial, map_location='cpu', weights_only=False)
            expected = mc.comparison_cohort(Path(study['cohort_dir']) / 'asd_records.csv')
            if checkpoint.get('cohort_sha256') != expected:
                raise ValueError('Initialization did not train on the audited ASD cohort')
            if loss_configuration(checkpoint.get('training_loss')) != model_cfg['training_loss']:
                raise ValueError('Initialization comes from a different loss arm')
            del checkpoint
        digest = hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest()[:8]
        return train(records, emb_dir, tag, ['train'], ['val'], model_cfg,
                     str(smk.output.ckpt), str(smk.output.latest), str(smk.output.model_config),
                     str(smk.output.metrics), str(smk.output.curve), str(smk.output.plot),
                     str(smk.output.run_config), f'{phase}__{objective}__{architecture}__{digest}',
                     digest, study, embedding_tag='esm2', initial_checkpoint=initial)
    if job == 'evaluate':
        from evaluate_mango import evaluate
        subsets = None
        if dataset == 'sabdab2':
            with open(smk.input.coverage) as handle:
                ids = [r['id'] for r in csv.DictReader(handle) if r['cath_assigned_all_antigen_chains'] == 'True']
            if len(ids) != audit['primary_cath_assigned_test_records']:
                raise ValueError('Native primary evaluation membership changed')
            subsets = {'test_cath_assigned': {'source_split': 'test', 'ids': ids}}
        result = evaluate(records, emb_dir, tag, str(smk.input.ckpt), str(smk.input.model_config),
                          ['train', 'val', 'test'], str(smk.output.evaluation), evaluation_subsets=subsets)
        path = Path(smk.output.evaluation)
        payload = json.loads(path.read_text())
        payload.update(experiment_phase=phase, objective=objective, architecture_label=architecture,
                       training_route='asd_pretrained' if phase == 'sabdab2_finetuned' else 'scratch')
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        return result
    raise ValueError(f'Unknown comparison job: {job}')


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            run(smk)
        except Exception:
            traceback.print_exc()
            raise
