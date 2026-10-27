"""Train native-only and ASD-pretrained IgLM on the rapid prototype cohorts."""
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abrank_transfer_job import validate_study


def run(smk):
    import torch
    import iglm_baseline as iglm
    from model_common import comparison_cohort

    study = dict(smk.params.study)
    mango_manifest = Path(smk.input.mango_manifest)
    mango = json.loads(mango_manifest.read_text())['configuration']
    if (not mango.get('prototype') or mango['model']['epochs'] != 1 or
            Path(mango['cohort_dir']).resolve() != Path(study['cohort_dir']).resolve()):
        raise ValueError('IgLM must match the one-epoch HER2-free MANGO prototype')
    study['mango_manifest_sha256'] = hashlib.sha256(mango_manifest.read_bytes()).hexdigest()
    study['model'] = dict(mango['model'], train_splits=['train'], val_splits=['val'])
    phase = smk.wildcards.phase
    records = str(smk.input.records)
    audit = validate_study(study, phase, records)
    torch.set_num_threads(int(smk.threads))
    if smk.params.job == 'train':
        initial = str(smk.input.initialization[0]) if smk.input.initialization else None
        if initial:
            source = torch.load(initial, map_location='cpu', weights_only=False)
            expected = comparison_cohort(Path(study['cohort_dir']) / 'asd_records.csv')
            if source['metadata']['cohort_sha256'] != expected or source['epoch'] != 0:
                raise ValueError('IgLM initialization must be the one-epoch prototype ASD checkpoint')
            del source
        identity = dict(study, architecture=iglm.ARCHITECTURE, model_config=iglm.MODEL_CONFIG,
                        vocabulary=iglm.VOCAB, phase=phase)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
        return iglm.train(records, study['model'], study['iglm_baseline'],
                          Path(smk.output.ckpt).parents[1], digest, study,
                          initial_checkpoint=initial)
    if smk.params.job == 'evaluate':
        subsets = None
        if phase != 'asd':
            with open(smk.input.coverage) as handle:
                ids = [r['id'] for r in csv.DictReader(handle)
                       if r['cath_assigned_all_antigen_chains'] == 'True']
            if len(ids) != audit['primary_cath_assigned_test_records']:
                raise ValueError('Native primary evaluation membership changed')
            subsets = {'test_cath_assigned': {'source_split': 'test', 'ids': ids}}
        result = iglm.evaluate(records, str(smk.input.ckpt), str(smk.output.evaluation),
                               evaluation_subsets=subsets)
        result.update(experiment_phase=phase, objective='cross_entropy', architecture_label='iglm',
                      training_route='asd_pretrained' if phase == 'sabdab2_finetuned' else 'scratch')
        iglm._write_json(smk.output.evaluation, result)
        return result
    raise ValueError(f'Unknown IgLM prototype job: {smk.params.job}')


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            run(smk)
        except Exception:
            traceback.print_exc()
            raise
