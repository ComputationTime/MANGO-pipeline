"""Frozen ESM2, two losses, two architectures, with/without ASD pretraining."""
from pathlib import Path

configfile: "config/asd_loss_comparison.yaml"

ROOT = config['experiment_root']
COHORT = config['cohort_dir']
EMBEDDINGS = config['embedding_root']
PHASES = ['asd', 'sabdab2_scratch', 'sabdab2_finetuned']
LOSSES = list(config['objectives'])
ARCHITECTURES = list(config['architectures'])

wildcard_constraints:
    phase='|'.join(PHASES),
    dataset='asd|sabdab2',
    objective='|'.join(LOSSES),
    architecture='|'.join(ARCHITECTURES)


def dataset(w):
    return 'asd' if w.phase == 'asd' else 'sabdab2'


def records(w):
    return f'{COHORT}/{dataset(w)}_records.csv'


rule all:
    input:
        f'{ROOT}/plots/sabdab2_loss_comparison.png',
        f'{ROOT}/plots/sabdab2_loss_comparison.csv',


rule frozen_esm2:
    input:
        records=lambda w: f'{COHORT}/{w.dataset}_records.csv',
        audit=f'{COHORT}/split_audit.json',
        original=config['original_sabdab_records'],
        implementation=['workflow/scripts/asd_transfer_embeddings.py', 'workflow/scripts/embed_antigen_esm2.py'],
    output:
        marker=f'{EMBEDDINGS}/{{dataset}}/embeddings/antigen/esm2/complete.json',
    params:
        job='embeddings', study=config,
    log:
        f'{ROOT}/logs/{{dataset}}_embed_esm2.log',
    threads: 2
    resources:
        gpu=1, mem_mb=20000,
    conda:
        'envs/embed_esm2.yaml'
    script:
        'scripts/asd_loss_comparison_job.py'


rule train_model:
    input:
        records=records,
        audit=f'{COHORT}/split_audit.json',
        embeddings=lambda w: f'{EMBEDDINGS}/{dataset(w)}/embeddings/antigen/esm2/complete.json',
        initialization=lambda w: [f'{ROOT}/asd/{w.objective}/{w.architecture}/checkpoints/best.pt'] if w.phase == 'sabdab2_finetuned' else [],
        implementation=['workflow/scripts/train_mango.py', 'workflow/scripts/model_common.py',
                        'workflow/scripts/training_losses.py', 'workflow/scripts/chain_examples.py',
                        'workflow/scripts/embedding_store.py', 'workflow/scripts/training_order.py'],
    output:
        ckpt=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/checkpoints/best.pt',
        latest=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/checkpoints/latest.pt',
        model_config=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/model_config.json',
        metrics=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/metrics.jsonl',
        curve=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/training_curve.csv',
        plot=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/training_curve.png',
        run_config=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/config.json',
    params:
        job='train', study=config,
    log:
        f'{ROOT}/logs/{{phase}}_{{objective}}_{{architecture}}_train.log',
    threads: 2
    resources:
        gpu=1, mem_mb=18000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/asd_loss_comparison_job.py'


rule evaluate_model:
    input:
        records=records,
        audit=f'{COHORT}/split_audit.json',
        coverage=f'{COHORT}/sabdab2_test_coverage.csv',
        embeddings=lambda w: f'{EMBEDDINGS}/{dataset(w)}/embeddings/antigen/esm2/complete.json',
        ckpt=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/checkpoints/best.pt',
        model_config=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/model_config.json',
        implementation=['workflow/scripts/evaluate_mango.py', 'workflow/scripts/model_common.py'],
    output:
        evaluation=f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/eval.json',
    params:
        job='evaluate', study=config,
    log:
        f'{ROOT}/logs/{{phase}}_{{objective}}_{{architecture}}_eval.log',
    threads: 2
    resources:
        gpu=1, mem_mb=14000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/asd_loss_comparison_job.py'


rule plot_comparison:
    input:
        evaluations=expand(f'{ROOT}/{{phase}}/{{objective}}/{{architecture}}/eval.json',
                           phase=['sabdab2_scratch', 'sabdab2_finetuned'],
                           objective=LOSSES, architecture=ARCHITECTURES),
    output:
        figure=f'{ROOT}/plots/sabdab2_loss_comparison.png',
        data=f'{ROOT}/plots/sabdab2_loss_comparison.csv',
    log:
        f'{ROOT}/logs/plot_comparison.log',
    conda:
        'envs/analysis_plot.yaml'
    script:
        'scripts/analysis/plot_asd_loss_comparison.py'
