"""Predeclared, one-epoch controlled follow-ups; one GPU, cached ESM2 only."""
from pathlib import Path
configfile: "config/antigen_diagnostic_training.yaml"
ROOT = config['experiment_root']

wildcard_constraints:
    arm='|'.join(config['arms']),
    phase='asd|native'


def initial(w):
    spec = config['arms'][w.arm]
    if w.phase == 'asd':
        if spec['source'] != 'prototype_asd':
            raise ValueError('Only the predeclared ASD arms may train a new ASD stage')
        return []
    if spec['source'] == 'prototype_asd':
        return [f'{ROOT}/{w.arm}/asd/best.pt']
    if spec['source'] == 'cross_finetuned':
        return [f"{config['mango_root']}/sabdab2_finetuned/cross_entropy/cross_attention/checkpoints/best.pt"]
    return [f"{config['iglm_root']}/sabdab2_finetuned/checkpoints/best.pt"]


rule all:
    input:
        expand(f'{ROOT}/{{arm}}/native/complete.json', arm=list(config['arms'])),
        f'{ROOT}/summary.json',
        f'{ROOT}/training_diagnostics.png',


rule train:
    input:
        initialization=initial,
        records=lambda w: f"{config['cohort_dir']}/{'asd' if w.phase == 'asd' else 'sabdab2'}_records.csv",
        audit=f"{config['cohort_dir']}/split_audit.json",
        implementation=['workflow/scripts/antigen_diagnostics.py', 'workflow/scripts/model_common.py',
                        'workflow/scripts/iglm_baseline.py'],
    output:
        checkpoint=f'{ROOT}/{{arm}}/{{phase}}/best.pt',
        metadata=f'{ROOT}/{{arm}}/{{phase}}/metadata.json',
        metrics=f'{ROOT}/{{arm}}/{{phase}}/metrics.json',
        curve=f'{ROOT}/{{arm}}/{{phase}}/training_curve.csv',
        plot=f'{ROOT}/{{arm}}/{{phase}}/training_curve.png',
        runtime=f'{ROOT}/{{arm}}/{{phase}}/training_runtime.json',
    params:
        study=config, job='train',
    log:
        f'{ROOT}/logs/{{arm}}_{{phase}}.log',
    threads: 2
    resources:
        gpu=1, mem_mb=18000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/train_antigen_diagnostic.py'


rule report:
    input:
        evaluations=expand(f'{ROOT}/{{arm}}/native/complete.json', arm=list(config['arms'])),
        regions=str(Path(ROOT).parent / 'regions.json'),
        config='config/antigen_diagnostic_training.yaml',
        implementation='workflow/scripts/report_diagnostic_training.py',
    output:
        summary=f'{ROOT}/summary.json',
        plot=f'{ROOT}/training_diagnostics.png',
        report=f'{ROOT}/results_report.json',
        scores=f'{ROOT}/scores.csv',
        comparisons=f'{ROOT}/paired_comparisons.csv',
        examples=f'{ROOT}/per_example.csv',
    log:
        f'{ROOT}/logs/report.log',
    resources:
        mem_mb=12000,
    conda:
        'envs/model.yaml'
    shell:
        'python workflow/scripts/report_diagnostic_training.py --config {input.config:q} '
        '--require-complete > {log:q} 2>&1'


rule native_evaluation:
    input:
        checkpoint=f'{ROOT}/{{arm}}/native/best.pt',
        records=f"{config['cohort_dir']}/sabdab2_records.csv",
        metadata=f'{ROOT}/{{arm}}/native/metadata.json',
        metrics=f'{ROOT}/{{arm}}/native/metrics.json',
        runtime=f'{ROOT}/{{arm}}/native/training_runtime.json',
    output:
        marker=f'{ROOT}/{{arm}}/native/complete.json',
        scores=f'{ROOT}/{{arm}}/native/token_scores.jsonl',
    params:
        study=config, job='evaluate',
    log:
        f'{ROOT}/logs/{{arm}}_evaluate.log',
    threads: 2
    resources:
        gpu=1, mem_mb=18000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/train_antigen_diagnostic.py'
