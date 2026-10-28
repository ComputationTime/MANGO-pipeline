"""Nested ASD-only 2k/4k cohorts, paired/shuffled ESM2 and IgLM controls."""
import sys
import os
sys.path.insert(0, 'workflow/scripts')
from prepare_asd_scaling import paths

STUDY_CONFIG = os.environ.get('MANGO_ASD_SCALING_CONFIG', 'config/asd_scaling.yaml')
configfile: STUDY_CONFIG
ROOT = config['experiment_root']
REPORT_SCRIPT = config.get('report_script', 'workflow/scripts/report_asd_scaling.py')

wildcard_constraints:
    size='|'.join(map(str, config['sizes'])),
    run='|'.join(config['runs'])


def records(w):
    return paths(config, config['runs'][w.run]['size'])['records']


rule all:
    input:
        expand(f'{ROOT}/runs/{{run}}/eval.json', run=list(config['runs'])),
        expand(f'{ROOT}/runs/{{run}}/far_eval.json', run=[r for r,s in config['runs'].items() if s['family'] != 'iglm']),
        f'{ROOT}/summary.json',
        f'{ROOT}/asd_scaling.png',


rule embeddings:
    input:
        records=lambda w: paths(config, int(w.size))['records'],
        audit=f'{ROOT}/cohort_audit.json',
        implementation=['workflow/scripts/asd_transfer_embeddings.py', 'workflow/scripts/prepare_asd_scaling.py'],
    output:
        marker=f'{ROOT}/n{{size}}/embeddings_complete.json',
    params:
        job='embeddings', study=config,
    log:
        f'{ROOT}/logs/n{{size}}_embeddings.log',
    threads: 2
    resources:
        gpu=1, mem_mb=20000,
    conda:
        'envs/embed_esm2.yaml'
    script:
        'scripts/asd_scaling_job.py'


rule number_regions:
    input:
        records=f'{ROOT}/evaluation_records.csv',
        implementation='workflow/scripts/number_diagnostic_regions.py',
    output:
        regions=f'{ROOT}/regions.json',
    log:
        f'{ROOT}/logs/number_regions.log',
    threads: 2
    resources:
        mem_mb=8000,
    conda:
        'envs/analysis_anarci.yaml'
    shell:
        'python workflow/scripts/number_diagnostic_regions.py --records {input.records:q} '
        '--output {output.regions:q} --threads {threads} > {log:q} 2>&1'


rule report:
    input:
        evaluations=expand(f'{ROOT}/runs/{{run}}/eval.json', run=list(config['runs'])),
        far=expand(f'{ROOT}/runs/{{run}}/far_eval.json', run=[r for r,s in config['runs'].items() if s['family'] != 'iglm']),
        regions=f'{ROOT}/regions.json',
        config=STUDY_CONFIG,
        implementation=REPORT_SCRIPT,
        helpers=['workflow/scripts/report_asd_scaling.py',
                 'workflow/scripts/summarize_antigen_diagnostics.py'],
        reference=[config['reference_config']] if 'reference_config' in config else [],
        availability=[f'{ROOT}/availability.json'] if 'reference_config' in config else [],
    output:
        summary=f'{ROOT}/summary.json',
        plot=f'{ROOT}/asd_scaling.png',
        report=f'{ROOT}/results_report.json',
        scores=f'{ROOT}/scores.csv',
        comparisons=f'{ROOT}/paired_comparisons.csv',
        findings=f'{ROOT}/findings.md',
    log:
        f'{ROOT}/logs/report.log',
    resources:
        mem_mb=12000,
    conda:
        'envs/model.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} '
        '--require-complete > {log:q} 2>&1'


rule far_probe:
    input:
        training=f'{ROOT}/runs/{{run}}/train_complete.json',
        heldout=f'{ROOT}/evaluation_records.csv',
        implementation='workflow/scripts/antigen_diagnostics.py',
    output:
        marker=f'{ROOT}/runs/{{run}}/far_eval.json',
        scores=f'{ROOT}/runs/{{run}}/far_token_scores.jsonl',
    params:
        study=config,
    log:
        f'{ROOT}/logs/{{run}}_far_probe.log',
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/asd_scaling_far_probe.py'


rule train:
    input:
        records=records,
        cohort_audit=f'{ROOT}/cohort_audit.json',
        embeddings=lambda w: f"{ROOT}/n{config['runs'][w.run]['size']}/embeddings_complete.json",
        implementation=['workflow/scripts/train_mango.py', 'workflow/scripts/iglm_baseline.py',
                        'workflow/scripts/model_common.py', 'workflow/scripts/chain_examples.py'],
    output:
        marker=f'{ROOT}/runs/{{run}}/train_complete.json',
        best=f'{ROOT}/runs/{{run}}/checkpoints/best.pt',
        latest=f'{ROOT}/runs/{{run}}/checkpoints/latest.pt',
        model_config=f'{ROOT}/runs/{{run}}/model_config.json',
        run_config=f'{ROOT}/runs/{{run}}/config.json',
        metrics=f'{ROOT}/runs/{{run}}/metrics.jsonl',
        curve=f'{ROOT}/runs/{{run}}/training_curve.csv',
        plot=f'{ROOT}/runs/{{run}}/training_curve.png',
        runtime=f'{ROOT}/runs/{{run}}/training_runtime.json',
    params:
        job='train', study=config,
    log:
        f'{ROOT}/logs/{{run}}_train.log',
    threads: 2
    resources:
        gpu=1, mem_mb=18000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/asd_scaling_job.py'


rule evaluate:
    input:
        training=f'{ROOT}/runs/{{run}}/train_complete.json',
        records=records,
        heldout=f'{ROOT}/evaluation_records.csv',
        implementation=['workflow/scripts/antigen_diagnostics.py', 'workflow/scripts/prepare_asd_scaling.py'],
    output:
        evaluation=f'{ROOT}/runs/{{run}}/eval.json',
        scores=f'{ROOT}/runs/{{run}}/token_scores.jsonl',
    params:
        job='evaluate', study=config,
    log:
        f'{ROOT}/logs/{{run}}_eval.log',
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/asd_scaling_job.py'
