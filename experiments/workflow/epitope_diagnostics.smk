"""Inference-only interventions and post-hoc structural annotation."""
import os
import json
STUDY_CONFIG=os.environ.get('MANGO_EPITOPE_CONFIG','experiments/config/epitope_diagnostics.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']

rule all:
    input: f'{ROOT}/results_report.json'

rule prepare:
    input:
        config=STUDY_CONFIG,implementation='experiments/scripts/prepare_epitope_diagnostics.py',
        common='experiments/scripts/epitope_diagnostics_common.py',worker='experiments/scripts/epitope_diagnostics_worker.py',
        report='experiments/scripts/report_epitope_diagnostics.py'
    output: f'{ROOT}/prepared.json'
    params: settings=json.dumps(dict(config),sort_keys=True)
    conda: '../../workflow/envs/analysis_anarci.yaml'
    threads: 4
    resources: mem_mb=12000
    log: f'{ROOT}/logs/prepare.log'
    shell: 'python {input.implementation:q} --config {input.config:q} --settings-json {params.settings:q} > {log:q} 2>&1'

rule probe:
    input: prepared=rules.prepare.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json'
    params: root=lambda w:ROOT
    conda: '../../workflow/envs/model.yaml'
    threads: 2
    resources: gpu=1,mem_mb=14000
    log: f'{ROOT}/logs/{{arm}}_s{{seed}}.log'
    shell: 'python experiments/scripts/epitope_diagnostics_worker.py --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule report:
    input: expand(f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json',arm=config['arms'],seed=config['seeds'])
    output: f'{ROOT}/results_report.json'
    params: root=lambda w:ROOT
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    resources: mem_mb=12000
    log: f'{ROOT}/logs/report.log'
    shell: 'python experiments/scripts/report_epitope_diagnostics.py --root {params.root:q} > {log:q} 2>&1'
