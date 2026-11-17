"""Isolated interface-routing study; all GPU tasks share one resource."""
import json
import os

SETTINGS=os.environ.get('MANGO_ATTENTION_CONFIG','experiments/config/interface_attention.yaml')
configfile: SETTINGS
ROOT=config['experiment_root']

rule all:
    input: f'{ROOT}/independent_validation.json'

rule prepare:
    input: config=SETTINGS,implementation='experiments/scripts/prepare_interface_attention.py',
           model='experiments/scripts/interface_attention_model.py',worker='experiments/scripts/interface_attention_worker.py',
           report='experiments/scripts/report_interface_attention.py',validator='experiments/scripts/validate_interface_attention.py'
    output: f'{ROOT}/prepared.json'
    params: settings=json.dumps(dict(config),sort_keys=True)
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    resources: mem_mb=10000
    log: f'{ROOT}/logs/prepare.log'
    shell: 'python {input.implementation:q} --config {input.config:q} --settings-json {params.settings:q} > {log:q} 2>&1'

rule fit:
    input: prepared=rules.prepare.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json'
    params: root=ROOT
    conda: '../../workflow/envs/model.yaml'
    threads: 2
    resources: gpu=1,mem_mb=16000
    log: f'{ROOT}/logs/fit_{{arm}}_s{{seed}}.log'
    shell: 'python experiments/scripts/interface_attention_worker.py fit --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule score:
    input: rules.fit.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json'
    params: root=ROOT
    conda: '../../workflow/envs/model.yaml'
    threads: 2
    resources: gpu=1,mem_mb=16000
    log: f'{ROOT}/logs/score_{{arm}}_s{{seed}}.log'
    shell: 'python experiments/scripts/interface_attention_worker.py score --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule report:
    input: expand(f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json',arm=config['arms'],seed=config['seeds'])
    output: f'{ROOT}/results_report.json'
    params: root=ROOT
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    resources: mem_mb=12000
    log: f'{ROOT}/logs/report.log'
    shell: 'python experiments/scripts/report_interface_attention.py --root {params.root:q} > {log:q} 2>&1'

rule validate:
    input: rules.report.output
    output: f'{ROOT}/independent_validation.json'
    params: root=ROOT
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    resources: mem_mb=10000
    log: f'{ROOT}/logs/validate.log'
    shell: 'python experiments/scripts/validate_interface_attention.py --root {params.root:q} > {log:q} 2>&1'
