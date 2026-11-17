"""A paired five-seed geometry/contact-gradient factorial study on one GPU."""
import os
import json
STUDY_CONFIG=os.environ.get('MANGO_INTERFACE_CONFIG','config/interface_supervision.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']

rule all:
    input: f'{ROOT}/results_report.json'

rule prepare:
    input: config=STUDY_CONFIG,implementation='workflow/scripts/prepare_interface_supervision.py',
           common='workflow/scripts/interface_supervision_common.py',model='workflow/scripts/interface_supervision_model.py',
           worker='workflow/scripts/interface_supervision_worker.py',report='workflow/scripts/report_interface_supervision.py'
    output: f'{ROOT}/prepared.json'
    params: settings=json.dumps(dict(config),sort_keys=True)
    conda: 'envs/analysis_anarci.yaml'
    threads: 2
    resources: mem_mb=14000
    log: f'{ROOT}/logs/prepare.log'
    shell: 'python {input.implementation:q} --config {input.config:q} --settings-json {params.settings:q} > {log:q} 2>&1'

rule fit:
    input: prepared=rules.prepare.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json'
    params: root=lambda w:ROOT
    conda: 'envs/model.yaml'
    threads: 2
    resources: gpu=1,mem_mb=16000
    log: f'{ROOT}/logs/fit_{{arm}}_s{{seed}}.log'
    shell: 'python workflow/scripts/interface_supervision_worker.py fit --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule score:
    input: fit=rules.fit.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json'
    params: root=lambda w:ROOT
    conda: 'envs/model.yaml'
    threads: 2
    resources: gpu=1,mem_mb=16000
    log: f'{ROOT}/logs/score_{{arm}}_s{{seed}}.log'
    shell: 'python workflow/scripts/interface_supervision_worker.py score --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule report:
    input: expand(f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json',arm=config['arms'],seed=config['seeds'])
    output: f'{ROOT}/results_report.json'
    params: root=lambda w:ROOT
    conda: 'envs/openalphaseq_data.yaml'
    resources: mem_mb=12000
    log: f'{ROOT}/logs/report.log'
    shell: 'python workflow/scripts/report_interface_supervision.py --root {params.root:q} > {log:q} 2>&1'
