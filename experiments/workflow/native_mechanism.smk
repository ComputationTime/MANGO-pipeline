"""Isolated native mechanism audit, with exclusive single-GPU search resources."""
import os
from pathlib import Path
STUDY_CONFIG=os.environ.get('MANGO_NATIVE_MECHANISM_CONFIG','experiments/config/native_mechanism_study.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']
AUDIT_SCRIPT=('experiments/scripts/prepare_native_smoke.py' if config.get('smoke') else
              'experiments/scripts/prepare_native_pilot.py' if 'reference_audit_root' in config else 'experiments/scripts/prepare_native_mechanism.py')

rule all:
    input: f'{ROOT}/cohort_audit.json'

rule audit:
    input: config=STUDY_CONFIG, implementation=AUDIT_SCRIPT
    output: f'{ROOT}/cohort_audit.json'
    threads: 4
    resources: gpu=1,mem_mb=16000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/audit.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'

rule schedules:
    input: audit=rules.audit.output,protocol=f'{ROOT}/training_protocol.json',implementation='experiments/scripts/prepare_native_schedules.py'
    output: f'{ROOT}/schedules_complete.json'
    params: root=ROOT
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/schedules.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'

rule embeddings:
    input: audit=rules.audit.output,protocol=f'{ROOT}/training_protocol.json',implementation='experiments/scripts/native_mechanism_embeddings.py'
    output: f'{ROOT}/embeddings_complete.json'
    params: root=ROOT
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/embed_esm2.yaml'
    log: f'{ROOT}/logs/embeddings.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'

rule fit:
    input: audit=rules.audit.output,protocol=f'{ROOT}/training_protocol.json',schedules=rules.schedules.output,
           implementation='experiments/scripts/native_mechanism_worker.py'
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json'
    params: root=ROOT
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/{{arm}}_s{{seed}}.log'
    shell: 'python {input.implementation:q} train --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed:q} > {log:q} 2>&1'

rule external:
    input: fit=rules.fit.output,implementation='experiments/scripts/native_mechanism_worker.py'
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/affinity_complete.json'
    params: root=ROOT
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/external_{{arm}}_s{{seed}}.log'
    shell: 'python {input.implementation:q} external --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed:q} > {log:q} 2>&1'

rule report:
    input: protocol=f'{ROOT}/training_protocol.json',state=f'{ROOT}/state.json',implementation='experiments/scripts/report_native_mechanism.py'
    output: f'{ROOT}/results_report.json'
    params: root=ROOT
    resources: mem_mb=12000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'
