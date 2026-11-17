"""One frozen checkpoint per GPU invocation; report runs despite scoring errors."""
import os
from pathlib import Path
STUDY_CONFIG=os.environ.get('MANGO_ADAPTER_AFFINITY_CONFIG','experiments/config/legacy_adapter_affinity.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']

rule all:
    input: f'{ROOT}/results_report.json'

rule score:
    input: protocol=f'{ROOT}/protocol.json',implementation='experiments/scripts/legacy_adapter_affinity.py'
    output: f'{ROOT}/scores/{{run}}.complete.json'
    params: root=ROOT
    threads: 2
    resources: gpu=1,mem_mb=16000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/{{run}}.log'
    shell: 'python {input.implementation:q} score --root {params.root:q} --run {wildcards.run:q} > {log:q} 2>&1'

rule report:
    input: protocol=f'{ROOT}/protocol.json',state=f'{ROOT}/state.json',implementation='experiments/scripts/legacy_adapter_affinity.py'
    output: f'{ROOT}/results_report.json'
    params: root=ROOT
    resources: mem_mb=8000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} report --root {params.root:q} > {log:q} 2>&1'
