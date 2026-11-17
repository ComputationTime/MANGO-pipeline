"""One isolated stage per invocation, including evaluation-amendment jobs."""
import os
from pathlib import Path
STUDY_CONFIG=os.environ.get('MANGO_CONTROLLED_CONFIG','experiments/config/controlled_antigen_study.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']
WORKER='experiments/scripts/controlled_antigen_worker.py'

rule all:
    input: f'{ROOT}/results_report.json'

rule audit:
    input: config=STUDY_CONFIG,implementation='experiments/scripts/prepare_controlled_antigen.py'
    output: f'{ROOT}/cohort_audit.json'
    threads: 4
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/audit.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'

rule embeddings:
    input: audit=rules.audit.output,protocol=f'{ROOT}/protocol.json',implementation='experiments/scripts/controlled_antigen_embeddings.py'
    output: f'{ROOT}/embeddings_complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/embed_esm2.yaml'
    log: f'{ROOT}/logs/embeddings.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'

rule regions:
    input: audit=rules.audit.output,implementation='experiments/scripts/number_diagnostic_regions.py'
    output: f'{ROOT}/regions.json'
    params: records=lambda w,input: str(Path(input[0]).parent/'records.csv')
    threads: 4
    conda: '../../workflow/envs/analysis_anarci.yaml'
    log: f'{ROOT}/logs/regions.log'
    shell: 'python {input.implementation:q} --records {params.records:q} --output {output:q} --threads {threads} > {log:q} 2>&1'

rule train:
    input: embeddings=rules.embeddings.output,regions=rules.regions.output,protocol=f'{ROOT}/protocol.json',
           spec=f'{ROOT}/specs/{{run}}.json',implementation=WORKER
    output: f'{ROOT}/runs/{{run}}/complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/{{run}}.log'
    shell: 'python {input.implementation:q} train --root {params.root:q} --spec {input.spec:q} > {log:q} 2>&1'

rule amendment:
    input: protocol=f'{ROOT}/protocol.json',implementation=WORKER
    output: f'{ROOT}/amendment_v1/{{run}}.complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/amendment_{{run}}.log'
    shell: 'python {input.implementation:q} amendment --root {params.root:q} --run {wildcards.run:q} > {log:q} 2>&1'

rule external:
    input: protocol=f'{ROOT}/protocol.json',selection=f'{ROOT}/selection.json',implementation=WORKER,
           training=f'{ROOT}/runs/{{run}}/complete.json'
    output: f'{ROOT}/external/{{run}}.complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/external_{{run}}.log'
    shell: 'python {input.implementation:q} external --root {params.root:q} --run {wildcards.run:q} > {log:q} 2>&1'

rule report:
    input: protocol=f'{ROOT}/protocol.json',state=f'{ROOT}/state.json',implementation='experiments/scripts/report_controlled_antigen.py'
    output: f'{ROOT}/results_report.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    resources: mem_mb=12000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'
