"""Each invocation executes one audited, isolated study stage on one GPU."""
import os
from pathlib import Path
STUDY_CONFIG = os.environ.get('MANGO_AFFINITY_STUDY_CONFIG','config/affinity_improvement_study.yaml')
configfile: STUDY_CONFIG
ROOT = config['experiment_root']
WORKER = 'workflow/scripts/affinity_study_worker.py'

rule all:
    input: f'{ROOT}/results_report.json'

rule audit:
    input: config=STUDY_CONFIG, implementation='workflow/scripts/prepare_affinity_study.py'
    output: f'{ROOT}/cohort_audit.json'
    threads: 4
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/audit.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'

rule embeddings:
    input: audit=rules.audit.output, protocol=f'{ROOT}/protocol.json', implementation=WORKER
    output: f'{ROOT}/embeddings_complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1, mem_mb=20000
    conda: 'envs/embed_esm2.yaml'
    log: f'{ROOT}/logs/embeddings.log'
    shell: 'python {input.implementation:q} embeddings --root {params.root:q} > {log:q} 2>&1'

rule regions:
    input: audit=rules.audit.output, implementation='workflow/scripts/number_diagnostic_regions.py'
    output: f'{ROOT}/regions.json'
    params: records=f'{ROOT}/pool.csv'
    threads: 4
    conda: 'envs/analysis_anarci.yaml'
    log: f'{ROOT}/logs/regions.log'
    shell: 'python {input.implementation:q} --records {params.records:q} --output {output:q} --threads {threads} > {log:q} 2>&1'

rule train:
    input:
        embeddings=rules.embeddings.output,
        regions=rules.regions.output,
        protocol=f'{ROOT}/protocol.json',
        spec=f'{ROOT}/specs/{{run}}.json',
        implementation=WORKER,
    output: f'{ROOT}/runs/{{run}}/complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1, mem_mb=18000
    conda: 'envs/model.yaml'
    log: f'{ROOT}/logs/{{run}}.log'
    shell: 'python {input.implementation:q} train --root {params.root:q} --spec {input.spec:q} > {log:q} 2>&1'

rule probes:
    input: protocol=f'{ROOT}/protocol.json', implementation=WORKER
    output: f'{ROOT}/probes/complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1, mem_mb=18000
    conda: 'envs/model.yaml'
    log: f'{ROOT}/logs/latest_probes.log'
    shell: 'python {input.implementation:q} probes --root {params.root:q} > {log:q} 2>&1'

rule external:
    input: selection=f'{ROOT}/selection.json', implementation=WORKER
    output: f'{ROOT}/external/complete.json'
    params: root=lambda w,input: str(Path(input.selection).parent)
    threads: 2
    resources: gpu=1, mem_mb=18000
    conda: 'envs/model.yaml'
    log: f'{ROOT}/logs/external.log'
    shell: 'python {input.implementation:q} external --root {params.root:q} > {log:q} 2>&1'

rule report:
    input: protocol=f'{ROOT}/protocol.json', state=f'{ROOT}/state.json', implementation='workflow/scripts/report_affinity_study.py'
    output: f'{ROOT}/results_report.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    resources: mem_mb=12000
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'
