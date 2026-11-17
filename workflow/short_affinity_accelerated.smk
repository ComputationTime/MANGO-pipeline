"""Independently executable, environment-isolated short affinity stages."""
import os
from pathlib import Path
CONFIG=os.environ.get('MANGO_SHORT_CONFIG','config/short_affinity_tests_accelerated.yaml')
configfile: CONFIG
ROOT=config['experiment_root']

rule all:
    input: f'{ROOT}/results_report.json'

rule selection:
    input: protocol=f'{ROOT}/protocol.json',implementation='workflow/scripts/accelerated_affinity_search.py'
    output: f'{ROOT}/diagnostic_selection.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/selection.log'
    shell: 'python {input.implementation:q} selection --root {params.root:q} > {log:q} 2>&1'

rule audit:
    input: protocol=f'{ROOT}/protocol.json',implementation='workflow/scripts/accelerated_affinity_search.py'
    output: f'{ROOT}/cohort_audit.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 4
    resources: gpu=1,mem_mb=18000
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/audit.log'
    shell: 'python {input.implementation:q} audit --root {params.root:q} > {log:q} 2>&1'

rule diagnostic:
    input: selection=rules.selection.output,protocol=f'{ROOT}/protocol.json',implementation='workflow/scripts/short_affinity_worker.py'
    output: f'{ROOT}/diagnostics/{{run}}.complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: 'envs/model.yaml'
    log: f'{ROOT}/logs/diagnostic_{{run}}.log'
    shell: 'python {input.implementation:q} diagnostic --root {params.root:q} --run {wildcards.run:q} > {log:q} 2>&1'

rule embeddings:
    input: audit=rules.audit.output,protocol=f'{ROOT}/protocol.json',implementation='workflow/scripts/short_affinity_embeddings.py'
    output: f'{ROOT}/embeddings_complete.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: 'envs/embed_esm2.yaml'
    log: f'{ROOT}/logs/embeddings.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'

rule regions:
    input: audit=rules.audit.output,implementation='workflow/scripts/number_diagnostic_regions.py'
    output: f'{ROOT}/regions.json'
    params: records=lambda w,input: str(Path(input[0]).parent/'records.csv')
    threads: 4
    conda: 'envs/analysis_anarci.yaml'
    log: f'{ROOT}/logs/regions.log'
    shell: 'python {input.implementation:q} --records {params.records:q} --output {output:q} --threads {threads} > {log:q} 2>&1'

rule train:
    input: embeddings=rules.embeddings.output,regions=rules.regions.output,protocol=f'{ROOT}/protocol.json',
           spec=f'{ROOT}/specs/{{run}}.json',implementation='workflow/scripts/short_affinity_worker.py'
    output: f'{ROOT}/runs/{{run}}/status.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    threads: 2
    resources: gpu=1,mem_mb=18000
    conda: 'envs/model.yaml'
    log: f'{ROOT}/logs/{{run}}.log'
    shell: 'python {input.implementation:q} train --root {params.root:q} --spec {input.spec:q} > {log:q} 2>&1'

rule report:
    input: protocol=f'{ROOT}/protocol.json',state=f'{ROOT}/state.json',implementation='workflow/scripts/report_short_affinity.py'
    output: f'{ROOT}/results_report.json'
    params: root=lambda w,input: str(Path(input.protocol).parent)
    resources: mem_mb=12000
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} --root {params.root:q} > {log:q} 2>&1'
