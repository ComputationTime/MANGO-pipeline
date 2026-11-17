"""One-card hardware/search validation, independent of frozen model studies."""
from pathlib import Path
ROOT='artifacts/experiments/homology_backend_probe_v1'
SCRIPT='experiments/scripts/benchmark_homology_backend.py'

rule all:
    input: f'{ROOT}/results_report.json'

rule prepare:
    input: SCRIPT
    output: f'{ROOT}/prepared.json'
    threads: 4
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/prepare_launcher.log'
    shell: 'python {input:q} prepare > {log:q} 2>&1'

rule search:
    input: prepared=rules.prepare.output,script=SCRIPT
    output: f'{ROOT}/{{backend,cpu|gpu|gpu_graph}}.json'
    resources: gpu=lambda w: int(w.backend!='cpu'),mem_mb=16000
    threads: 4
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/{{backend}}_launcher.log'
    shell: 'python {input.script:q} {wildcards.backend:q} > {log:q} 2>&1'

rule report:
    input: expand(f'{ROOT}/{{backend}}.json',backend=['cpu','gpu','gpu_graph']),script=SCRIPT
    output: f'{ROOT}/results_report.json'
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/report_launcher.log'
    shell: 'python {input.script:q} report > {log:q} 2>&1'
