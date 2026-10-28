"""Small paired inference diagnostics, serialized on the existing GPU."""
configfile: "config/antigen_diagnostics.yaml"
ROOT = config['experiment_root']


def checkpoint(w):
    spec = config['probes'][w.model]
    if spec['family'] == 'iglm':
        return f"{config['iglm_root']}/{spec['phase']}/checkpoints/best.pt"
    return f"{config['mango_root']}/{spec['phase']}/cross_entropy/{spec['architecture']}/checkpoints/best.pt"


rule all:
    input:
        expand(f'{ROOT}/probes/{{model}}/complete.json', model=list(config['probes'])),
        f'{ROOT}/summary.json',
        f'{ROOT}/diagnostics.png',


rule probe:
    input:
        records=f"{config['cohort_dir']}/sabdab2_records.csv",
        audit=f"{config['cohort_dir']}/split_audit.json",
        checkpoint=checkpoint,
        implementation=['workflow/scripts/model_common.py', 'workflow/scripts/iglm_baseline.py'],
    output:
        complete=f'{ROOT}/probes/{{model}}/complete.json',
        scores=f'{ROOT}/probes/{{model}}/token_scores.jsonl',
        inputs=f'{ROOT}/probes/{{model}}/inputs.json',
    params:
        study=config,
    log:
        f'{ROOT}/logs/probe_{{model}}.log',
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/antigen_diagnostics.py'


rule number_regions:
    input:
        records=f"{config['cohort_dir']}/sabdab2_records.csv",
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


rule summarize:
    input:
        probes=expand(f'{ROOT}/probes/{{model}}/complete.json', model=list(config['probes'])),
        regions=f'{ROOT}/regions.json',
        implementation='workflow/scripts/summarize_antigen_diagnostics.py',
        config='config/antigen_diagnostics.yaml',
    output:
        summary=f'{ROOT}/summary.json',
        figure=f'{ROOT}/diagnostics.png',
        examples=f'{ROOT}/per_example.csv',
        scores=f'{ROOT}/scores.csv',
        comparisons=f'{ROOT}/paired_comparisons.csv',
        lengths=f'{ROOT}/length_associations.csv',
    log:
        f'{ROOT}/logs/summarize.log',
    resources:
        mem_mb=8000,
    conda:
        'envs/model.yaml'
    shell:
        'python workflow/scripts/summarize_antigen_diagnostics.py --config {input.config:q} > {log:q} 2>&1'
