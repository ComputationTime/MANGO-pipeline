"""Nine frozen likelihood models on the three local binding CSVs."""
import sys
from pathlib import Path
sys.path[:0] = ['experiments/scripts', 'workflow/scripts']

configfile: 'experiments/config/mango_binding_affinity.yaml'
ROOT = config['experiment_root']
STUDY = f'{ROOT}/study.json'
MODELS = ['one_hot','biopython','esm2','esm3','iglm','pyrosetta_pre','esm3_structure','esmif','proteinmpnn']
STRUCTURAL = {'pyrosetta_pre','esm3_structure','esmif','proteinmpnn'}
ENVIRONMENTS = dict(one_hot='embed_onehot', biopython='embed_biopython',
    pyrosetta_pre='embed_pyrosetta', esm2='embed_esm2', esm3='embed_esm3',
    esm3_structure='embed_esm3', esmif='embed_esmif', proteinmpnn='embed_proteinmpnn')
REUSE = 'experiments/scripts/reuse_mango_binding_cache.py'


rule all:
    input: f'{ROOT}/correlations/local_summary.json'


rule prepare:
    input:
        config='experiments/config/mango_binding_affinity.yaml',
        sources=expand(str(Path(config['source_dir']) / '{target}_data.csv'), target=['HER2','PD1','TIGIT']),
        implementation='experiments/scripts/prepare_mango_binding_affinity.py',
    output:
        study=STUDY,
        marker=f'{ROOT}/raw_manifest.json',
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/prepare.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'


rule finalize:
    input:
        study=rules.prepare.output.study,
        marker=rules.prepare.output.marker,
        implementation='experiments/scripts/finalize_openalphaseq_affinity.py',
    output: f'{ROOT}/measurements_complete.json'
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/finalize.log'
    shell: 'python {input.implementation:q} --study-path {input.study:q} > {log:q} 2>&1'


rule structures:
    input: study=STUDY, reuse=REUSE, implementation='experiments/scripts/fold_openalphaseq_antigens.py'
    output: f'{ROOT}/structures_complete.json'
    threads: 2
    resources: gpu=1, mem_mb=24000
    conda: '../../workflow/envs/structure_chai.yaml'
    log: f'{ROOT}/logs/structures.log'
    shell:
        '(python {input.reuse:q} structures --study-path {input.study:q} && '
        'python {input.implementation:q} --study-path {input.study:q}) > {log:q} 2>&1'


rule embeddings:
    input:
        study=STUDY,
        structures=lambda w: [f'{ROOT}/structures_complete.json'] if w.model in STRUCTURAL else [],
        reuse=REUSE,
        implementation='experiments/scripts/openalphaseq_affinity_embeddings.py',
    output: f'{ROOT}/embeddings/{{model}}/.batch_complete.json'
    wildcard_constraints: model='('+'|'.join(ENVIRONMENTS)+')'
    threads: 2
    resources: gpu=1, mem_mb=16000
    conda: lambda w: f'../../workflow/envs/{ENVIRONMENTS[w.model]}.yaml'
    log: f'{ROOT}/logs/embeddings_{{model}}.log'
    shell:
        'python {input.reuse:q} embeddings --study-path {input.study:q} --model-name {wildcards.model:q} '
        '--fallback-script {input.implementation:q} > {log:q} 2>&1'


rule scores:
    input:
        study=STUDY,
        cohort=f'{ROOT}/measurements_complete.json',
        embeddings=lambda w: [] if w.model=='iglm' else [f'{ROOT}/embeddings/{w.model}/.batch_complete.json'],
        reuse=REUSE,
        implementation='experiments/scripts/score_openalphaseq_affinity.py',
    output: f'{ROOT}/scoring/{{model}}.complete.json'
    wildcard_constraints: model='('+'|'.join(MODELS)+')'
    threads: 2
    resources: gpu=1, mem_mb=12000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/scores_{{model}}.log'
    shell:
        'python {input.reuse:q} scores --study-path {input.study:q} --model-name {wildcards.model:q} '
        '--fallback-script {input.implementation:q} > {log:q} 2>&1'


rule report:
    input:
        study=STUDY,
        scores=expand(f'{ROOT}/scoring/{{model}}.complete.json',model=MODELS),
        implementation='experiments/scripts/report_mango_binding_affinity.py',
    output: f'{ROOT}/correlations/local_summary.json'
    threads: 2
    resources: mem_mb=16000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell: 'python {input.implementation:q} --study-path {input.study:q} > {log:q} 2>&1'


rule partial_report:
    input: study=STUDY, implementation='experiments/scripts/report_mango_binding_affinity.py'
    output: f'{ROOT}/partial/local_summary.json'
    threads: 2
    resources: mem_mb=16000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/partial_report.log'
    shell: 'python {input.implementation:q} --study-path {input.study:q} --partial > {log:q} 2>&1'
