"""Full AlphaSeq affinity comparison of the eight native MANGO models and IgLM."""
import sys
from pathlib import Path
sys.path[:0] = ['experiments/scripts', 'workflow/scripts']
from prepare_openalphaseq_affinity import model_directories

STUDY_CONFIG = 'experiments/config/openalphaseq_affinity.yaml'
configfile: STUDY_CONFIG
ROOT = config['experiment_root']
STUDY = f'{ROOT}/study.json'
MODELS = model_directories(config)
STRUCTURAL = {'pyrosetta_pre', 'esm3_structure', 'esmif', 'proteinmpnn'}
ENVIRONMENTS = dict(one_hot='embed_onehot', biopython='embed_biopython',
    pyrosetta_pre='embed_pyrosetta', esm2='embed_esm2', esm3='embed_esm3',
    esm3_structure='embed_esm3', esmif='embed_esmif', proteinmpnn='embed_proteinmpnn')


rule all:
    input:
        f'{ROOT}/correlations/summary.json',


rule prepare:
    input:
        config=STUDY_CONFIG,
        records=config['training_records'],
        snapshot=f'{config["dataset_dir"]}/download_manifest.json',
        annotations=f'{config["annotation_source"]}/numbering_complete.json',
        implementation=['experiments/scripts/prepare_openalphaseq_affinity.py',
                        'experiments/scripts/prepare_openalphaseq_pairing.py'],
        training_configs=config['training_configs'],
    output:
        study=STUDY,
        raw=f'{ROOT}/raw_manifest.json',
        antigens=f'{ROOT}/antigens.csv',
        records=f'{ROOT}/antigen_records.csv',
        fasta=f'{ROOT}/training_numbering/components.fasta',
        training_audit=f'{ROOT}/training_audit.json',
        numbering=f'{ROOT}/numbering_complete.json',
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    resources: mem_mb=12000
    log: f'{ROOT}/logs/prepare.log'
    shell:
        'python experiments/scripts/prepare_openalphaseq_affinity.py --config {input.config:q} > {log:q} 2>&1'


rule number_training:
    input:
        fasta=rules.prepare.output.fasta,
        implementation='experiments/scripts/number_openalphaseq_components.py',
    output: f'{ROOT}/training_numbering/numbering_complete.json'
    params: root=f'{ROOT}/training_numbering'
    threads: 4
    resources: mem_mb=6000
    conda: '../../workflow/envs/analysis_anarci.yaml'
    log: f'{ROOT}/logs/number_training.log'
    shell:
        'python {input.implementation:q} --root {params.root:q} --threads {threads} > {log:q} 2>&1'


rule finalize:
    input:
        study=rules.prepare.output.study,
        raw=rules.prepare.output.raw,
        queries=rules.prepare.output.numbering,
        training=rules.number_training.output,
        implementation=['experiments/scripts/finalize_openalphaseq_affinity.py',
                        'experiments/scripts/finalize_openalphaseq_pairing.py'],
    output:
        marker=f'{ROOT}/measurements_complete.json',
        audit=f'{ROOT}/cohort_audit.json',
        cohort=f'{ROOT}/cohort.sqlite',
        measurements=f'{ROOT}/measurements.sqlite',
    resources: mem_mb=16000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/finalize.log'
    shell:
        'python experiments/scripts/finalize_openalphaseq_affinity.py --study-path {input.study:q} > {log:q} 2>&1'


rule structures:
    input:
        study=rules.prepare.output.study,
        records=rules.prepare.output.records,
        implementation='experiments/scripts/fold_openalphaseq_antigens.py',
    output: f'{ROOT}/structures_complete.json'
    threads: 2
    resources: gpu=1, mem_mb=24000
    conda: '../../workflow/envs/structure_chai.yaml'
    log: f'{ROOT}/logs/structures.log'
    shell:
        'python {input.implementation:q} --study-path {input.study:q} > {log:q} 2>&1'


rule embeddings:
    input:
        study=rules.prepare.output.study,
        records=rules.prepare.output.records,
        structures=lambda w: [f'{ROOT}/structures_complete.json'] if w.model in STRUCTURAL else [],
        implementation=['experiments/scripts/openalphaseq_affinity_embeddings.py', 'workflow/scripts/embed_batch.py'],
    output: f'{ROOT}/embeddings/{{model}}/.batch_complete.json'
    wildcard_constraints: model='('+'|'.join(ENVIRONMENTS)+')'
    threads: 2
    resources: gpu=1, mem_mb=16000
    conda: lambda w: f'../../workflow/envs/{ENVIRONMENTS[w.model]}.yaml'
    log: f'{ROOT}/logs/embeddings_{{model}}.log'
    shell:
        'python experiments/scripts/openalphaseq_affinity_embeddings.py '
        '--study-path {input.study:q} --model-name {wildcards.model:q} > {log:q} 2>&1'


rule scores:
    input:
        study=rules.prepare.output.study,
        cohort=rules.finalize.output.marker,
        embeddings=lambda w: [] if w.model=='iglm' else [f'{ROOT}/embeddings/{w.model}/.batch_complete.json'],
        checkpoint=lambda w: f'{MODELS[w.model]["directory"]}/checkpoints/best.pt',
        evaluation=lambda w: f'{MODELS[w.model]["directory"]}/eval.json',
        implementation=['experiments/scripts/score_openalphaseq_affinity.py',
                        'experiments/scripts/pairing_score_cache.py','experiments/scripts/pairing_prefix_score_cache.py'],
    output: f'{ROOT}/scoring/{{model}}.complete.json'
    wildcard_constraints: model='('+'|'.join(MODELS)+')'
    threads: 2
    resources: gpu=1, mem_mb=12000
    conda: '../../workflow/envs/model.yaml'
    log: f'{ROOT}/logs/scores_{{model}}.log'
    shell:
        'python experiments/scripts/score_openalphaseq_affinity.py '
        '--study-path {input.study:q} --model-name {wildcards.model:q} > {log:q} 2>&1'


rule report:
    input:
        study=rules.prepare.output.study,
        scores=expand(f'{ROOT}/scoring/{{model}}.complete.json',model=list(MODELS)),
        implementation='experiments/scripts/report_openalphaseq_affinity.py',
    output:
        summary=f'{ROOT}/correlations/summary.json',
        metrics=f'{ROOT}/correlations/pearson_correlations.csv',
        primary=f'{ROOT}/correlations/primary_correlations.csv',
        findings=f'{ROOT}/correlations/findings.md',
        plot=f'{ROOT}/correlations/perplexity_affinity.png',
    threads: 2
    resources: mem_mb=16000
    conda: '../../workflow/envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/report.log'
    shell:
        'python {input.implementation:q} --study-path {input.study:q} > {log:q} 2>&1'
