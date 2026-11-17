"""External pairing benchmark: frozen 16k models, novel antibody domains, 12 proteins."""
import sys
from pathlib import Path
sys.path[:0] = ['experiments/scripts', 'workflow/scripts']
from prepare_openalphaseq_pairing import FAMILIES

STUDY_CONFIG = 'experiments/config/openalphaseq_pairing.yaml'
configfile: STUDY_CONFIG
ROOT = config['experiment_root']
DATA = config['dataset_dir']
TRAINING = f'{ROOT}/training_numbering'


rule all:
    input:
        f'{ROOT}/summary.json',
        f'{ROOT}/findings.md',
        f'{ROOT}/pairing_comparison.png',
        f'{ROOT}/verification.json',


rule download:
    output:
        manifest=f'{DATA}/download_manifest.json',
        parquet=expand(f'{DATA}/data/{{experiment}}/data.parquet', experiment=sorted(FAMILIES)),
    params:
        root=lambda w, output: str(Path(output.manifest).parent),
    log:
        f'{ROOT}/logs/download.log',
    conda:
        '../../workflow/envs/openalphaseq_data.yaml'
    shell:
        'python experiments/scripts/download_openalphaseq.py --output {params.root:q} > {log:q} 2>&1'


rule raw:
    input:
        manifest=rules.download.output.manifest,
        parquet=rules.download.output.parquet,
        records=config['training_records'],
        audit=config['training_audit'],
        implementation='experiments/scripts/prepare_openalphaseq_pairing.py',
        config=STUDY_CONFIG,
    output:
        manifest=f'{ROOT}/raw_manifest.json',
        database=f'{ROOT}/raw.sqlite',
        components=f'{ROOT}/components.fasta',
        antigens=f'{ROOT}/antigens.csv',
    log:
        f'{ROOT}/logs/prepare.log',
    resources:
        mem_mb=10000,
    conda:
        '../../workflow/envs/openalphaseq_data.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'


rule training_fasta:
    input:
        records=config['training_records'],
        raw=rules.raw.output.manifest,
        implementation='experiments/scripts/finalize_openalphaseq_pairing.py',
        config=STUDY_CONFIG,
    output:
        fasta=f'{TRAINING}/components.fasta',
        manifest=f'{TRAINING}/inputs.json',
    log:
        f'{ROOT}/logs/training_fasta.log',
    conda:
        '../../workflow/envs/openalphaseq_data.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} --training-fasta-only > {log:q} 2>&1'


rule number_training:
    input:
        fasta=rules.training_fasta.output.fasta,
        implementation='experiments/scripts/number_openalphaseq_components.py',
    output:
        marker=f'{TRAINING}/numbering_complete.json',
    params:
        root=lambda w, input: str(Path(input.fasta).parent),
        batch_size=config['numbering_batch_size'],
    log:
        f'{ROOT}/logs/number_training.log',
    priority: 90
    threads: 2
    resources:
        mem_mb=6000,
    conda:
        '../../workflow/envs/analysis_anarci.yaml'
    shell:
        'python {input.implementation:q} --root {params.root:q} --threads {threads} '
        '--batch-size {params.batch_size} > {log:q} 2>&1'


rule number_queries:
    input:
        fasta=rules.raw.output.components,
        implementation='experiments/scripts/number_openalphaseq_components.py',
    output:
        marker=f'{ROOT}/numbering_complete.json',
    params:
        root=lambda w, input: str(Path(input.fasta).parent),
        batch_size=config['numbering_batch_size'],
    log:
        f'{ROOT}/logs/number_queries.log',
    priority: 40
    threads: 6
    resources:
        mem_mb=12000,
    conda:
        '../../workflow/envs/analysis_anarci.yaml'
    shell:
        'python {input.implementation:q} --root {params.root:q} --threads {threads} '
        '--batch-size {params.batch_size} > {log:q} 2>&1'


rule embeddings:
    input:
        antigens=rules.raw.output.antigens,
        implementation='experiments/scripts/asd_transfer_embeddings.py',
    output:
        marker=f'{ROOT}/embeddings_complete.json',
    params:
        study=config,
    log:
        f'{ROOT}/logs/embeddings.log',
    priority: 70
    threads: 2
    resources:
        gpu=1, mem_mb=20000,
    conda:
        '../../workflow/envs/embed_esm2.yaml'
    script:
        '../scripts/openalphaseq_pairing_embeddings.py'


rule scores:
    input:
        embeddings=rules.embeddings.output.marker,
        # The consumer streams committed ANARCI batches while numbering runs.
        # The report depends on both completion markers.
        fasta=rules.raw.output.components,
        checkpoints=[f'{s["directory"]}/checkpoints/best.pt' for s in config['models'].values()],
        markers=[f'{s["directory"]}/train_complete.json' for s in config['models'].values()],
        config=STUDY_CONFIG,
        implementation=['experiments/scripts/score_openalphaseq_pairing.py',
                        'experiments/scripts/pairing_score_cache.py',
                        'experiments/scripts/pairing_prefix_score_cache.py'],
    output:
        marker=f'{ROOT}/scoring_complete.json',
    log:
        f'{ROOT}/logs/scores.log',
    priority: 60
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    conda:
        '../../workflow/envs/model.yaml'
    shell:
        'python experiments/scripts/score_openalphaseq_pairing.py --config {input.config:q} > {log:q} 2>&1'


rule heldout_cohort:
    input:
        raw=rules.raw.output.manifest,
        queries=rules.number_queries.output.marker,
        training=rules.number_training.output.marker,
        implementation='experiments/scripts/finalize_openalphaseq_pairing.py',
        config=STUDY_CONFIG,
    output:
        audit=f'{ROOT}/cohort_audit.json',
        database=f'{ROOT}/cohort.sqlite',
        queries=f'{ROOT}/heldout_antibodies.csv',
        chains=f'{ROOT}/heldout_chains.csv',
        exclusions=f'{ROOT}/excluded.csv',
        labels=f'{ROOT}/labels.csv',
    log:
        f'{ROOT}/logs/cohort.log',
    resources:
        mem_mb=14000,
    conda:
        '../../workflow/envs/openalphaseq_data.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'


rule report:
    input:
        scores=rules.scores.output.marker,
        audit=rules.heldout_cohort.output.audit,
        implementation='experiments/scripts/report_openalphaseq_pairing.py',
        helpers='experiments/scripts/finalize_openalphaseq_pairing.py',
        config=STUDY_CONFIG,
    output:
        summary=f'{ROOT}/summary.json',
        findings=f'{ROOT}/findings.md',
        figure=f'{ROOT}/pairing_comparison.png',
        metrics=f'{ROOT}/metrics.csv',
        families=f'{ROOT}/per_family_metrics.csv',
        binders=f'{ROOT}/binder_metrics.csv',
        rankings=f'{ROOT}/rankings.csv',
        controls=f'{ROOT}/specificity_controls.csv',
        matrices=expand(f'{ROOT}/{{model}}_scores.npz', model=list(config['models'])),
    log:
        f'{ROOT}/logs/report.log',
    resources:
        mem_mb=16000,
    conda:
        '../../workflow/envs/openalphaseq_data.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'


rule verify:
    input:
        summary=rules.report.output.summary,
        matrices=rules.report.output.matrices,
        config=STUDY_CONFIG,
        implementation='experiments/scripts/verify_openalphaseq_pairing.py',
    output:
        marker=f'{ROOT}/verification.json',
    log:
        f'{ROOT}/logs/verify.log',
    threads: 2
    resources:
        mem_mb=14000,
    conda:
        '../../workflow/envs/model.yaml'
    shell:
        'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'
