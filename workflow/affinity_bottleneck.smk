"""Read-only data/graph audit, using the existing isolated analysis environment."""
import os
from pathlib import Path
import yaml
STUDY_CONFIG=os.environ.get('MANGO_AFFINITY_AUDIT_CONFIG','config/affinity_bottleneck_audit.yaml')
LINEAGE_CONFIG=os.environ.get('MANGO_AFFINITY_LINEAGE_CONFIG','config/affinity_lineage_review.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']
LINEAGE=yaml.safe_load(Path(LINEAGE_CONFIG).read_text())
if LINEAGE['audit_root'] != ROOT or LINEAGE['experiment_root'] != f'{ROOT}/lineage_review_v1':
    raise ValueError('Custom audit roots require a matching MANGO_AFFINITY_LINEAGE_CONFIG.')
REFERENCE=yaml.safe_load(Path(config['reference_config']).read_text())
SEARCH_ROOT=Path(REFERENCE['reference_audit_root'])/'searches'
VERIFIED_INPUTS=[
    config['reference_config'],REFERENCE['source_records'],
    str(Path(REFERENCE['inventory_root'])/'membership.csv'),
    str(Path(REFERENCE['experiment_root'])/'cohort_audit.json'),
    str(Path(REFERENCE['experiment_root'])/'component_membership.csv'),
    'workflow/scripts/prepare_native_pilot.py','workflow/scripts/affinity_study_common.py',
    'workflow/scripts/prepare_abrank_transfer.py',
] + [str(p) for p in SEARCH_ROOT.glob('*/hits.tsv')]

rule all:
    input: f'{ROOT}/results_report.json',f'{ROOT}/findings.md',f'{ROOT}/bottleneck_audit.png',
           f'{ROOT}/lineage_review_v1/results.json'

rule audit:
    input:
        config=STUDY_CONFIG,
        implementation='workflow/scripts/audit_affinity_bottleneck.py',
        ledger=config['primary_ledger'],
        links=config['candidate_links'],
        skempi=config['skempi'],
        sabdab=config['sabdab_affinity'],
        roles=config['skempi_antibody_pairs'],
        processed=config['skempi_processed'],
        regions=config['region_cache'],
        verified=VERIFIED_INPUTS
    output: f'{ROOT}/results_report.json',f'{ROOT}/findings.md',f'{ROOT}/bottleneck_audit.png'
    threads: 1
    resources: mem_mb=12000
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/audit.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'

rule lineage_review:
    input:
        report=f'{ROOT}/results_report.json',
        implementation='workflow/scripts/review_affinity_lineage_bridges.py',
        config=LINEAGE_CONFIG,
        caches=LINEAGE['region_caches'],
        source=LINEAGE['source_records'],
        candidates=LINEAGE['candidate_links']
    output: f'{ROOT}/lineage_review_v1/results.json'
    conda: 'envs/openalphaseq_data.yaml'
    resources: mem_mb=12000
    log: f'{ROOT}/logs/lineage_review.log'
    shell: 'python {input.implementation:q} --config {input.config:q} > {log:q} 2>&1'
