"""Four-arm exploratory objective study, serialized on one GPU."""
import os
STUDY_CONFIG=os.environ.get('MANGO_RANKING_PILOT_CONFIG','config/affinity_ranking_pilot.yaml')
configfile: STUDY_CONFIG
ROOT=config['experiment_root']
SEEDS=config['seeds']
ARMS=['IgLM_CE','MANGO_CE','IgLM_rank','MANGO_rank']

rule all:
    input: f'{ROOT}/results_report.json'

rule prepare:
    input: config=STUDY_CONFIG,implementation='workflow/scripts/affinity_ranking_pilot.py',ledger=config['primary_ledger']
    output: f'{ROOT}/cohort_audit.json',f'{ROOT}/training_protocol.json',f'{ROOT}/records.csv'
    conda: 'envs/openalphaseq_data.yaml'
    log: f'{ROOT}/logs/prepare.log'
    shell: 'python {input.implementation:q} prepare --config {input.config:q} > {log:q} 2>&1'

rule embeddings:
    input: audit=rules.prepare.output
    output: f'{ROOT}/embeddings_complete.json'
    params: root=lambda w:ROOT
    conda: 'envs/embed_esm2.yaml'
    resources: gpu=1,mem_mb=18000
    threads: 2
    log: f'{ROOT}/logs/embeddings.log'
    shell: 'python workflow/scripts/native_mechanism_embeddings.py --root {params.root:q} > {log:q} 2>&1'

rule pretrain:
    input: embeddings=rules.embeddings.output
    output: f'{ROOT}/pretraining/{{family}}_s{{seed}}/complete.json'
    wildcard_constraints: family='IgLM|MANGO'
    params: root=lambda w:ROOT
    conda: 'envs/model.yaml'
    resources: gpu=1,mem_mb=14000
    threads: 2
    log: f'{ROOT}/logs/pretrain_{{family}}_s{{seed}}.log'
    shell: 'python workflow/scripts/affinity_ranking_pilot.py fit --root {params.root:q} --family {wildcards.family:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule adapt:
    input: pretraining=lambda w:f'{ROOT}/pretraining/{w.arm.split("_")[0]}_s{w.seed}/complete.json'
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/complete.json'
    params: root=lambda w:ROOT,family=lambda w:w.arm.split('_')[0]
    conda: 'envs/model.yaml'
    resources: gpu=1,mem_mb=14000
    threads: 2
    log: f'{ROOT}/logs/adapt_{{arm}}_s{{seed}}.log'
    shell: 'python workflow/scripts/affinity_ranking_pilot.py fit --root {params.root:q} --family {params.family:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule score:
    input: fit=rules.adapt.output
    output: f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json'
    params: root=lambda w:ROOT
    conda: 'envs/model.yaml'
    resources: gpu=1,mem_mb=14000
    threads: 2
    log: f'{ROOT}/logs/score_{{arm}}_s{{seed}}.log'
    shell: 'python workflow/scripts/affinity_ranking_pilot.py score --root {params.root:q} --arm {wildcards.arm:q} --seed {wildcards.seed} > {log:q} 2>&1'

rule report:
    input: expand(f'{ROOT}/runs/{{arm}}_s{{seed}}/evaluation.json',arm=ARMS,seed=SEEDS)
    output: f'{ROOT}/results_report.json'
    params: root=lambda w:ROOT
    conda: 'envs/openalphaseq_data.yaml'
    resources: mem_mb=8000
    log: f'{ROOT}/logs/report.log'
    shell: 'python workflow/scripts/affinity_ranking_pilot.py report --root {params.root:q} > {log:q} 2>&1'
