"""Add budget/cohort-matched IgLM controls without rerunning MANGO."""
configfile: "config/asd_prototype_iglm.yaml"

ROOT = config['experiment_root']
MANGO = config['mango_root']
COHORT = config['cohort_dir']
PHASES = ['asd', 'sabdab2_scratch', 'sabdab2_finetuned']

wildcard_constraints:
    phase='|'.join(PHASES)


def records(w):
    dataset = 'asd' if w.phase == 'asd' else 'sabdab2'
    return f'{COHORT}/{dataset}_records.csv'


rule all:
    input:
        expand(f'{ROOT}/{{phase}}/eval.json', phase=PHASES),
        f'{MANGO}/plots/sabdab2_loss_comparison_with_iglm.png',
        f'{MANGO}/plots/sabdab2_loss_comparison_with_iglm.csv',


rule train_iglm:
    input:
        records=records,
        audit=f'{COHORT}/split_audit.json',
        mango_manifest=f'{MANGO}/experiment_manifest.json',
        initialization=lambda w: [f'{ROOT}/asd/checkpoints/best.pt'] if w.phase == 'sabdab2_finetuned' else [],
        implementation=['workflow/scripts/iglm_baseline.py', 'workflow/scripts/model_common.py',
                        'workflow/scripts/chain_examples.py', 'workflow/scripts/training_order.py'],
    output:
        ckpt=f'{ROOT}/{{phase}}/checkpoints/best.pt',
        latest=f'{ROOT}/{{phase}}/checkpoints/latest.pt',
        model_config=f'{ROOT}/{{phase}}/model_config.json',
        config=f'{ROOT}/{{phase}}/config.json',
        metrics=f'{ROOT}/{{phase}}/metrics.jsonl',
        curve=f'{ROOT}/{{phase}}/training_curve.csv',
        plot=f'{ROOT}/{{phase}}/training_curve.png',
        runtime=f'{ROOT}/{{phase}}/training_runtime.json',
    params:
        job='train', study=config,
    log:
        f'{ROOT}/logs/{{phase}}_train.log',
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/prototype_iglm_job.py'


rule evaluate_iglm:
    input:
        records=records,
        audit=f'{COHORT}/split_audit.json',
        coverage=f'{COHORT}/sabdab2_test_coverage.csv',
        mango_manifest=f'{MANGO}/experiment_manifest.json',
        ckpt=f'{ROOT}/{{phase}}/checkpoints/best.pt',
        model_config=f'{ROOT}/{{phase}}/model_config.json',
        implementation=['workflow/scripts/iglm_baseline.py', 'workflow/scripts/chain_examples.py'],
    output:
        evaluation=f'{ROOT}/{{phase}}/eval.json',
    params:
        job='evaluate', study=config,
    log:
        f'{ROOT}/logs/{{phase}}_eval.log',
    threads: 2
    resources:
        gpu=1, mem_mb=14000,
    conda:
        'envs/model.yaml'
    script:
        'scripts/prototype_iglm_job.py'


rule plot_comparison:
    input:
        evaluations=expand(f'{MANGO}/{{phase}}/{{objective}}/{{architecture}}/eval.json',
                           phase=['sabdab2_scratch', 'sabdab2_finetuned'],
                           objective=['cross_entropy', 'focal'], architecture=['mango', 'cross_attention']),
        iglm_evaluations=expand(f'{ROOT}/{{phase}}/eval.json',
                               phase=['sabdab2_scratch', 'sabdab2_finetuned']),
    output:
        figure=f'{MANGO}/plots/sabdab2_loss_comparison_with_iglm.png',
        data=f'{MANGO}/plots/sabdab2_loss_comparison_with_iglm.csv',
    log:
        f'{ROOT}/logs/plot_comparison.log',
    conda:
        'envs/analysis_plot.yaml'
    script:
        'scripts/analysis/plot_asd_loss_comparison.py'
