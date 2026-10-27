"""A separate, antigen-free IgLM architecture baseline trained from scratch."""

from iglm_baseline import experiment_hash as iglm_experiment_hash

_IGLM_SPEC = config.get("iglm_baseline", {})
_IGLM_HASH = iglm_experiment_hash(config) if _IGLM_SPEC else "unconfigured"
_IGLM_RUN = f"{RUNS_DIR}/iglm_scratch__{_IGLM_HASH}"
IGLM_EVAL = f"{_IGLM_RUN}/eval.json"


def _iglm_records(w):
    if not _IGLM_SPEC:
        raise ValueError("load config/iglm_baseline.yaml for scratch IgLM training")
    return RECORDS_CSV


rule train_iglm_baseline:
    input:
        records=_iglm_records,
        implementation="workflow/scripts/iglm_baseline.py",
        training="workflow/scripts/train_mango.py",
        ordering="workflow/scripts/training_order.py",
        plotter="workflow/scripts/training_curve.py",
    output:
        checkpoint=f"{_IGLM_RUN}/checkpoints/best.pt",
        latest=f"{_IGLM_RUN}/checkpoints/latest.pt",
        model_config=f"{_IGLM_RUN}/model_config.json",
        run_config=f"{_IGLM_RUN}/config.json",
        metrics=f"{_IGLM_RUN}/metrics.jsonl",
        curve=f"{_IGLM_RUN}/training_curve.csv",
        plot=f"{_IGLM_RUN}/training_curve.png",
        runtime=f"{_IGLM_RUN}/training_runtime.json",
    params:
        action="train", model_cfg=config["model"], spec=_IGLM_SPEC,
        run_dir=_IGLM_RUN, run_hash=_IGLM_HASH, global_state=config,
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    log:
        f"{LOG_DIR}/train_iglm_scratch__{_IGLM_HASH}.log",
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/iglm_baseline.py"


rule evaluate_iglm_baseline:
    input:
        records=_iglm_records,
        checkpoint=rules.train_iglm_baseline.output.checkpoint,
    output:
        evaluation=IGLM_EVAL,
    params:
        action="evaluate", splits=["train", "val", "test"],
    threads: 2
    resources:
        gpu=1, mem_mb=16000,
    log:
        f"{LOG_DIR}/evaluate_iglm_scratch__{_IGLM_HASH}.log",
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/iglm_baseline.py"


rule iglm_baseline:
    input:
        IGLM_EVAL,
