# =============================================================================
# model.smk -- train / evaluate / predict / generate, one run per embedder.
# =============================================================================
# THE TASK: given the antigen and the LIGHT chain, predict the HEAVY chain.
# Every rule here consumes the same two conditioning inputs -- the antigen
# embedding for {tag} and the light-chain context embedding -- and produces or
# scores heavy chains. The heavy chain is never an input, so no rule has to
# remember to hide it.
#
# Run directory (the training contract):
#
#   runs/<tag>__<experiment_hash>/
#       config.json                full global state this run was built from
#       model_config.json          architecture + antigen dim (inference needs this)
#       metrics.jsonl              one JSON object per epoch
#       training_curve.csv         iteration-level train + validation losses
#       training_curve.png         incrementally refreshed learning curve
#       checkpoints/best.pt        lowest val loss
#       checkpoints/latest.pt      last epoch (resume point)
#       eval.json                  split NLL/perplexity  -> figure 1
#       predictions_test.csv       test-split reconstructions
#
# `experiment_hash` covers the dataset, filters, embedding choices, the
# embedder's own parameters and the model hyperparameters, so editing any of
# them yields a new run directory instead of silently overwriting a result.
#
# Rules carry a {run} wildcard and recover their embedder tag through
# tag_for_run(), which keeps one rule serving all 8 models.
#
# Retraining is gated by config["model"]["retrain"]:
#   false -> training inputs are ancient(), so upstream mtime changes never
#            retrigger it; the checkpoint is still built once if missing.
#            Force a fresh model with `snakemake train --forcerun train_model`.
#   true  -> normal dependencies; training reruns when its inputs change.

_RETRAIN = bool(config["model"].get("retrain", False))


def _maybe_ancient(paths):
    if _RETRAIN:
        return paths
    return [ancient(p) for p in paths]


def _train_records(w):
    return RECORDS_CSV if _RETRAIN else ancient(RECORDS_CSV)


def _train_embeddings(w):
    """Antigen + antibody embeddings for every train/val id of this run."""
    tag = tag_for_run(w.run)
    splits = list(config["model"]["train_splits"]) + list(config["model"]["val_splits"])
    return _maybe_ancient(
        antigen_embedding_dependencies(tag, splits)
        + antibody_embedding_dependencies(splits)
    )


rule train_model:
    input:
        records=_train_records,
        embeddings=_train_embeddings,
        embedder_config=lambda w: ancient(antigen_emb_config(tag_for_run(w.run))),
        plotter="workflow/scripts/training_curve.py",
    output:
        ckpt=f"{RUNS_DIR}/{{run}}/checkpoints/best.pt",
        latest=f"{RUNS_DIR}/{{run}}/checkpoints/latest.pt",
        model_config=f"{RUNS_DIR}/{{run}}/model_config.json",
        metrics=f"{RUNS_DIR}/{{run}}/metrics.jsonl",
        training_curve=f"{RUNS_DIR}/{{run}}/training_curve.csv",
        training_plot=f"{RUNS_DIR}/{{run}}/training_curve.png",
        run_config=f"{RUNS_DIR}/{{run}}/config.json",
    params:
        emb_dir=EMB_DIR,
        tag=lambda w: tag_for_run(w.run),
        antibody_dir=ANTIBODY_DIR,
        run_id=lambda w: w.run,
        experiment_hash=lambda w: experiment_hash(tag_for_run(w.run)),
        train_splits=config["model"]["train_splits"],
        val_splits=config["model"]["val_splits"],
        model_cfg=config["model"],
        global_state=config,
    log:
        f"{LOG_DIR}/train_{{run}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=24000,
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/train_mango.py"


def _eval_embeddings(w):
    """Both conditioning inputs for every split reported by evaluation."""
    tag = tag_for_run(w.run)
    splits = ["train", "val", "test"]
    return antigen_embedding_dependencies(tag, splits) + antibody_embedding_dependencies(splits)


rule evaluate_model:
    """Split NLL + perplexity for a trained run -- the data behind figure 1."""
    input:
        records=RECORDS_CSV,
        ckpt=f"{RUNS_DIR}/{{run}}/checkpoints/best.pt",
        model_config=f"{RUNS_DIR}/{{run}}/model_config.json",
        embeddings=_eval_embeddings,
    output:
        eval_json=f"{RUNS_DIR}/{{run}}/eval.json",
    params:
        emb_dir=EMB_DIR,
        tag=lambda w: tag_for_run(w.run),
        antibody_dir=ANTIBODY_DIR,
        splits=["train", "val", "test"],
    log:
        f"{LOG_DIR}/evaluate_{{run}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=24000,
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/evaluate_mango.py"


def _predict_embeddings(w):
    """Both conditioning inputs for every id the run predicts on."""
    splits = list(config["model"]["predict_splits"])
    return antigen_embedding_dependencies(tag_for_run(w.run), splits) + (
        antibody_embedding_dependencies(splits)
    )


rule predict_model:
    """Reconstruct heavy chains for the test split (sanity + recovery metrics)."""
    input:
        records=RECORDS_CSV,
        ckpt=f"{RUNS_DIR}/{{run}}/checkpoints/best.pt",
        model_config=f"{RUNS_DIR}/{{run}}/model_config.json",
        embeddings=_predict_embeddings,
    output:
        predictions=f"{RUNS_DIR}/{{run}}/predictions_test.csv",
    params:
        emb_dir=EMB_DIR,
        tag=lambda w: tag_for_run(w.run),
        antibody_dir=ANTIBODY_DIR,
        predict_splits=config["model"]["predict_splits"],
        gen_cfg=config["generation"],
    log:
        f"{LOG_DIR}/predict_{{run}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=24000,
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/predict_mango.py"


rule generate_designs:
    """Aim 2: de novo heavy chains against one held-out structure.

    One job per (run, structure) so the most expensive step in the pipeline
    parallelises and a single failed target does not cost the whole sweep. The
    structure comes from `generation.source` -- the test split by default.
    """
    input:
        ckpt=f"{RUNS_DIR}/{{run}}/checkpoints/best.pt",
        model_config=f"{RUNS_DIR}/{{run}}/model_config.json",
        records=records_for_split(GENERATION_SPLIT),
        antigen_ready=lambda w: (
            antigen_batch_marker(tag_for_run(w.run)) if BATCH_EMBEDDINGS
            else generation_antigen_emb(tag_for_run(w.run), w.instance)
        ),
        antibody_ready=lambda w: (
            antibody_batch_marker() if BATCH_EMBEDDINGS
            else generation_antibody_emb(w.instance)
        ),
    output:
        designs=f"{DESIGNS_DIR}/{{run}}/{{instance}}/designs.csv",
    params:
        tag=lambda w: tag_for_run(w.run),
        split=GENERATION_SPLIT,
        gen_cfg=config["generation"],
        seed=config["experiment"]["seed"],
        antigen_emb=lambda w: generation_antigen_emb(tag_for_run(w.run), w.instance),
        antibody_emb=lambda w: generation_antibody_emb(w.instance),
    log:
        f"{LOG_DIR}/generate_{{run}}_{{instance}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=24000,
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/generate_designs.py"


# --- aggregation targets -----------------------------------------------------
rule train:
    input:
        [run_ckpt(t) for t in ACTIVE_EMBEDDERS],


rule evaluate:
    input:
        [eval_json(t) for t in ACTIVE_EMBEDDERS],


rule predict:
    input:
        [predictions_csv(t) for t in ACTIVE_EMBEDDERS],


# An input FUNCTION, not a list: the generation set comes from the process
# checkpoint, so it must not be resolved while the workflow is being parsed.
rule generate:
    input:
        lambda w: [
            designs_csv(t, i)
            for t in ACTIVE_EMBEDDERS
            for i in generation_instances()
        ],


# The first completion milestone: train the active baseline and run both
# quantitative evaluation and held-out heavy-chain reconstruction. De novo
# generation remains an explicit target because it can produce thousands of
# sequences and should not happen accidentally on a default invocation.
rule inference:
    input:
        [eval_json(t) for t in ACTIVE_EMBEDDERS],
        [predictions_csv(t) for t in ACTIVE_EMBEDDERS],
