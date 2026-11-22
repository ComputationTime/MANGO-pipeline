#!/usr/bin/env python
"""Master training script: the whole training regimen, one model per embedder.

    python run_training.py --all                       # every embedder
    python run_training.py --embedders one_hot esm2    # a chosen subset
    python run_training.py --all --stages train evaluate

For every selected embedder it runs, in order (``--stages`` picks a subset):

  train      fit MANGO on the train split, early-stop on val   training/train_mango.py
  evaluate   split NLL / perplexity -> eval.json (Figure 1)    training/evaluate_mango.py
  predict    reconstruct the held-out test chains              training/predict_mango.py
  generate   de novo designs against one held-out target per
             test cluster (Figures 3-6 inputs)                 training/generate_designs.py

Embeddings (``run_embedding.py``) and the dataset (``build_dataset.py``) must
exist first. A run directory is ``artifacts/runs/<tag>__<hash>/`` where the hash
covers the dataset, processing, embedder and model config, so changing any of
them gives a new run instead of overwriting a result. An existing best
checkpoint is reused unless ``--retrain`` is given; existing design tables are
reused unless ``--regenerate`` is given. A failure in one embedder is reported
and the remaining embedders still run.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS)] + [
    str(SCRIPTS / d) for d in ("lib", "data", "embedding", "training")
]

import pipeline as pl  # noqa: E402

STAGES = ["train", "evaluate", "predict", "generate"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pl.add_common_args(parser)
    pl.add_embedder_args(parser)
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=STAGES,
        help="stages to run (default: all four, in order)",
    )
    parser.add_argument("--retrain", action="store_true",
                        help="retrain even if a best checkpoint already exists")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate designs even if designs.csv exists")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override model.epochs (e.g. for a quick check)")
    parser.add_argument("--chain", choices=["heavy", "light"], default=None,
                        help="override generation.chain")
    return parser.parse_args()


def _paths(pipe, tag):
    run = pipe.run_dir(tag)
    return {
        "run": run,
        "ckpt": run / "checkpoints" / "best.pt",
        "latest": run / "checkpoints" / "latest.pt",
        "model_config": run / "model_config.json",
        "metrics": run / "metrics.jsonl",
        "curve_csv": run / "training_curve.csv",
        "curve_png": run / "training_curve.png",
        "run_config": run / "config.json",
        "eval": run / "eval.json",
        "predictions": run / "predictions_test.csv",
    }


def stage_train(pipe, tag, p, retrain):
    import train_mango

    if p["ckpt"].is_file() and not (retrain or pipe.config["model"].get("retrain")):
        print(f"[{tag}] reusing checkpoint {p['ckpt']}", flush=True)
        return
    cfg = pipe.config
    train_mango.train(
        records_csv=str(pipe.records_csv),
        emb_dir=str(pipe.emb_dir),
        tag=tag,
        train_splits=list(cfg["model"]["train_splits"]),
        val_splits=list(cfg["model"]["val_splits"]),
        model_cfg=pipe.model_config(tag),
        out_ckpt=str(p["ckpt"]),
        out_latest=str(p["latest"]),
        out_model_config=str(p["model_config"]),
        out_metrics=str(p["metrics"]),
        out_training_curve=str(p["curve_csv"]),
        out_training_plot=str(p["curve_png"]),
        out_run_config=str(p["run_config"]),
        run_id=pipe.run_id(tag),
        experiment_hash=pipe.experiment_hash(tag),
        # Record one run identically whether trained alone or with others.
        global_state={**cfg, "active_embedders": [tag]},
        embedding_tag=pipe.embedding_tag(tag),
    )


def stage_evaluate(pipe, tag, p):
    import evaluate_mango

    evaluate_mango.evaluate(
        str(pipe.records_csv), str(pipe.emb_dir), tag,
        str(p["ckpt"]), str(p["model_config"]),
        ["train", "val", "test"], str(p["eval"]),
    )


def stage_predict(pipe, tag, p):
    import predict_mango

    cfg = pipe.config
    predict_mango.predict(
        str(pipe.records_csv), str(pipe.emb_dir), tag,
        str(p["ckpt"]), str(p["model_config"]),
        list(cfg["model"]["predict_splits"]), dict(cfg["generation"]),
        str(p["predictions"]),
    )


def stage_generate(pipe, tag, p, regenerate):
    import generate_designs

    cfg = pipe.config
    gen = cfg["generation"]
    emb_tag = pipe.embedding_tag(tag)
    instances = pipe.generation_instances()
    print(f"[{tag}] generating for {len(instances)} held-out targets", flush=True)
    for instance in instances:
        out = pipe.designs_csv(tag, instance)
        if out.is_file() and not regenerate:
            continue
        generate_designs.generate(
            ckpt_path=str(p["ckpt"]),
            model_config_path=str(p["model_config"]),
            records_csv=str(pipe.records_csv),
            antigen_emb_path=str(
                pipe.emb_dir / "antigen" / emb_tag / gen["source"] / f"{instance}.pt"
            ),
            target_id=instance,
            split=gen["source"],
            tag=tag,
            gen_cfg=dict(gen),
            seed=cfg["experiment"]["seed"],
            out_csv=str(out),
        )


def main():
    args = parse_args()
    config = pl.load_config(*args.config)
    if args.epochs is not None:
        config["model"]["epochs"] = args.epochs
    if args.chain is not None:
        config["generation"]["chain"] = args.chain
    pipe = pl.Pipeline(config, args.artifact_root)
    tags = pipe.select_embedders(args)
    pl.require_file(pipe.records_csv, "run `python build_dataset.py` first")
    stages = [s for s in STAGES if s in args.stages]

    failures = {}
    for tag in tags:
        print(f"\n=== {tag} ({pipe.run_id(tag)}) ===", flush=True)
        started = time.time()
        p = _paths(pipe, tag)
        try:
            for stage in stages:
                print(f"[{tag}] stage: {stage}", flush=True)
                if stage == "train":
                    stage_train(pipe, tag, p, args.retrain)
                    continue
                pl.require_file(p["ckpt"], "run the train stage first")
                if stage == "evaluate":
                    stage_evaluate(pipe, tag, p)
                elif stage == "predict":
                    stage_predict(pipe, tag, p)
                elif stage == "generate":
                    stage_generate(pipe, tag, p, args.regenerate)
            print(f"[{tag}] finished in {time.time() - started:.0f}s", flush=True)
        except (Exception, SystemExit) as exc:  # keep the other embedders going
            traceback.print_exc()
            failures[tag] = f"{type(exc).__name__}: {exc}"

    summary = {
        tag: {"run_id": pipe.run_id(tag), "status": failures.get(tag, "ok")}
        for tag in tags
    }
    pipe.runs_dir.mkdir(parents=True, exist_ok=True)
    (pipe.runs_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("\n=== training summary ===")
    for tag, info in summary.items():
        print(f"{tag:<16} {info['run_id']:<24} {info['status']}")
    if failures:
        raise SystemExit(f"{len(failures)} embedder(s) failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
