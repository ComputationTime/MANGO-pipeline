"""Train antigen + chain-type conditioned MANGO on heavy and light targets.

Each selected chain is a separate example after the group-aware split. Antibody
sequences are decoder targets only; no antibody embeddings are loaded.
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc
import device_common as dc
from training_curve import write_training_plot
from training_order import epoch_training_rows


def normalize_gradients(parameters, n_tokens):
    """Convert accumulated token-sum gradients to token-mean gradients."""
    if n_tokens <= 0:
        raise ValueError("Gradient window must contain target tokens")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(n_tokens)


def _rows(records_csv, splits, target_chains):
    from types import SimpleNamespace
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    df = df[df["split"].isin(set(splits))]
    examples = []
    for row in df.itertuples(index=False):
        for chain in target_chains:
            sequence = getattr(row, "resolved_H_seq" if chain == "heavy" else "resolved_L_seq")
            if not sequence:
                raise ValueError(f"{row.id}: missing {chain} target sequence")
            examples.append(SimpleNamespace(id=row.id, split=row.split,
                                           chain=chain, sequence=sequence))
    return examples


def _path(emb_dir, tag, row):
    return str(Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt")


def measure_training(function):
    """Write training-only wall clock and VRAM alongside the run artifacts."""
    import functools
    import inspect
    import os
    import time
    from vram_monitor import VramMonitor

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        output = Path(bound.arguments["out_model_config"]).parent / "training_runtime.json"
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        monitor = VramMonitor(os.getpid())
        monitor.start()
        started = time.monotonic()
        status = "failed"
        try:
            value = function(*args, **kwargs)
            status = "complete"
            return value
        finally:
            if gpu:
                torch.cuda.synchronize()
            result = {"status": status, "embedder": bound.arguments["tag"],
                      "wall_seconds": time.monotonic() - started,
                      "scope": "Training function: validation, checkpoint and live-curve writes included; embedding, installation and downloads excluded",
                      "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if gpu else 0,
                      "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if gpu else 0,
                      **monitor.finish()}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    return wrapped


@measure_training
def train(
    records_csv,
    emb_dir,
    tag,
    train_splits,
    val_splits,
    model_cfg,
    out_ckpt,
    out_latest,
    out_model_config,
    out_metrics,
    out_training_curve,
    out_training_plot,
    out_run_config,
    run_id,
    experiment_hash,
    global_state,
):
    cohort_sha256 = mc.comparison_cohort(records_csv)
    device = dc.get_device(f"training {tag}")
    torch.manual_seed(int(model_cfg.get("seed", 0)))

    target_chains = tuple(mc.MangoModel.normalize_chain(c) for c in model_cfg.get("target_chains", ["heavy", "light"]))
    train_rows = _rows(records_csv, train_splits, target_chains)
    val_rows = _rows(records_csv, val_splits, target_chains)
    if set(train_splits) & set(val_splits):
        raise ValueError("Training and validation splits overlap")
    if not val_rows:
        raise ValueError("Validation cohort is required for checkpoint selection")
    if not train_rows:
        raise RuntimeError(f"no training rows for splits {train_splits}")
    print(f"[{tag}] train={len(train_rows)} val={len(val_rows)} device={device}", flush=True)

    # Antigen dim is read from a real embedding, so a mis-declared H is impossible.
    sample_ag = _path(emb_dir, tag, train_rows[0])
    d_ag = mc.load_embedding(sample_ag).shape[-1]
    print(f"[{tag}] antigen dim d_Ag_rep={d_ag}", flush=True)

    architecture = model_cfg.get("architecture", mc.MangoModel.ARCHITECTURE)
    if architecture != mc.MangoModel.ARCHITECTURE:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    model = mc.MangoModel(d_ag=d_ag, target_chains=target_chains).to(device)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(model_cfg["lr"]),
    )
    grad_accum = max(1, int(model_cfg.get("grad_accum", 1)))
    patience = int(model_cfg.get("patience", 3))
    seed = int(model_cfg.get("seed", 0))
    shuffle_train = bool(model_cfg.get("shuffle_train", True))

    for path in (
        out_ckpt, out_latest, out_model_config, out_metrics,
        out_training_curve, out_training_plot, out_run_config,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Written up front so an interrupted run still records what it was.
    with open(out_run_config, "w") as fh:
        json.dump(
            {"run_id": run_id, "experiment_hash": experiment_hash,
             "embedder": tag, "global_state": global_state},
            fh, indent=2, sort_keys=True, default=str,
        )
        fh.write("\n")

    model_config = {
        "cohort_sha256": cohort_sha256,
        "run_id": run_id,
        "experiment_hash": experiment_hash,
        "embedder": tag,
        # Recorded so a checkpoint can never be misread as a different task.
        "loss_reduction": "target_token_mean_including_eos",
        "target_contract": "imgt_variable_vh_vl_v1",
        "iglm_reference_parameters": 12889600,
        "task": model.TASK,
        "target_chains": list(model.target_chains),
        "chain_selector_trainable_parameters": model.chain_embedding.weight.numel(),
        "d_ag": int(d_ag),
        "d_model": int(model.d_model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "decoder_trainable_parameters": sum(p.numel() for p in model.lm.parameters() if p.requires_grad),
        "projection_trainable_parameters": sum(p.numel() for p in model.antigen_projection.parameters() if p.requires_grad),
        "architecture": model.ARCHITECTURE,
        "vocab_size": int(model.configs.vocab_size),
        "n_layer": int(model.configs.n_layer),
        "n_head": int(model.configs.n_head),
        "n_positions": int(model.configs.n_positions),
    }
    with open(out_model_config, "w") as fh:
        json.dump(model_config, fh, indent=2, sort_keys=True)
        fh.write("\n")

    history = []
    global_iteration = 0
    plot_interval = max(1, int(model_cfg.get("loss_plot_interval", 250)))

    curve_fh = open(out_training_curve, "w", newline="")
    curve_writer = csv.DictWriter(
        curve_fh, fieldnames=["iteration", "epoch", "phase", "loss", "record_id", "chain_type"]
    )
    curve_writer.writeheader()
    curve_fh.flush()

    def record_point(iteration, epoch, phase, loss, record_id="", chain_type=""):
        point = {
            "iteration": iteration,
            "epoch": epoch,
            "phase": phase,
            "loss": float(loss),
            "record_id": record_id,
            "chain_type": chain_type,
        }
        history.append(point)
        curve_writer.writerow(point)
        curve_fh.flush()

    def run_epoch(rows, train_mode, epoch):
        nonlocal global_iteration
        model.train(train_mode)
        total = 0.0
        total_tokens = 0
        window_tokens = 0
        if train_mode:
            opt.zero_grad()
        for i, row in enumerate(rows):
            h_ag = mc.load_embedding(_path(emb_dir, tag, row)).to(device)
            target_ids = model.target_token_ids(row.sequence).to(device)
            with torch.set_grad_enabled(train_mode):
                loss = model.loss(h_ag, row.chain, target_ids)
            n_tokens = model.n_target_tokens(target_ids)
            if train_mode:
                (loss * n_tokens).backward()
                window_tokens += n_tokens
                if (i + 1) % grad_accum == 0 or (i + 1) == len(rows):
                    normalize_gradients(model.parameters(), window_tokens)
                    opt.step()
                    window_tokens = 0
                    opt.zero_grad()
            loss_value = float(loss.item())
            total += loss_value * n_tokens
            total_tokens += n_tokens
            if train_mode:
                global_iteration += 1
                record_point(global_iteration, epoch, "train", loss_value, row.id, row.chain)
                if global_iteration % plot_interval == 0:
                    write_training_plot(history, out_training_plot, run_id)
        return total / total_tokens

    def checkpoint(path, epoch, val_loss):
        torch.save(
            {
                "cohort_sha256": cohort_sha256,
                "architecture": model.ARCHITECTURE,
                "task": model.TASK,
                "target_chains": list(model.target_chains),
                "chain_embedding": model.chain_embedding.state_dict(),
                "antigen_projection": model.antigen_projection.state_dict(),
                "lm": model.lm.state_dict(),
                "optimizer": opt.state_dict(),
                "d_ag": int(d_ag),
                "embedder": tag,
                "run_id": run_id,
                "experiment_hash": experiment_hash,
                "val_loss": val_loss,
                "epoch": epoch,
            },
            path,
        )

    best = float("inf")
    since_improved = 0
    try:
        with open(out_metrics, "w") as metrics_fh:
            for epoch in range(int(model_cfg["epochs"])):
                epoch_rows = epoch_training_rows(
                    train_rows, seed=seed, epoch=epoch, shuffle=shuffle_train
                )
                tr = run_epoch(epoch_rows, True, epoch)
                va = run_epoch(val_rows, False, epoch) if val_rows else tr
                record_point(global_iteration, epoch, "train_epoch", tr)
                record_point(global_iteration, epoch, "validation", va)
                write_training_plot(history, out_training_plot, run_id)
                improved = va < best

                metrics_fh.write(
                    json.dumps(
                        {"epoch": epoch, "iteration": global_iteration,
                         "train_loss": tr, "val_loss": va,
                         "train_order_seed": seed + epoch,
                         "train_shuffled": shuffle_train,
                         "embedder": tag, "run_id": run_id, "improved": improved}
                    )
                    + "\n"
                )
                metrics_fh.flush()
                print(
                    f"[{tag}] iteration {global_iteration}: "
                    f"train_loss={tr:.4f} val_loss={va:.4f}"
                    f"{' *' if improved else ''}",
                    flush=True,
                )

                checkpoint(out_latest, epoch, va)
                if improved:
                    best = va
                    since_improved = 0
                    checkpoint(out_ckpt, epoch, best)
                else:
                    since_improved += 1
                    if since_improved >= patience:
                        print(
                            f"[{tag}] early stop at iteration {global_iteration} "
                            f"(best {best:.4f})", flush=True,
                        )
                        break
    finally:
        curve_fh.close()

    if not Path(out_ckpt).is_file():
        raise RuntimeError(
            "no best checkpoint written -- epochs may be 0, or validation never "
            "improved on the initial value"
        )
    print(f"[{tag}] best val_loss={best:.4f} -> {out_ckpt}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        train(
            records_csv=smk.input.records,
            emb_dir=smk.params.emb_dir,
            tag=smk.params.tag,
            train_splits=list(smk.params.train_splits),
            val_splits=list(smk.params.val_splits),
            model_cfg=dict(smk.params.model_cfg),
            out_ckpt=smk.output.ckpt,
            out_latest=smk.output.latest,
            out_model_config=smk.output.model_config,
            out_metrics=smk.output.metrics,
            out_training_curve=smk.output.training_curve,
            out_training_plot=smk.output.training_plot,
            out_run_config=smk.output.run_config,
            run_id=smk.params.run_id,
            experiment_hash=smk.params.experiment_hash,
            global_state=dict(smk.params.global_state),
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--train-splits", default="train")
    p.add_argument("--val-splits", default="val")
    p.add_argument("--model-cfg", default="{}", help="JSON of model config overrides")
    a = p.parse_args()

    defaults = {
        "architecture": mc.MangoModel.ARCHITECTURE, "epochs": 5,
        "lr": 1e-4, "patience": 3, "grad_accum": 1, "seed": 0,
        "shuffle_train": True,
    }
    defaults.update(json.loads(a.model_cfg))
    rd = Path(a.run_dir)
    train(
        a.records, a.emb_dir, a.tag,
        a.train_splits.split(","), a.val_splits.split(","), defaults,
        str(rd / "checkpoints" / "best.pt"), str(rd / "checkpoints" / "latest.pt"),
        str(rd / "model_config.json"), str(rd / "metrics.jsonl"),
        str(rd / "training_curve.csv"), str(rd / "training_curve.png"),
        str(rd / "config.json"), rd.name, "manual", {},
    )


if __name__ == "__main__":
    main()
