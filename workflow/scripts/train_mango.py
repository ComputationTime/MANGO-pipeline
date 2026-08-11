"""Train MANGO (cross-attention + GPT2 head) on precomputed embeddings.

Teacher-forced causal LM over the HEAVY chain, conditioned on the antigen
embedding and the light-chain context embedding. Loss is computed on heavy
tokens only (see model_common for the exact layout), so the reported numbers
are heavy-chain NLLs and the heavy chain never enters the input.

Optimises on train, early-stops on val, and never reads test. Every model in the
study is identical except for the antigen embedding it consumes -- that is the
whole experiment.

Writes the full run directory (see workflow/rules/model.smk for the contract),
including an iteration-level ``training_curve.csv`` and an atomically refreshed
``training_curve.png`` that can be viewed while the job is still running.
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


def _rows(records_csv, splits):
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    df = df[df["split"].isin(set(splits))]
    return list(
        df[["id", "split", "resolved_H_seq", "resolved_L_seq"]].itertuples(index=False)
    )


def _paths(emb_dir, tag, antibody_dir, row):
    ag = Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt"
    ab = Path(emb_dir) / "antibody" / antibody_dir / row.split / f"{row.id}.pt"
    return str(ag), str(ab)


def train(
    records_csv,
    emb_dir,
    tag,
    antibody_dir,
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
    device = dc.get_device(f"training {tag}")
    torch.manual_seed(int(model_cfg.get("seed", 0)))

    train_rows = _rows(records_csv, train_splits)
    val_rows = _rows(records_csv, val_splits)
    if not train_rows:
        raise RuntimeError(f"no training rows for splits {train_splits}")
    print(f"[{tag}] train={len(train_rows)} val={len(val_rows)} device={device}", flush=True)

    # Antigen dim is read from a real embedding, so a mis-declared H is impossible.
    sample_ag, _ = _paths(emb_dir, tag, antibody_dir, train_rows[0])
    d_ag = mc.load_embedding(sample_ag).shape[-1]
    print(f"[{tag}] antigen dim d_Ag_rep={d_ag}", flush=True)

    n_heads = int(model_cfg["n_cross_attn_heads"])
    n_layers = int(model_cfg["n_cross_attn_layers"])
    model = mc.MangoModel(d_ag=d_ag, n_heads=n_heads, n_layers=n_layers).to(device)

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
        "run_id": run_id,
        "experiment_hash": experiment_hash,
        "embedder": tag,
        "antibody_embedder": antibody_dir,
        # Recorded so a checkpoint can never be misread as a different task.
        "task": "antigen+light->heavy",
        "d_ag": int(d_ag),
        "d_model": int(model.d_model),
        "n_cross_attn_heads": n_heads,
        "n_cross_attn_layers": n_layers,
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
        curve_fh, fieldnames=["iteration", "epoch", "phase", "loss", "record_id"]
    )
    curve_writer.writeheader()
    curve_fh.flush()

    def record_point(iteration, epoch, phase, loss, record_id=""):
        point = {
            "iteration": iteration,
            "epoch": epoch,
            "phase": phase,
            "loss": float(loss),
            "record_id": record_id,
        }
        history.append(point)
        curve_writer.writerow(point)
        curve_fh.flush()

    def run_epoch(rows, train_mode, epoch):
        nonlocal global_iteration
        model.train(train_mode)
        total = 0.0
        if train_mode:
            opt.zero_grad()
        for i, row in enumerate(rows):
            ag_path, ab_path = _paths(emb_dir, tag, antibody_dir, row)
            h_ag = mc.load_embedding(ag_path).to(device)
            # Light-chain context only; the heavy chain appears solely as labels.
            x_ctx = mc.load_embedding(ab_path).to(device)
            heavy_ids = model.heavy_token_ids(row.resolved_H_seq).to(device)
            with torch.set_grad_enabled(train_mode):
                loss = model.loss(x_ctx, h_ag, heavy_ids)
            if train_mode:
                (loss / grad_accum).backward()
                if (i + 1) % grad_accum == 0 or (i + 1) == len(rows):
                    opt.step()
                    opt.zero_grad()
            loss_value = float(loss.item())
            total += loss_value
            if train_mode:
                global_iteration += 1
                record_point(global_iteration, epoch, "train", loss_value, row.id)
                if global_iteration % plot_interval == 0:
                    write_training_plot(history, out_training_plot, run_id)
        return total / max(1, len(rows))

    def checkpoint(path, epoch, val_loss):
        torch.save(
            {
                "cross_attn": model.cross_attn.state_dict(),
                "lm": model.lm.state_dict(),
                "optimizer": opt.state_dict(),
                "d_ag": int(d_ag),
                "n_heads": n_heads,
                "n_layers": n_layers,
                "embedder": tag,
                "antibody_embedder": antibody_dir,
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
            antibody_dir=smk.params.antibody_dir,
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
    p.add_argument("--antibody-dir", default="ablang2_light_only")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--train-splits", default="train")
    p.add_argument("--val-splits", default="val")
    p.add_argument("--model-cfg", default="{}", help="JSON of model config overrides")
    a = p.parse_args()

    defaults = {
        "n_cross_attn_heads": 1, "n_cross_attn_layers": 1, "epochs": 5,
        "lr": 1e-4, "patience": 3, "grad_accum": 1, "seed": 0,
        "shuffle_train": True,
    }
    defaults.update(json.loads(a.model_cfg))
    rd = Path(a.run_dir)
    train(
        a.records, a.emb_dir, a.tag, a.antibody_dir,
        a.train_splits.split(","), a.val_splits.split(","), defaults,
        str(rd / "checkpoints" / "best.pt"), str(rd / "checkpoints" / "latest.pt"),
        str(rd / "model_config.json"), str(rd / "metrics.jsonl"),
        str(rd / "training_curve.csv"), str(rd / "training_curve.png"),
        str(rd / "config.json"), rd.name, "manual", {},
    )


if __name__ == "__main__":
    main()
