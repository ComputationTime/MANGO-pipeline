"""Test-split heavy-chain reconstruction -> predictions CSV.

For each test structure, generates a heavy chain from the precomputed antigen
embedding plus the light-chain context embedding -- the model's full and only
conditioning signal -- and records it alongside the true heavy chain. The true
heavy chain is read for comparison only; it is never fed to the model, and the
context embedding was built with the heavy slot masked, so there is no leak.

This is the reconstruction sanity check; Aim 2's de novo design lives in
generate_designs.py.
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

COLUMNS = [
    "embedder", "run_id", "status", "split", "id", "pdb_path", "antigen_chains",
    "light_seq", "true_heavy_seq", "predicted_heavy_seq", "prediction_length",
    "checkpoint",
]


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mc.MangoModel(
        d_ag=int(ckpt["d_ag"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
    )
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    model.lm.load_state_dict(ckpt["lm"])
    return model.to(device).eval(), ckpt


def predict(records_csv, emb_dir, tag, antibody_dir, ckpt_path, model_config_path,
            predict_splits, gen_cfg, out_csv):
    device = dc.get_device(f"prediction {tag}")
    model, ckpt = _load_model(ckpt_path, device)
    with open(model_config_path) as fh:
        run_id = json.load(fh).get("run_id", "")

    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    df = df[df["split"].isin(set(predict_splits))]
    print(f"[{tag}] predicting {len(df)} structures", flush=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in df.itertuples(index=False):
            base = {
                "embedder": tag,
                "run_id": run_id,
                "split": row.split,
                "id": row.id,
                "pdb_path": row.pdb_path,
                "antigen_chains": row.antigen_chains,
                "light_seq": row.resolved_L_seq,
                "true_heavy_seq": row.resolved_H_seq,
                "checkpoint": ckpt_path,
            }
            try:
                ag_path = Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt"
                ctx_path = (
                    Path(emb_dir) / "antibody" / antibody_dir / row.split / f"{row.id}.pt"
                )
                h_ag = mc.load_embedding(str(ag_path)).to(device)
                x_ctx = mc.load_embedding(str(ctx_path)).to(device)
                ids = model.generate_heavy(
                    x_ctx,
                    h_ag,
                    max_new_tokens=int(gen_cfg["max_new_tokens"]),
                    do_sample=bool(gen_cfg.get("do_sample", True)),
                    top_p=float(gen_cfg.get("top_p", 1.0)),
                    temperature=float(gen_cfg.get("temperature", 1.0)),
                )
                pred = model.decode_heavy(ids)
                base.update(
                    status="ok", predicted_heavy_seq=pred, prediction_length=len(pred)
                )
                n_ok += 1
            except Exception as e:  # keep going; record the failure per-structure
                base.update(
                    status=f"error: {type(e).__name__}: {e}",
                    predicted_heavy_seq="",
                    prediction_length=0,
                )
            writer.writerow(base)

    print(f"[{tag}] wrote {n_ok}/{len(df)} predictions -> {out_csv}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        predict(
            records_csv=smk.input.records,
            emb_dir=smk.params.emb_dir,
            tag=smk.params.tag,
            antibody_dir=smk.params.antibody_dir,
            ckpt_path=smk.input.ckpt,
            model_config_path=smk.input.model_config,
            predict_splits=list(smk.params.predict_splits),
            gen_cfg=dict(smk.params.gen_cfg),
            out_csv=smk.output.predictions,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--antibody-dir", default="ablang2_light_only")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--predict-splits", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--gen-cfg", default="{}")
    a = p.parse_args()
    gen = {"max_new_tokens": 130, "do_sample": True, "top_p": 1.0, "temperature": 1.0}
    gen.update(json.loads(a.gen_cfg))
    predict(
        a.records, a.emb_dir, a.tag, a.antibody_dir, a.ckpt, a.model_config,
        a.predict_splits.split(","), gen, a.out,
    )


if __name__ == "__main__":
    main()
