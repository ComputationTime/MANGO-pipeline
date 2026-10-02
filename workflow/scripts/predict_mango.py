"""Held-out requested-chain reconstruction from antigen embeddings only.

Reference antibody sequences are output comparisons, never model inputs.
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
    "chain_type", "true_sequence", "predicted_sequence",
    "light_seq", "true_heavy_seq", "predicted_heavy_seq", "prediction_length",
    "checkpoint",
]


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mc.MangoModel.from_checkpoint(ckpt)
    return model.to(device).eval(), ckpt


def predict(records_csv, emb_dir, tag, ckpt_path, model_config_path,
            predict_splits, gen_cfg, out_csv):
    device = dc.get_device(f"prediction {tag}")
    model, ckpt = _load_model(ckpt_path, device)
    chain = model.normalize_chain(gen_cfg.get("chain", "heavy"))
    if chain not in model.target_chains:
        raise ValueError(f"Checkpoint was not trained for {chain} chains")
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
                "chain_type": chain,
                "true_sequence": row.resolved_H_seq if chain == "heavy" else row.resolved_L_seq,
            }
            try:
                ag_path = Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt"
                h_ag = mc.load_embedding(str(ag_path)).to(device)
                ids = model.generate_chain(
                    h_ag, chain,
                    max_new_tokens=int(gen_cfg["max_new_tokens"]),
                    do_sample=bool(gen_cfg.get("do_sample", True)),
                    top_p=float(gen_cfg.get("top_p", 1.0)),
                    temperature=float(gen_cfg.get("temperature", 1.0)),
                )
                pred = model.decode_chain(ids)
                base.update(
                    status="ok", predicted_sequence=pred,
                    predicted_heavy_seq=pred if chain == "heavy" else "", prediction_length=len(pred)
                )
                n_ok += 1
            except Exception as e:  # keep going; record the failure per-structure
                base.update(
                    status=f"error: {type(e).__name__}: {e}",
                    predicted_sequence="", predicted_heavy_seq="",
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--predict-splits", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--gen-cfg", default="{}")
    p.add_argument("--chain", choices=["heavy", "light", "H", "L"], default="heavy")
    a = p.parse_args()
    gen = {"max_new_tokens": 130, "do_sample": True, "top_p": 1.0, "temperature": 1.0}
    gen.update(json.loads(a.gen_cfg))
    gen["chain"] = a.chain
    predict(
        a.records, a.emb_dir, a.tag, a.ckpt, a.model_config,
        a.predict_splits.split(","), gen, a.out,
    )


if __name__ == "__main__":
    main()
