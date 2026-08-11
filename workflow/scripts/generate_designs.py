"""Aim 2: de novo heavy-chain designs against one held-out structure.

Generates `n_per_target` heavy chains conditioned on that structure's antigen
embedding and its light-chain context embedding -- the same pair the model was
trained on, so sampling matches the trained conditional. Only heavy chains are
ever produced: that is the task.

The structure comes from whichever dataset split `generation.source` names. It
is the test split by default, so designs are made against held-out complexes
from the dataset itself. External therapeutic panels are deliberately deferred
from the active workflow.

Output is one row per design, tagged with the embedder that produced it, so
downstream screening can compare representations directly.

Duplicates are kept, not deduplicated: the fraction of repeats is itself a
signal about how sharply a representation constrains generation, and silently
collapsing them would bias every downstream distribution.
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
    "embedder", "run_id", "target_id", "split", "design_index",
    "sequence", "length", "status",
]

# Progress cadence: generation is the slowest step, so say something regularly.
_REPORT_EVERY = 500


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mc.MangoModel(
        d_ag=int(ckpt["d_ag"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
    )
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    model.lm.load_state_dict(ckpt["lm"])
    return model.to(device).eval()


def _target_row(records_csv, target_id):
    """The record being designed against -- present so failures name a structure."""
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    hit = df[df["id"] == target_id]
    if hit.empty:
        raise KeyError(
            f"generation target {target_id!r} is not in {records_csv}; "
            "generation.source must name the split this record lives in"
        )
    return hit.iloc[0]


def generate(ckpt_path, model_config_path, records_csv, antigen_emb_path,
             antibody_emb_path, target_id, split, tag, gen_cfg, seed, out_csv):
    device = dc.get_device(f"generation {tag}")
    torch.manual_seed(int(seed))

    model = _load_model(ckpt_path, device)
    with open(model_config_path) as fh:
        run_id = json.load(fh).get("run_id", "")

    row = _target_row(records_csv, target_id)
    h_ag = mc.load_embedding(antigen_emb_path).to(device)
    x_ctx = mc.load_embedding(antibody_emb_path).to(device)

    n = int(gen_cfg["n_per_target"])
    print(
        f"[{tag}] {target_id} ({split}): generating {n} heavy chains against "
        f"antigen chains {row['antigen_chains']} + light chain "
        f"({len(row['resolved_L_seq'])} aa) on {device}",
        flush=True,
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(n):
            base = {
                "embedder": tag,
                "run_id": run_id,
                "target_id": target_id,
                "split": split,
                "design_index": i,
            }
            try:
                ids = model.generate_heavy(
                    x_ctx,
                    h_ag,
                    max_new_tokens=int(gen_cfg["max_new_tokens"]),
                    do_sample=bool(gen_cfg.get("do_sample", True)),
                    top_p=float(gen_cfg.get("top_p", 1.0)),
                    temperature=float(gen_cfg.get("temperature", 1.0)),
                )
                seq = model.decode_heavy(ids)
                base.update(sequence=seq, length=len(seq), status="ok")
                n_ok += 1
            except Exception as e:  # keep going; record the failure per design
                base.update(
                    sequence="", length=0,
                    status=f"error: {type(e).__name__}: {e}",
                )
            writer.writerow(base)

            if (i + 1) % _REPORT_EVERY == 0:
                print(f"[{tag}] {target_id}: {i + 1}/{n}", flush=True)

    print(f"[{tag}] {target_id}: wrote {n_ok}/{n} designs -> {out_csv}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        generate(
            ckpt_path=smk.input.ckpt,
            model_config_path=smk.input.model_config,
            records_csv=smk.input.records,
            antigen_emb_path=smk.params.antigen_emb,
            antibody_emb_path=smk.params.antibody_emb,
            target_id=smk.wildcards.instance,
            split=smk.params.split,
            tag=smk.params.tag,
            gen_cfg=dict(smk.params.gen_cfg),
            seed=smk.params.seed,
            out_csv=smk.output.designs,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--antigen-emb", required=True)
    p.add_argument("--antibody-emb", required=True)
    p.add_argument("--target-id", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--gen-cfg", default="{}")
    a = p.parse_args()
    gen = {
        "n_per_target": 100, "max_new_tokens": 130,
        "do_sample": True, "top_p": 1.0, "temperature": 1.0,
    }
    gen.update(json.loads(a.gen_cfg))
    generate(
        a.ckpt, a.model_config, a.records, a.antigen_emb, a.antibody_emb,
        a.target_id, a.split, a.tag, gen, a.seed, a.out,
    )


if __name__ == "__main__":
    main()
