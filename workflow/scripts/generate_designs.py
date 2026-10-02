"""Generate requested antibody chains from antigen embeddings and chain type.

No antibody sequence or embedding is required. Records are optional metadata
for the study workflow. Every design explicitly records its chain type.
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
    "chain_type", "sequence", "length", "status",
]

# Progress cadence: generation is the slowest step, so say something regularly.
_REPORT_EVERY = 500


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mc.MangoModel.from_checkpoint(ckpt)
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
             target_id, split, tag, gen_cfg, seed, out_csv):
    device = dc.get_device(f"generation {tag}")
    torch.manual_seed(int(seed))

    model = _load_model(ckpt_path, device)
    run_id = ""
    if model_config_path:
        with open(model_config_path) as fh:
            run_id = json.load(fh).get("run_id", "")
    if records_csv:
        _target_row(records_csv, target_id)  # Metadata identity check only.
    chain = model.normalize_chain(gen_cfg.get("chain", "heavy"))
    if chain not in model.target_chains:
        raise ValueError(f"Checkpoint was not trained for {chain} chains")
    h_ag = mc.load_embedding(antigen_emb_path).to(device)

    n = int(gen_cfg["n_per_target"])
    # The compact decoder fits a batch of 128 comfortably on the required
    # 48 GB-class study GPU while avoiding thousands of one-sample launches.
    batch_size = max(1, int(gen_cfg.get("batch_size", 128)))
    print(f"[{tag}] {target_id}: generating {n} {chain} chains from antigen only on {device}", flush=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        reported = 0
        for batch_start in range(0, n, batch_size):
            current_batch = min(batch_size, n - batch_start)
            try:
                sampled = model.generate_chain_batch(
                    h_ag, chain,
                    batch_size=current_batch,
                    max_new_tokens=int(gen_cfg["max_new_tokens"]),
                    do_sample=bool(gen_cfg.get("do_sample", True)),
                    top_p=float(gen_cfg.get("top_p", 1.0)),
                    temperature=float(gen_cfg.get("temperature", 1.0)),
                )
            except Exception as e:  # keep going; record the failure per design
                sampled = [e] * current_batch

            for offset, result in enumerate(sampled):
                base = {
                    "embedder": tag,
                    "run_id": run_id,
                    "target_id": target_id,
                    "split": split,
                    "chain_type": chain,
                    "design_index": batch_start + offset,
                }
                if isinstance(result, Exception):
                    base.update(
                        sequence="", length=0,
                        status=f"error: {type(result).__name__}: {result}",
                    )
                else:
                    seq = model.decode_chain(result)
                    base.update(sequence=seq, length=len(seq), status="ok")
                    n_ok += 1
                writer.writerow(base)
            fh.flush()

            completed = batch_start + current_batch
            if completed // _REPORT_EVERY > reported // _REPORT_EVERY:
                print(f"[{tag}] {target_id}: {completed}/{n}", flush=True)
            reported = completed

    print(f"[{tag}] {target_id}: wrote {n_ok}/{n} designs -> {out_csv}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        generate(
            ckpt_path=smk.input.ckpt,
            model_config_path=smk.input.model_config,
            records_csv=smk.input.records,
            antigen_emb_path=smk.params.antigen_emb,
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
    p.add_argument("--model-config")
    p.add_argument("--records")
    p.add_argument("--antigen-emb", required=True)
    p.add_argument("--target-id", default="antigen")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default="manual")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--gen-cfg", default="{}")
    p.add_argument("--chain", choices=["heavy", "light", "H", "L"], default="heavy")
    a = p.parse_args()
    gen = {
        "n_per_target": 100, "max_new_tokens": 130,
        "do_sample": True, "top_p": 1.0, "temperature": 1.0,
    }
    gen.update(json.loads(a.gen_cfg))
    gen["chain"] = a.chain
    generate(
        a.ckpt, a.model_config, a.records, a.antigen_emb,
        a.target_id, a.split, a.tag, gen, a.seed, a.out,
    )


if __name__ == "__main__":
    main()
