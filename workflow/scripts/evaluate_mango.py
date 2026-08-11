"""Per-split NLL and perplexity for a trained run -> eval.json.

This is the data behind handbook figure 1 ("Ag embeddings have negligible
effect..."): the same number computed identically for every antigen
representation, so the bars are comparable.

NLL is token-weighted, not example-weighted: total cross-entropy over all
predicted tokens divided by the number of predicted tokens. That makes it
insensitive to the length distribution of a split, so train/val/test bars can
be read against each other.

"Predicted tokens" means the HEAVY chain plus its end token -- the context block
is masked out of the loss -- so the number is a heavy-chain NLL and is not
diluted by antigen or light-chain length.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc
import device_common as dc

# exp() of a large NLL overflows; cap it the way perplexity is normally reported.
MAX_EXPONENT = 50.0


def _rows(records_csv, split):
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    df = df[df["split"] == split]
    return list(
        df[["id", "split", "resolved_H_seq", "resolved_L_seq"]].itertuples(index=False)
    )


def _paths(emb_dir, tag, antibody_dir, row):
    ag = Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt"
    ab = Path(emb_dir) / "antibody" / antibody_dir / row.split / f"{row.id}.pt"
    return str(ag), str(ab)


@torch.no_grad()
def _split_nll(model, rows, emb_dir, tag, antibody_dir, device):
    """(nll_per_token, n_tokens, n_examples, n_skipped) for one split."""
    total_nll = 0.0
    total_tokens = 0
    n_examples = 0
    skipped = 0

    for row in rows:
        ag_path, ab_path = _paths(emb_dir, tag, antibody_dir, row)
        if not (Path(ag_path).is_file() and Path(ab_path).is_file()):
            skipped += 1
            continue
        h_ag = mc.load_embedding(ag_path).to(device)
        x_ctx = mc.load_embedding(ab_path).to(device)
        heavy_ids = model.heavy_token_ids(row.resolved_H_seq).to(device)

        # len(H) + 1 for the end token; the context block is labelled -100.
        n_pred = model.n_target_tokens(heavy_ids)
        if n_pred <= 0:
            skipped += 1
            continue
        loss = model.loss(x_ctx, h_ag, heavy_ids)      # mean CE over those tokens
        total_nll += float(loss.item()) * n_pred
        total_tokens += n_pred
        n_examples += 1

    nll = total_nll / total_tokens if total_tokens else float("nan")
    return nll, total_tokens, n_examples, skipped


def evaluate(records_csv, emb_dir, tag, antibody_dir, ckpt_path, model_config_path,
             splits, out_json):
    device = dc.get_device(f"evaluation {tag}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(model_config_path) as fh:
        model_config = json.load(fh)

    model = mc.MangoModel(
        d_ag=int(ckpt["d_ag"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
    )
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    model.lm.load_state_dict(ckpt["lm"])
    model = model.to(device).eval()

    result = {
        "run_id": model_config.get("run_id"),
        "experiment_hash": model_config.get("experiment_hash"),
        "embedder": tag,
        "task": model_config.get("task"),
        "d_ag": int(ckpt["d_ag"]),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "splits": {},
    }

    for split in splits:
        rows = _rows(records_csv, split)
        if not rows:
            print(f"[{tag}] split {split!r}: no rows, skipping", flush=True)
            continue
        nll, n_tokens, n_examples, skipped = _split_nll(
            model, rows, emb_dir, tag, antibody_dir, device
        )
        ppl = math.exp(min(nll, MAX_EXPONENT)) if n_tokens else float("nan")
        result["splits"][split] = {
            "nll": nll,
            "perplexity": ppl,
            "n_tokens": n_tokens,
            "n_examples": n_examples,
            "n_skipped_missing_embeddings": skipped,
        }
        print(
            f"[{tag}] {split}: nll={nll:.4f} ppl={ppl:.2f} "
            f"({n_examples} examples, {n_tokens} tokens, {skipped} skipped)",
            flush=True,
        )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out_json}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        evaluate(
            records_csv=smk.input.records,
            emb_dir=smk.params.emb_dir,
            tag=smk.params.tag,
            antibody_dir=smk.params.antibody_dir,
            ckpt_path=smk.input.ckpt,
            model_config_path=smk.input.model_config,
            splits=list(smk.params.splits),
            out_json=smk.output.eval_json,
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
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluate(
        a.records, a.emb_dir, a.tag, a.antibody_dir, a.ckpt, a.model_config,
        a.splits.split(","), a.out,
    )


if __name__ == "__main__":
    main()
