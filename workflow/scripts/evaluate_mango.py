"""Token-weighted per-split and per-chain NLL for antigen + chain MANGO."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc
import device_common as dc
from chain_examples import example_chains
from embedding_store import embedding_available

# exp() of a large NLL overflows; cap it the way perplexity is normally reported.
MAX_EXPONENT = 50.0


def _rows(records_csv, split):
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    df = df[df["split"] == split]
    columns = ["id", "split", "resolved_H_seq", "resolved_L_seq"]
    if 'example_target_chains' in df.columns:
        columns.append('example_target_chains')
    return list(df[columns].itertuples(index=False))




@torch.no_grad()
def _split_nll(model, rows, emb_dir, tag, chain, device, context_lengths=None):
    """(nll_per_token, n_tokens, n_examples, n_skipped) for one split."""
    total_nll = 0.0
    total_tokens = 0
    n_examples = 0
    skipped = 0

    for row in rows:
        if chain not in example_chains(row):
            continue
        ag_path = Path(emb_dir) / "antigen" / tag / row.split / f"{row.id}.pt"
        if not embedding_available(ag_path):
            raise FileNotFoundError(f"Incomplete comparison cohort: {ag_path}")
        h_ag = mc.load_embedding(ag_path).to(device)
        if context_lengths is not None:
            context_lengths[row.id] = int(h_ag.shape[1])
        sequence = row.resolved_H_seq if chain == "heavy" else row.resolved_L_seq
        if not sequence:
            raise ValueError(f"{row.id}: missing {chain} target")
        target_ids = model.target_token_ids(sequence).to(device)

        # len(H) + 1 for the end token; the context block is labelled -100.
        n_pred = model.n_target_tokens(target_ids)
        if n_pred <= 0:
            skipped += 1
            continue
        loss = model.loss(h_ag, chain, target_ids)      # mean CE over those tokens
        total_nll += float(loss.item()) * n_pred
        total_tokens += n_pred
        n_examples += 1

    nll = total_nll / total_tokens if total_tokens else float("nan")
    return nll, total_tokens, n_examples, skipped


def evaluate(records_csv, emb_dir, tag, ckpt_path, model_config_path,
             splits, out_json, evaluation_subsets=None):
    device = dc.get_device(f"evaluation {tag}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(model_config_path) as fh:
        model_config = json.load(fh)

    cohort_sha256 = mc.comparison_cohort(records_csv)
    if ckpt.get("cohort_sha256") != cohort_sha256:
        raise ValueError("Evaluation cohort differs from training checkpoint records")
    model = mc.MangoModel.from_checkpoint(ckpt)
    model = model.to(device).eval()
    embedding_tag = ckpt.get("embedding_tag", tag)

    result = {
        "cohort_sha256": cohort_sha256,
        "loss_reduction": "target_token_mean_including_eos",
        "run_id": model_config.get("run_id"),
        "experiment_hash": model_config.get("experiment_hash"),
        "embedder": tag,
        "embedding_tag": embedding_tag,
        "architecture": model.architecture,
        "antigen_in_decoder_prefix": not model.uses_cross_attention,
        "cross_attention_trainable_parameters": model.cross_attention_parameters,
        "task": model_config.get("task"),
        "d_ag": int(ckpt["d_ag"]),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "initialization": ckpt.get("initialization"),
        "training_loss": ckpt.get('training_loss', {'name': 'cross_entropy'}),
        "checkpoint_selection": ckpt.get('checkpoint_selection', 'validation_nll'),
        "evaluation_loss": "cross_entropy",
        "splits": {},
    }

    planned = [(split, _rows(records_csv, split)) for split in splits]
    for label, subset in (evaluation_subsets or {}).items():
        if label in splits:
            raise ValueError(f"Evaluation subset duplicates split: {label}")
        ids = set(subset["ids"])
        rows = [r for r in _rows(records_csv, subset["source_split"]) if r.id in ids]
        if not ids or {r.id for r in rows} != ids:
            raise ValueError(f"Evaluation subset {label} has missing or unknown IDs")
        planned.append((label, rows))

    for split, rows in planned:
        if not rows:
            print(f"[{tag}] split {split!r}: no rows, skipping", flush=True)
            continue
        by_chain = {}
        context_lengths = {}
        total_nll = 0.0
        n_tokens = n_examples = skipped = 0
        for chain in model.target_chains:
            value, tokens, examples, missing = _split_nll(
                model, rows, emb_dir, embedding_tag, chain, device, context_lengths
            )
            by_chain[chain] = {"nll": value, "perplexity": math.exp(min(value, MAX_EXPONENT)) if tokens else float("nan"),
                               "n_tokens": tokens, "n_examples": examples, "n_skipped_missing_embeddings": missing}
            total_nll += value * tokens if tokens else 0.0
            n_tokens += tokens
            n_examples += examples
            skipped += missing
        nll = total_nll / n_tokens if n_tokens else float("nan")
        ppl = math.exp(min(nll, MAX_EXPONENT)) if n_tokens else float("nan")
        result["splits"][split] = {
            "nll": nll, "perplexity": ppl, "n_tokens": n_tokens,
            "n_examples": n_examples, "n_skipped_missing_embeddings": skipped,
            "by_chain": by_chain,
            "antigen_context_tokens": {
                "n_records": len(context_lengths),
                "min": min(context_lengths.values()),
                "max": max(context_lengths.values()),
                "mean": sum(context_lengths.values()) / len(context_lengths),
            },
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluate(
        a.records, a.emb_dir, a.tag, a.ckpt, a.model_config,
        a.splits.split(","), a.out,
    )


if __name__ == "__main__":
    main()
