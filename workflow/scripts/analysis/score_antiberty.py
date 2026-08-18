"""Score generated heavy chains with AntiBERTy pseudo-log-likelihood."""

import sys

from pathlib import Path

import pandas as pd
import torch
from antiberty import AntiBERTyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import device_common as dc


def _pseudo_log_likelihood(model, sequence, batch_size):
    """AntiBERTy's masked-residue PLL with labels on the model's device."""
    masked = []
    for i in range(len(sequence)):
        tokens = list(sequence)
        tokens[i] = "[MASK]"
        masked.append(" ".join(tokens))
    encoded = model.tokenizer(masked, return_tensors="pt", padding=True)
    tokens = encoded["input_ids"].to(model.device)
    attention = encoded["attention_mask"].to(model.device)
    logits = []
    with torch.no_grad():
        for start in range(0, len(masked), int(batch_size)):
            stop = min(start + int(batch_size), len(masked))
            output = model.model(input_ids=tokens[start:stop],
                                 attention_mask=attention[start:stop])
            logits.append(output.prediction_logits)
    logits = torch.cat(logits, dim=0)
    logits[:, :, model.tokenizer.all_special_ids] = -float("inf")
    logits = logits[:, 1:-1]
    logits = torch.diagonal(logits, dim1=0, dim2=1).unsqueeze(0)
    labels = model.tokenizer.encode(
        " ".join(list(sequence)), return_tensors="pt"
    )[:, 1:-1].to(model.device)
    return -torch.nn.functional.cross_entropy(logits, labels, reduction="mean").item()


def score_antiberty(cohort_csv, batch_size, out_csv):
    dc.get_device("AntiBERTy scoring")
    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    model = AntiBERTyRunner()
    scores, statuses = [], []
    for sequence in df["sequence"]:
        try:
            scores.append(_pseudo_log_likelihood(model, sequence, batch_size))
            statuses.append("ok")
        except Exception as exc:
            scores.append(float("nan"))
            statuses.append(f"error: {type(exc).__name__}: {exc}")
    df["antiberty_pseudo_log_likelihood"] = scores
    df["antiberty_status"] = statuses
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"AntiBERTy scored {sum(s == 'ok' for s in statuses)}/{len(df)} "
          f"sequences on {model.device} -> {out_csv}", flush=True)
    if any(status != "ok" for status in statuses):
        raise RuntimeError(
            "AntiBERTy did not score every selected sequence; diagnostics were "
            f"written to {out_csv}"
        )
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_antiberty(smk.input.cohort, smk.params.batch_size, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_antiberty(a.cohort, a.batch_size, a.out)


if __name__ == "__main__":
    main()
