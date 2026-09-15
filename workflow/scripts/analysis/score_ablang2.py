"""Score generated antibodies with AbLang2 pseudo-log-likelihood."""

import math
from pathlib import Path
import sys

import ablang2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import device_common as dc


def score_ablang2(cohort_csv, model_dir, out_csv):
    """
    Score heavy-only or paired heavy/light antibodies with AbLang2.
    
    Parameters
    ----------
    cohort_csv : str
        Path to input CSV with a heavy-chain ``sequence`` column. An optional
        ``light_sequence`` column enables paired H|L scoring.
    model_dir : str
        Path to AbLang2 model directory.
    out_csv : str
        Path to output CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added pseudo-log-likelihood scores.
    """
    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    if "sequence" not in df:
        raise ValueError("AbLang2 scoring requires a 'sequence' column")
    if (df["sequence"].str.strip() == "").any():
        raise ValueError("AbLang2 scoring requires a heavy sequence for every design")
    if "light_sequence" not in df:
        df["light_sequence"] = ""
    
    device = str(dc.get_device("AbLang2 scoring"))
    model = ablang2.pretrained(
        model_to_use=str(model_dir), random_init=False, ncpu=1, device=device
    )
    
    paired = [
        (heavy.strip(), light.strip())
        for heavy, light in zip(df["sequence"], df["light_sequence"])
    ]
    unique_pairs = list(dict.fromkeys(paired))
    
    values = model(
        [list(pair) for pair in unique_pairs], mode="pseudo_log_likelihood"
    )
    
    # AbLang2 computes cross-entropy with reduction="mean", so these values are
    # already normalized per non-special residue and must not be divided again.
    score_by_pair = dict(zip(unique_pairs, map(float, values)))
    scores = [score_by_pair[pair] for pair in paired]
    df["ablang2_pseudo_log_likelihood"] = scores
    df["ablang2_pseudo_perplexity"] = [
        math.exp(-score) for score in scores
    ]
    df["ablang2_mode"] = "pseudo_log_likelihood"
    df["ablang2_context"] = [
        "paired_hl" if light else "heavy_only" for _, light in paired
    ]
    df["ablang2_status"] = "ok"
    
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"AbLang2 scored {len(df)} antibodies "
        f"({len(unique_pairs)} unique) on {device} -> {out_csv}",
        flush=True,
    )
    
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_ablang2(smk.input.cohort, smk.params.model_dir, smk.output.metrics)
        return
    
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True, help="Input CSV with sequences")
    p.add_argument("--model-dir", required=True, help="Path to AbLang2 model")
    p.add_argument("--out", required=True, help="Output CSV path")
    a = p.parse_args()
    score_ablang2(a.cohort, a.model_dir, a.out)


if __name__ == "__main__":
    main()