"""Score generated heavy/light pairs with an explicitly staged AbLang2 model."""

from pathlib import Path
import sys

import ablang2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import device_common as dc


VALID_MODES = {"confidence", "pseudo_log_likelihood"}


def score_ablang2(cohort_csv, model_dir, mode, out_csv):
    if mode not in VALID_MODES:
        raise ValueError(f"AbLang2 score mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    df = pd.read_csv(
        cohort_csv, dtype={"sequence": str, "light_sequence": str}, keep_default_na=False
    )
    if (df["light_sequence"] == "").any():
        raise ValueError("AbLang2 paired scoring requires a light_sequence for every design")
    device = str(dc.get_device("AbLang2 scoring"))
    model = ablang2.pretrained(
        model_to_use=str(model_dir), random_init=False, ncpu=1, device=device
    )
    # AbLang2's paired API expects an outer batch of [heavy, light] pairs. It
    # inserts the `|` separator itself; passing prejoined strings makes it treat
    # each string as an iterable pair and produces a malformed sequence.
    paired = [[heavy, light] for heavy, light in zip(
        df["sequence"], df["light_sequence"]
    )]
    values = model(paired, mode=mode)
    column = f"ablang2_{mode}"
    df[column] = [float(x) for x in values]
    df["ablang2_status"] = "ok"
    df["ablang2_mode"] = mode
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"AbLang2 {mode} scored {len(df)} H|L pairs on {device} -> {out_csv}", flush=True)
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_ablang2(smk.input.cohort, smk.params.model_dir,
                      smk.params.mode, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mode", choices=sorted(VALID_MODES), default="confidence")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_ablang2(a.cohort, a.model_dir, a.mode, a.out)


if __name__ == "__main__":
    main()
