"""Score generated antibodies with AbLang2 pseudo-log-likelihood."""

import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
import chain_common


def _independent_model_inputs(frame):
    """Build AbLang2's (heavy, light) slots with exactly one populated."""
    return [
        (sequence, "") if chain_type == "Heavy" else ("", sequence)
        for sequence, chain_type in zip(frame["sequence"], frame["chain_type"])
    ]


def score_ablang2(cohort_csv, model_dir, out_csv):
    """
    Score Heavy and Light chains independently with AbLang2.
    
    Parameters
    ----------
    cohort_csv : str
        Path to a long-format Heavy/Light CSV or a wide CSV with ``sequence``
        and optional ``light_sequence`` columns.
    model_dir : str
        Path to AbLang2 model directory.
    out_csv : str
        Path to output CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added pseudo-log-likelihood scores.
    """
    df = chain_common.read_chain_rows(cohort_csv)

    import ablang2
    import device_common as dc
    
    device = str(dc.get_device("AbLang2 scoring"))
    model = ablang2.pretrained(
        model_to_use=str(model_dir), random_init=False, ncpu=1, device=device
    )
    
    model_inputs = _independent_model_inputs(df)
    unique_inputs = list(dict.fromkeys(model_inputs))
    
    values = model(
        [list(item) for item in unique_inputs], mode="pseudo_log_likelihood"
    )
    
    # AbLang2 computes cross-entropy with reduction="mean", so these values are
    # already normalized per non-special residue and must not be divided again.
    score_by_input = dict(zip(unique_inputs, map(float, values)))
    scores = [score_by_input[item] for item in model_inputs]
    df["ablang2_pseudo_log_likelihood"] = scores
    df["ablang2_pseudo_perplexity"] = [
        math.exp(-score) for score in scores
    ]
    df["ablang2_mode"] = "pseudo_log_likelihood"
    df["ablang2_context"] = df["chain_type"].str.lower() + "_only"
    df["ablang2_status"] = "ok"
    
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"AbLang2 scored {len(df)} independent chains "
        f"({len(unique_inputs)} unique chain inputs) on {device} -> {out_csv}",
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
    p.add_argument(
        "--cohort",
        required=True,
        help="Wide or Heavy/Light long-format input CSV with sequences",
    )
    p.add_argument("--model-dir", required=True, help="Path to AbLang2 model")
    p.add_argument("--out", required=True, help="Output CSV path")
    a = p.parse_args()
    score_ablang2(a.cohort, a.model_dir, a.out)


if __name__ == "__main__":
    main()