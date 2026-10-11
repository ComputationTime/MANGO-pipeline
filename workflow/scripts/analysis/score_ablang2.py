"""Score generated antibodies with AbLang2 pseudo-log-likelihood."""

import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PAIR_KEY_COLUMNS = ["embedder", "run_id", "target_id", "split", "design_index"]


def _prepare_cohort(cohort_csv):
    """Return one row per antibody from wide or long-format input."""
    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    if "sequence" not in df:
        raise ValueError("AbLang2 scoring requires a 'sequence' column")

    if "chain_type" not in df:
        if (df["sequence"].str.strip() == "").any():
            raise ValueError("AbLang2 scoring requires a heavy sequence for every design")
        if "light_sequence" not in df:
            df["light_sequence"] = ""
        return df

    missing_keys = [column for column in PAIR_KEY_COLUMNS if column not in df]
    if missing_keys:
        raise ValueError(
            "Long-format AbLang2 input is missing pairing columns: "
            + ", ".join(missing_keys)
        )

    chain_types = (
        df["chain_type"].astype(str).str.strip().str.lower().map(
            {"h": "heavy", "heavy": "heavy", "l": "light", "light": "light"}
        )
    )
    if chain_types.isna().any():
        invalid = sorted(set(df.loc[chain_types.isna(), "chain_type"].astype(str)))
        raise ValueError(
            "Long-format AbLang2 input contains unsupported chain_type values: "
            + ", ".join(invalid)
        )

    long_df = df.assign(_chain_type=chain_types)
    duplicate = long_df.duplicated(PAIR_KEY_COLUMNS + ["_chain_type"], keep=False)
    if duplicate.any():
        example = long_df.loc[duplicate, PAIR_KEY_COLUMNS].iloc[0].to_dict()
        raise ValueError(
            "Long-format AbLang2 input has duplicate chain rows for design "
            f"{example}"
        )

    counts = long_df.groupby(PAIR_KEY_COLUMNS, dropna=False)["_chain_type"].agg(set)
    incomplete = counts[counts != {"heavy", "light"}]
    if not incomplete.empty:
        raise ValueError(
            "Long-format AbLang2 input requires exactly one Heavy and one Light "
            f"row per design; found {len(incomplete)} incomplete design(s)"
        )

    heavy = long_df.loc[long_df["_chain_type"] == "heavy"].drop(
        columns=["chain_type", "_chain_type"]
    )
    light_columns = PAIR_KEY_COLUMNS + ["sequence"]
    light_renames = {"sequence": "light_sequence"}
    if "length" in long_df:
        light_columns.append("length")
        light_renames["length"] = "light_length"
    light = long_df.loc[
        long_df["_chain_type"] == "light", light_columns
    ].rename(columns=light_renames)

    paired = heavy.merge(
        light,
        on=PAIR_KEY_COLUMNS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if (paired["sequence"].str.strip() == "").any():
        raise ValueError("AbLang2 scoring requires a heavy sequence for every design")
    if (paired["light_sequence"].str.strip() == "").any():
        raise ValueError("Long-format AbLang2 input contains an empty light sequence")
    return paired


def score_ablang2(cohort_csv, model_dir, out_csv):
    """
    Score heavy-only or paired heavy/light antibodies with AbLang2.
    
    Parameters
    ----------
    cohort_csv : str
        Path to input CSV. Supported layouts are one row per antibody with a
        heavy-chain ``sequence`` and optional ``light_sequence``, or long
        format with separate Heavy and Light rows identified by ``chain_type``.
    model_dir : str
        Path to AbLang2 model directory.
    out_csv : str
        Path to output CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added pseudo-log-likelihood scores.
    """
    df = _prepare_cohort(cohort_csv)

    import ablang2
    import device_common as dc
    
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