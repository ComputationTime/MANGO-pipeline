"""AlphaSeq preparation and FLAb average perplexity.

FLAb scores each chain on its own. Chain perplexity is

    exp(-mean log-likelihood) = exp(mean token NLL)

and a paired scFv's plotted score is the arithmetic mean of those two
perplexities:

    average_perplexity = (heavy_perplexity + light_perplexity) / 2

A perplexity of 1 would mean every residue had probability 1. That is not a
valid model score, so values at or below 1 are rejected.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


LINKER = "GGGGS" * 3
TARGETS = ("HER2", "PD1", "TIGIT")
LANGUAGE_MODELS = ("iglm", "antiberty", "ablang2")
MODEL_LABELS = {
    "iglm": "IgLM",
    "antiberty": "AntiBERTy",
    "ablang2": "AbLang2",
    "one_hot": "One-hot",
    "biopython": "BioPython",
    "pyrosetta_pre": "PyRosetta PRE",
    "esm2": "ESM2",
    "esm3": "ESM3 (sequence)",
    "esmif": "ESM-IF",
    "proteinmpnn": "ProteinMPNN",
}
SCORE_KEYS = ("target", "heavy_sequence", "light_sequence", "alphaseq_affinity")


def split_scfv(sequence: str) -> tuple[str, str]:
    """Split one AlphaSeq scFv on the single (GGGGS)3 linker."""
    sequence = str(sequence).strip().upper()
    found = sequence.count(LINKER)
    if found != 1:
        raise ValueError(f"expected one (GGGGS)3 linker, found {found}")
    heavy, light = sequence.split(LINKER)
    if not heavy or not light:
        raise ValueError("linker split produced an empty chain")
    return heavy, light


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep well-observed scFvs and average affinity across replicate rows."""
    required = [
        "mata_sequence",
        "alphaseq_affinity",
        "sufficient_replicate_observations",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{target} is missing columns: {missing}")

    work = df.loc[_as_bool(df["sufficient_replicate_observations"])].copy()
    work = work.dropna(subset=["mata_sequence", "alphaseq_affinity"])
    work["mata_sequence"] = work["mata_sequence"].astype(str).str.strip().str.upper()
    work = work.loc[~work["mata_sequence"].isin({"", "NAN", "NONE"})]

    rows = []
    skipped = []
    for sequence, group in work.groupby("mata_sequence", sort=False):
        try:
            heavy, light = split_scfv(sequence)
        except ValueError as exc:
            skipped.append({"target": target, "mata_sequence": sequence, "reason": str(exc)})
            continue
        antigen = ""
        if "matalpha_sequence" in group.columns and group["matalpha_sequence"].notna().any():
            antigen = str(group["matalpha_sequence"].dropna().astype(str).iloc[0])
        rows.append({
            "target": target,
            "heavy_sequence": heavy,
            "light_sequence": light,
            "heavy_length": len(heavy),
            "light_length": len(light),
            "alphaseq_affinity": float(group["alphaseq_affinity"].mean()),
            "affinity_lower_bound": _mean_column(group, "affinity_lower_bound"),
            "affinity_upper_bound": _mean_column(group, "affinity_upper_bound"),
            "n_replicates": int(len(group)),
            "matalpha_sequence": antigen,
        })
    prepared = pd.DataFrame(rows)
    if not prepared.empty:
        prepared = prepared.sort_values(
            ["heavy_sequence", "light_sequence"]
        ).reset_index(drop=True)
    return prepared, pd.DataFrame(skipped)


def _mean_column(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    if values.notna().any():
        return float(values.mean())
    return float("nan")


def _valid_perplexity(value: float) -> float:
    """Reject perfect or impossible perplexity. Valid scores are strictly above 1."""
    value = float(value)
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError(
            f"perplexity {value} is not a valid model score; "
            "average perplexity must be greater than 1"
        )
    return value


def chain_perplexity_from_log_likelihood(mean_log_likelihood: float) -> float:
    """FLAb chain perplexity, exp(-mean log-likelihood)."""
    return _valid_perplexity(math.exp(-float(mean_log_likelihood)))


def chain_perplexity_from_nll(mean_nll: float) -> float:
    """Same perplexity when the model already returned mean token NLL."""
    return _valid_perplexity(math.exp(float(mean_nll)))


def average_perplexity(heavy_perplexity: float, light_perplexity: float) -> float:
    """Arithmetic mean of the two chain perplexities, as in FLAb scoring."""
    heavy = _valid_perplexity(heavy_perplexity)
    light = _valid_perplexity(light_perplexity)
    return (heavy + light) / 2.0


def apply_chain_nlls(
    frame: pd.DataFrame,
    model: str,
    heavy_nll: dict[str, float],
    light_nll: dict[str, float],
) -> pd.DataFrame:
    """Attach per-chain perplexities and their arithmetic mean."""
    records = frame.copy()
    heavy_values, light_values, averaged, heavy_nlls, light_nlls, status = (
        [], [], [], [], [], [],
    )
    for heavy, light in zip(records["heavy_sequence"], records["light_sequence"]):
        heavy_score = heavy_nll.get(heavy)
        light_score = light_nll.get(light)
        if (
            heavy_score is None
            or light_score is None
            or not np.isfinite(heavy_score)
            or not np.isfinite(light_score)
        ):
            heavy_values.append(np.nan)
            light_values.append(np.nan)
            averaged.append(np.nan)
            heavy_nlls.append(np.nan)
            light_nlls.append(np.nan)
            status.append("error: missing chain score")
            continue
        try:
            heavy_ppl = chain_perplexity_from_nll(heavy_score)
            light_ppl = chain_perplexity_from_nll(light_score)
            combined = average_perplexity(heavy_ppl, light_ppl)
        except ValueError as exc:
            heavy_values.append(np.nan)
            light_values.append(np.nan)
            averaged.append(np.nan)
            heavy_nlls.append(float(heavy_score))
            light_nlls.append(float(light_score))
            status.append(f"error: {exc}")
            continue
        heavy_values.append(heavy_ppl)
        light_values.append(light_ppl)
        averaged.append(combined)
        heavy_nlls.append(float(heavy_score))
        light_nlls.append(float(light_score))
        status.append("ok")
    records[f"{model}_heavy_nll"] = heavy_nlls
    records[f"{model}_light_nll"] = light_nlls
    records[f"{model}_heavy_perplexity"] = heavy_values
    records[f"{model}_light_perplexity"] = light_values
    records[f"{model}_average_perplexity"] = averaged
    records[f"{model}_status"] = status
    return records


def log_likelihoods_to_nlls(scores: dict[str, float]) -> dict[str, float]:
    """Convert mean log-likelihoods to mean token NLL."""
    converted = {}
    for sequence, score in scores.items():
        if score is None or not np.isfinite(score):
            converted[sequence] = float("nan")
        else:
            converted[sequence] = -float(score)
    return converted


def perplexity_column(model: str) -> str:
    return f"{model}_average_perplexity"


def model_ids(frame: pd.DataFrame) -> list[str]:
    """Trained models that already have an average-perplexity column."""
    suffix = "_average_perplexity"
    return [column[: -len(suffix)] for column in frame.columns if column.endswith(suffix)]


def load_score_tables(paths) -> pd.DataFrame:
    """Merge per-model score tables on target, chains, and affinity."""
    merged = None
    for path in paths:
        frame = pd.read_csv(path)
        missing = [column for column in SCORE_KEYS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        models = model_ids(frame)
        if not models:
            raise ValueError(f"{path} has no average-perplexity columns")
        keep = list(SCORE_KEYS)
        for model in models:
            keep.extend([
                f"{model}_heavy_nll",
                f"{model}_light_nll",
                f"{model}_heavy_perplexity",
                f"{model}_light_perplexity",
                perplexity_column(model),
                f"{model}_status",
            ])
        if merged is None and "n_replicates" in frame.columns:
            keep.append("n_replicates")
        slim = frame.loc[:, [column for column in keep if column in frame.columns]].copy()
        if merged is None:
            merged = slim
            continue
        overlap = set(model_ids(merged)) & set(models)
        if overlap:
            raise ValueError(f"duplicate scores for {sorted(overlap)} in {path}")
        merged = merged.merge(slim, on=list(SCORE_KEYS), how="outer", suffixes=("", "_dup"))
        duplicate_columns = [column for column in merged.columns if column.endswith("_dup")]
        if duplicate_columns:
            raise ValueError(f"could not merge {path}: {duplicate_columns}")
    if merged is None:
        raise ValueError("no score tables were provided")
    return merged
