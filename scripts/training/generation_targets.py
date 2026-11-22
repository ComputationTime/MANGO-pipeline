"""Deterministic, leakage-safe target selection for de novo generation."""

import pandas as pd


def select_generation_targets(
    records: pd.DataFrame,
    split: str,
    strategy: str,
    cluster_column: str,
    max_targets=None,
) -> list[str]:
    """Select held-out targets and reject any cluster seen outside the split."""
    required = {"id", "split", cluster_column}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(
            f"generation target selection is missing column(s) {sorted(missing)}"
        )

    work = records.loc[:, ["id", "split", cluster_column]].copy()
    work["id"] = work["id"].astype(str).str.strip()
    work["split"] = work["split"].astype(str).str.strip().str.lower()
    work[cluster_column] = work[cluster_column].astype(str).str.strip()
    if (work["id"] == "").any() or (work[cluster_column] == "").any():
        raise ValueError("generation targets require non-empty ids and cluster labels")

    split = str(split).strip().lower()
    candidates = work.loc[work["split"] == split].copy()
    if candidates.empty:
        raise ValueError(f"generation source split {split!r} contains no records")

    split_counts = work.groupby(cluster_column)["split"].nunique()
    crossing = set(split_counts[split_counts > 1].index)
    selected_crossing = sorted(set(candidates[cluster_column]) & crossing)
    if selected_crossing:
        raise ValueError(
            "generation target cluster leakage across train/val/test: "
            f"{selected_crossing[:5]}"
        )

    if strategy == "one_per_cluster":
        selected = (
            candidates.sort_values([cluster_column, "id"])
            .groupby(cluster_column, sort=True, as_index=False)
            .first()
            .sort_values([cluster_column, "id"])
        )
    elif strategy == "all_records":
        selected = candidates.sort_values("id")
    else:
        raise ValueError(
            f"unsupported generation target strategy {strategy!r}; "
            "use 'one_per_cluster' or 'all_records'"
        )

    if max_targets is not None:
        limit = int(max_targets)
        if limit < 1:
            raise ValueError("generation.max_targets must be null or a positive integer")
        selected = selected.head(limit)
    return selected["id"].tolist()
