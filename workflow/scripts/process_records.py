"""standardized.csv -> records.csv (filtered, resolved, split-assigned).

This is the last dataset-agnostic stage before embedding. It:
  1. applies the configured filters,
  2. guarantees a *resolved* sequence for heavy, light, and each antigen chain
     (passthrough from the standardizer when available, else parsed from the
     structure file),
  3. carves a validation set out of train,
  4. writes one tidy row per kept structure.

Output schema (the pipeline's central table)
--------------------------------------------
id, pdb_path, antigen_chains, chains,
expected_heavy_seq, expected_light_seq, expected_ag_seq,
resolved_H_seq, resolved_L_seq, resolved_ag_seq,
split                                    (train | val | test)
+ any passthrough columns the standardizer supplied.

Multi-chain antigen fields are comma-separated, one segment per antigen chain,
in the same order as `antigen_chains`.

Filters (config["processing"])
------------------------------
require_paired          keep only rows with BOTH a heavy and a light chain
antigen_types           every antigen chain type must be in the allowed set
max_antigen_len         drop rows whose summed resolved antigen length exceeds
                        the cap (null disables)
require_structure_file  drop rows whose structure file is missing

Validation set: the source split is train/test only, so `val` is carved out of
train by holding out complete antibody-antigen clusters. Row-wise random
splitting is rejected because it can leak related complexes across splits.
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import standardize_common as sc

SEP = sc.SEP

OUTPUT_COLUMNS = [
    "id",
    "pdb_path",
    "antigen_chains",
    "chains",
    "expected_heavy_seq",
    "expected_light_seq",
    "expected_ag_seq",
    "resolved_H_seq",
    "resolved_L_seq",
    "resolved_ag_seq",
    "split",
]


def _nonempty(series: pd.Series) -> pd.Series:
    """Boolean mask: value is present and not an empty/whitespace string."""
    return series.notna() & (series.astype(str).str.strip() != "")


def _all_types_allowed(agtypes: str, allowed: set) -> bool:
    """True if agtypes is non-empty and every comma-separated type is allowed."""
    if not isinstance(agtypes, str) or agtypes.strip() == "":
        return False
    chains = [t.strip() for t in agtypes.split(SEP) if t.strip()]
    return bool(chains) and all(t.upper() in allowed for t in chains)


def _total_antigen_len(agseqs: str) -> int:
    if not isinstance(agseqs, str):
        return 0
    return sum(len(s.strip()) for s in agseqs.split(SEP))


# --- resolved sequences ------------------------------------------------------
def _parse_resolved(pdb_path: str, antigen_chains: str, heavy: str, light: str):
    """Parse resolved sequences from a structure file (fallback path).

    Used only when the standardizer did not supply resolved_* columns. Returns
    (H, L, ag_csv) using the same backbone-complete residue definition the
    structure embedders use, so lengths stay consistent across representations.
    """
    from lib.mmcif import backbone_dict

    chains = [c for c in antigen_chains.split(SEP) if c]
    wanted = [c for c in ([heavy] if heavy else []) + ([light] if light else []) + chains]
    d = backbone_dict(pdb_path, wanted)

    def seq_of(chain):
        return d.get(f"seq_chain_{chain}", "") if chain else ""

    return seq_of(heavy), seq_of(light), SEP.join(seq_of(c) for c in chains)


def ensure_resolved(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee resolved_H_seq / resolved_L_seq / resolved_ag_seq exist."""
    have = {"resolved_H_seq", "resolved_L_seq", "resolved_ag_seq"} <= set(df.columns)
    if have:
        print("  resolved sequences: passthrough from standardizer", flush=True)
        return df

    print(f"  resolved sequences: parsing {len(df)} structure files", flush=True)
    heavy_col = df["heavy_chain"] if "heavy_chain" in df.columns else pd.Series("", index=df.index)
    light_col = df["light_chain"] if "light_chain" in df.columns else pd.Series("", index=df.index)

    parsed = [
        _parse_resolved(r.pdb_path, r.antigen_chains, h, l)
        for r, h, l in zip(df.itertuples(index=False), heavy_col, light_col)
    ]
    df = df.copy()
    df["resolved_H_seq"] = [p[0] for p in parsed]
    df["resolved_L_seq"] = [p[1] for p in parsed]
    df["resolved_ag_seq"] = [p[2] for p in parsed]
    return df


# --- validation split --------------------------------------------------------
def assign_validation(df: pd.DataFrame, val_cfg: dict) -> pd.Series:
    """Return splits with validation carved out of train at cluster level.

    Rows sharing an antibody-antigen cluster must remain in the same split. A
    missing cluster annotation is therefore an error rather than a reason to
    silently fall back to row-wise random sampling.
    """
    split = df["split"].astype(str).str.strip().str.lower().copy()
    strategy = val_cfg.get("strategy", "cluster")

    train_mask = split == "train"
    if int(train_mask.sum()) == 0:
        return split

    if strategy != "cluster":
        raise ValueError(
            f"val strategy {strategy!r} is not supported; use 'cluster' so "
            "antibody-antigen clusters cannot leak between train and validation"
        )

    fraction = float(val_cfg.get("fraction", 0.10))
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"val fraction must satisfy 0 <= fraction < 1, got {fraction}")
    if fraction == 0.0:
        return split

    cluster_column = str(val_cfg.get("cluster_column", "ab_ag_cluster"))
    if cluster_column not in df.columns:
        raise ValueError(
            f"cluster-aware validation requires column {cluster_column!r}; "
            f"available columns: {sorted(df.columns)}"
        )

    train_clusters = df.loc[train_mask, cluster_column].astype(str).str.strip()
    missing = train_clusters == ""
    if missing.any():
        examples = list(df.loc[train_clusters[missing].index, "id"].astype(str).head(5))
        raise ValueError(
            f"cluster-aware validation found {int(missing.sum())} training rows "
            f"without {cluster_column!r}; example ids: {examples}"
        )

    clusters = sorted(train_clusters.unique())
    if len(clusters) < 2:
        raise ValueError(
            f"cluster-aware validation needs at least two training clusters in "
            f"{cluster_column!r}, found {len(clusters)}"
        )

    # Sample clusters, not rows. Sorting before seeded sampling makes the split
    # reproducible even if the input table is reordered.
    n_val_clusters = int(round(len(clusters) * fraction))
    n_val_clusters = max(1, min(len(clusters) - 1, n_val_clusters))
    seed = int(val_cfg.get("seed", 42))
    val_clusters = set(random.Random(seed).sample(clusters, n_val_clusters))
    cluster_values = df[cluster_column].astype(str).str.strip()
    split.loc[train_mask & cluster_values.isin(val_clusters)] = "val"

    train_after = set(cluster_values.loc[split == "train"])
    val_after = set(cluster_values.loc[split == "val"])
    overlap = train_after & val_after
    if overlap:
        raise AssertionError(
            f"cluster leakage between train and validation: {sorted(overlap)[:5]}"
        )

    print(
        f"  validation clusters: {len(val_clusters)} / {len(clusters)} "
        f"({int((split == 'val').sum())} / {int(train_mask.sum())} source-train rows), "
        f"column={cluster_column}, seed={seed}",
        flush=True,
    )
    return split


def validate_cluster_partition(df: pd.DataFrame, split: pd.Series, val_cfg: dict) -> None:
    """Fail if any antibody-antigen cluster crosses train, validation, or test."""
    cluster_column = str(val_cfg.get("cluster_column", "ab_ag_cluster"))
    if cluster_column not in df.columns:
        raise ValueError(
            f"cluster-aware partition validation requires column {cluster_column!r}"
        )
    model_mask = split.isin({"train", "val", "test"})
    clusters = df.loc[model_mask, cluster_column].astype(str).str.strip()
    missing = clusters == ""
    if missing.any():
        examples = list(df.loc[missing[missing].index, "id"].astype(str).head(5))
        raise ValueError(
            f"found {int(missing.sum())} model rows without {cluster_column!r}; "
            f"example ids: {examples}"
        )
    audit = pd.DataFrame({"cluster": clusters, "split": split.loc[model_mask]})
    crossing = audit.groupby("cluster")["split"].nunique()
    crossing = crossing[crossing > 1]
    if not crossing.empty:
        examples = sorted(crossing.index.astype(str))[:5]
        raise ValueError(
            f"{cluster_column!r} leakage across train/val/test for "
            f"{len(crossing)} cluster(s); examples: {examples}"
        )


def select_cluster_subset(df: pd.DataFrame, subset_cfg: dict, val_cfg: dict) -> pd.DataFrame:
    """Select a tiny deterministic smoke cohort without breaking cluster isolation."""
    if not subset_cfg.get("enabled", False):
        return df
    cluster_column = str(val_cfg.get("cluster_column", "ab_ag_cluster"))
    if cluster_column not in df.columns:
        raise ValueError(f"smoke subset requires cluster column {cluster_column!r}")

    work = df.copy()
    work["_subset_cluster"] = work[cluster_column].astype(str).str.strip()
    work["_subset_split"] = work["split"].astype(str).str.strip().str.lower()
    if (work["_subset_cluster"] == "").any():
        raise ValueError("smoke subset cannot select rows with blank cluster annotations")

    # Never select a cluster that already occurs in multiple source splits.
    split_counts = work.groupby("_subset_cluster")["_subset_split"].nunique()
    exclusive = set(split_counts[split_counts == 1].index)
    work = work[work["_subset_cluster"].isin(exclusive)].copy()
    work["_subset_ag_len"] = work["resolved_ag_seq"].apply(_total_antigen_len)

    rows_per_cluster = int(subset_cfg.get("rows_per_cluster", 1))
    if rows_per_cluster < 1:
        raise ValueError("processing.subset.rows_per_cluster must be >= 1")
    selected = []
    requests = {
        "train": int(subset_cfg.get("train_clusters", 4)),
        "test": int(subset_cfg.get("test_clusters", 1)),
    }
    for source_split, n_clusters in requests.items():
        candidates = work[work["_subset_split"] == source_split].copy()
        order = (
            candidates.groupby("_subset_cluster")["_subset_ag_len"]
            .min().reset_index()
            .sort_values(["_subset_ag_len", "_subset_cluster"])
        )
        chosen = list(order["_subset_cluster"].head(n_clusters))
        if len(chosen) < n_clusters:
            raise ValueError(
                f"smoke subset requested {n_clusters} {source_split} clusters but "
                f"only {len(chosen)} split-exclusive clusters survived filtering"
            )
        part = candidates[candidates["_subset_cluster"].isin(chosen)]
        part = (
            part.sort_values(["_subset_cluster", "_subset_ag_len", "id"])
            .groupby("_subset_cluster", sort=True, group_keys=False)
            .head(rows_per_cluster)
        )
        selected.append(part)

    out = pd.concat(selected, ignore_index=False).sort_values(
        ["_subset_split", "_subset_cluster", "id"]
    )
    print(
        "  smoke subset: "
        f"{out['_subset_cluster'].nunique()} clusters / {len(out)} rows "
        f"(train-source={requests['train']}, test={requests['test']})",
        flush=True,
    )
    return out.drop(columns=["_subset_cluster", "_subset_split", "_subset_ag_len"])


# --- main --------------------------------------------------------------------
def process(standardized_csv: str, processing: dict, records_out: str) -> pd.DataFrame:
    df = pd.read_csv(standardized_csv, dtype=str, keep_default_na=False)
    sc.validate(df, where=standardized_csv)
    print(f"Loaded {len(df)} standardized rows from {standardized_csv}", flush=True)

    def report(mask: pd.Series, label: str) -> pd.Series:
        print(f"  filter {label:<24} keep {int(mask.sum()):>6} / {len(mask)}", flush=True)
        return mask

    mask = pd.Series(True, index=df.index)

    if processing.get("require_paired", True):
        if {"heavy_chain", "light_chain"} <= set(df.columns):
            paired = _nonempty(df["heavy_chain"]) & _nonempty(df["light_chain"])
        else:
            paired = _nonempty(df["expected_heavy_seq"]) & _nonempty(
                df["expected_light_seq"]
            )
        mask &= report(paired, "paired H+L")

    mask &= report(_nonempty(df["antigen_chains"]), "has antigen")

    allowed = {t.upper() for t in processing.get("antigen_types", ["PROTEIN"])}
    if "antigen_types" in df.columns:
        mask &= report(
            df["antigen_types"].apply(lambda v: _all_types_allowed(v, allowed)),
            f"antigen types {sorted(allowed)}",
        )
    else:
        print(
            "  filter antigen_types           skipped (no antigen_types column)",
            flush=True,
        )

    if processing.get("require_structure_file", True):
        exists = df["pdb_path"].apply(lambda p: Path(p).is_file())
        mask &= report(exists, "structure file present")

    df = df[mask].copy()
    if df.empty:
        raise RuntimeError(
            "no rows survived filtering -- check config['processing'] and that "
            "the dataset was fetched correctly"
        )

    df = ensure_resolved(df)

    # Length cap runs after resolution, since it is defined on resolved length.
    max_len = processing.get("max_antigen_len", None)
    if max_len is not None:
        keep = df["resolved_ag_seq"].apply(_total_antigen_len) <= int(max_len)
        print(
            f"  filter antigen_len<={int(max_len):<11} keep {int(keep.sum()):>6} / {len(keep)}",
            flush=True,
        )
        df = df[keep].copy()

    df = select_cluster_subset(
        df, processing.get("subset", {}), processing.get("val", {})
    )

    # Every chain the pipeline touches, heavy/light first then antigen.
    def _chains(row):
        parts = []
        for key in ("heavy_chain", "light_chain"):
            v = str(row.get(key, "")).strip()
            if v:
                parts.append(v)
        parts += [c for c in str(row["antigen_chains"]).split(SEP) if c]
        return SEP.join(parts)

    df["chains"] = df.apply(_chains, axis=1)
    assigned_split = assign_validation(df, processing.get("val", {}))
    validate_cluster_partition(df, assigned_split, processing.get("val", {}))
    df["split"] = assigned_split.values

    extras = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    out = df[OUTPUT_COLUMNS + extras]

    Path(records_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(records_out, index=False)

    counts = out["split"].value_counts().to_dict()
    print(
        f"Wrote {len(out)} records -> {records_out}  "
        f"(train={counts.get('train', 0)}, val={counts.get('val', 0)}, "
        f"test={counts.get('test', 0)})",
        flush=True,
    )
    return out


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        process(
            standardized_csv=smk.input.standardized,
            processing=dict(smk.params.processing),
            records_out=smk.output.records,
        )
        return

    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standardized", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--processing", default="{}", help="JSON processing config")
    a = p.parse_args()
    defaults = {
        "require_paired": True,
        "antigen_types": ["PROTEIN"],
        "max_antigen_len": 1000,
        "require_structure_file": True,
        "val": {
            "strategy": "cluster",
            "fraction": 0.10,
            "seed": 42,
            "cluster_column": "ab_ag_cluster",
        },
    }
    defaults.update(json.loads(a.processing))
    process(a.standardized, defaults, a.out)


if __name__ == "__main__":
    main()
