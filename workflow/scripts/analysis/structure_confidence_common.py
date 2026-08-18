"""Shared selection, input, and output helpers for structure-confidence backends."""

import csv
import hashlib
from pathlib import Path

import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = [
    "embedder", "run_id", "target_id", "design_index", "sequence",
    "predictor", "sample_index", "confidence_score", "ptm", "iptm",
    "complex_plddt", "mean_pae", "has_inter_chain_clashes",
    "structure_path", "status",
]


def _rank(seed, target_id, design_index):
    value = f"{int(seed)}\0{target_id}\0{int(design_index)}".encode()
    return hashlib.sha256(value).hexdigest()


def clean_sequence(value):
    seq = str(value).strip().upper()
    if not seq or any(aa not in CANONICAL for aa in seq):
        raise ValueError("sequence must contain only canonical amino acids")
    return seq


def select_designs(designs_csv, n_designs, seed):
    df = pd.read_csv(designs_csv, dtype={"sequence": str}, keep_default_na=False)
    required = {"embedder", "run_id", "target_id", "design_index", "sequence", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{designs_csv} is missing columns {sorted(missing)}")
    df = df.loc[df["status"] == "ok"].copy()
    if df.empty:
        raise RuntimeError(f"{designs_csv} contains no successful designs")
    requested = int(n_designs)
    if len(df) < requested:
        raise RuntimeError(
            f"{designs_csv} contains only {len(df)} successful designs; "
            f"{requested} are required"
        )
    df["design_index"] = pd.to_numeric(df["design_index"], errors="raise").astype(int)
    df["selection_key"] = [
        _rank(seed, target, index)
        for target, index in zip(df["target_id"], df["design_index"])
    ]
    return df.sort_values(["selection_key", "design_index"]).head(requested)


def target_context(records_csv, target_id):
    df = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    hit = df.loc[df["id"] == target_id]
    if len(hit) != 1:
        raise ValueError(f"expected exactly one record for target {target_id!r}, found {len(hit)}")
    row = hit.iloc[0]
    antigen = [clean_sequence(s) for s in row["resolved_ag_seq"].split(",") if s]
    light = clean_sequence(row["resolved_L_seq"])
    if not antigen:
        raise ValueError(f"target {target_id!r} has no antigen sequence")
    return antigen, light


def complex_chains(heavy, light, antigen):
    return [("H", clean_sequence(heavy)), ("L", light)] + [
        (f"AG{i + 1}", seq) for i, seq in enumerate(antigen)
    ]


def write_rows(rows, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
