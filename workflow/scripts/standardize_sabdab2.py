"""SAbDab2 -> standardized table.

Pure renaming/reshaping: no filtering, no validation-set carving, no structure
parsing. Those belong to `process`, which runs next. Keeping this stage dumb is
what makes swapping in another dataset a one-file change.

SAbDab2 separates multi-chain antigen fields with "/"; the standardized table
uses "," throughout (standardize_common.SEP).

The generation target is the IMGT-numberable variable heavy domain (VH), not
the full deposited heavy chain.  SAbDab2 supplies that target directly as
``VH_numerable_seq``.  Both expected_heavy_seq and resolved_H_seq use this same
VH sequence so training, evaluation, reconstruction and generation share one
unambiguous target contract.  The deposited full-heavy sequences are retained
under source_* provenance columns but never become model targets.

SAbDab2 also ships structurally resolved light and antigen sequences. `process`
prefers these passthrough columns over re-parsing the .cif files, which saves
parsing ~15k structures.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import standardize_common as sc

SRC_SEP = "/"

# SAbDab2 column -> standardized column, for the required contract fields.
REQUIRED_SRC = {
    "INSTANCE": "id",
    "agchains": "antigen_chains",
    "VH_numerable_seq": "expected_heavy_seq",
    "Lseq_expected": "expected_light_seq",
    "agexpectedseqs": "expected_ag_seq",
}

# Optional passthrough: kept when present, silently skipped when absent.
PASSTHROUGH_SRC = {
    "PDB_ID": "pdb_id",
    "Hchain": "heavy_chain",
    "Lchain": "light_chain",
    "agtypes": "antigen_types",
    "VH_numerable_seq": "resolved_H_seq",
    "Hseq": "source_resolved_full_heavy_seq",
    "Hseq_expected": "source_expected_full_heavy_seq",
    "Lseq": "resolved_L_seq",
    "agresolvedseqs": "resolved_ag_seq",
    "type": "ab_type",
    "resolution": "resolution",
    "ab_ag_cluster": "ab_ag_cluster",
}

# Columns whose values are multi-chain lists needing separator translation.
LIST_COLUMNS = {"antigen_chains", "antigen_types", "expected_ag_seq", "resolved_ag_seq"}


def _resep(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(SRC_SEP, sc.SEP, regex=False)


def standardize(
    split_csv: str,
    splits_dir: str,
    split_column: str,
    out_csv: str,
) -> pd.DataFrame:
    df = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    print(f"Loaded {len(df)} rows from {split_csv}", flush=True)

    missing = [c for c in REQUIRED_SRC if c not in df.columns]
    if missing:
        raise ValueError(
            f"{split_csv} is missing expected SAbDab2 column(s) {missing}. "
            f"Found: {sorted(df.columns)}"
        )
    if split_column not in df.columns:
        raise ValueError(
            f"split column {split_column!r} not in {split_csv}; "
            f"found: {sorted(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)
    for src, dst in REQUIRED_SRC.items():
        out[dst] = df[src]

    # One structure file per instance, named by instance id.
    out["pdb_path"] = df["INSTANCE"].apply(
        lambda inst: str(Path(splits_dir) / f"{inst}.cif")
    )
    # Source split is train/test only; `process` carves val out of train.
    out["split"] = df[split_column].astype(str).str.strip().str.lower()

    for src, dst in PASSTHROUGH_SRC.items():
        if src in df.columns:
            out[dst] = df[src]

    for col in LIST_COLUMNS & set(out.columns):
        out[col] = _resep(out[col])

    out = sc.order_columns(sc.validate(out, where=out_csv))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    counts = out["split"].value_counts().to_dict()
    print(f"Wrote {len(out)} standardized rows -> {out_csv}  splits={counts}", flush=True)
    return out


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        standardize(
            split_csv=smk.params.split_csv,
            splits_dir=smk.params.splits_dir,
            split_column=smk.params.split_column,
            out_csv=smk.output.standardized,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--split-column", default="ab_ag_split")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    standardize(a.split_csv, a.splits_dir, a.split_column, a.out)


if __name__ == "__main__":
    main()
