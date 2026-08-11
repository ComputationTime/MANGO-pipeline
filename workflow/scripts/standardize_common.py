"""The standardized-table contract, shared by every dataset adapter.

A standardizer's only job is to rename/reshape a raw dataset into these columns
so that nothing downstream needs dataset-specific knowledge.

Required columns
----------------
id                  unique row id; also the stem of the embedding filenames
pdb_path            path to the structure file for this row
antigen_chains      comma-separated author chain ids of the ANTIGEN chains
expected_heavy_seq  full annotated heavy sequence   (may be empty)
expected_light_seq  full annotated light sequence   (may be empty)
expected_ag_seq     full annotated antigen sequence, comma-separated per chain
                    in the same order as antigen_chains (may be empty)
split               train/val/test if the source provides one (may be empty;
                    `process` fills or overrides it)

Optional passthrough
--------------------
Adapters may emit extra columns (resolved sequences, chain types, resolution,
cluster ids, ...). `process` consumes them when present -- e.g. SAbDab2 already
ships resolved sequences, so we avoid re-parsing thousands of structures -- and
falls back to parsing the structure file when they are absent.
"""

REQUIRED_COLUMNS = [
    "id",
    "pdb_path",
    "antigen_chains",
    "expected_heavy_seq",
    "expected_light_seq",
    "expected_ag_seq",
    "split",
]

# Multi-chain antigen fields use this separator in the standardized table.
SEP = ","


def validate(df, where: str = "standardized table"):
    """Raise if the contract is violated; return the frame unchanged."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{where} is missing required column(s): {missing}")

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].unique()[:5].tolist()
        raise ValueError(f"{where} has duplicate id(s), e.g. {dupes}")

    for col in ("id", "pdb_path"):
        blank = df[col].astype(str).str.strip() == ""
        if blank.any():
            raise ValueError(f"{where} has {int(blank.sum())} empty {col} value(s)")

    # antigen_chains vs expected_ag_seq must agree in arity where both present.
    both = (df["antigen_chains"].astype(str).str.strip() != "") & (
        df["expected_ag_seq"].astype(str).str.strip() != ""
    )
    if both.any():
        n_chains = df.loc[both, "antigen_chains"].str.split(SEP).str.len()
        n_seqs = df.loc[both, "expected_ag_seq"].str.split(SEP).str.len()
        bad = n_chains != n_seqs
        if bad.any():
            example = df.loc[both][bad].iloc[0]["id"]
            raise ValueError(
                f"{where}: antigen_chains and expected_ag_seq disagree in length "
                f"for {int(bad.sum())} row(s), e.g. id={example!r}"
            )
    return df


def order_columns(df):
    """Contract columns first, dataset passthrough columns after."""
    extras = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[REQUIRED_COLUMNS + extras]
