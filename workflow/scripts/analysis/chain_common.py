"""Normalize antibody score inputs to one independently scored chain per row."""

import pandas as pd


CHAIN_TYPE_MAP = {
    "h": "Heavy",
    "heavy": "Heavy",
    "l": "Light",
    "light": "Light",
}


def read_chain_rows(path):
    """Read long or wide antibody CSV data as separate Heavy and Light rows."""
    frame = pd.read_csv(path, dtype={"sequence": str}, keep_default_na=False)
    if "sequence" not in frame:
        raise ValueError("Antibody scoring requires a 'sequence' column")

    if "chain_type" in frame:
        normalized = (
            frame["chain_type"].astype(str).str.strip().str.lower().map(CHAIN_TYPE_MAP)
        )
        if normalized.isna().any():
            invalid = sorted(set(frame.loc[normalized.isna(), "chain_type"].astype(str)))
            raise ValueError(
                "Unsupported chain_type values: " + ", ".join(invalid)
            )
        frame["chain_type"] = normalized
    else:
        heavy = frame.copy()
        heavy["chain_type"] = "Heavy"
        rows = [heavy]
        if "light_sequence" in frame:
            has_light = frame["light_sequence"].astype(str).str.strip().ne("")
            if has_light.any():
                light = frame.loc[has_light].copy()
                light["sequence"] = light["light_sequence"]
                if "light_length" in light:
                    light["length"] = light["light_length"]
                else:
                    light["length"] = light["sequence"].str.len()
                light["chain_type"] = "Light"
                rows.append(light)
        frame = pd.concat(rows, ignore_index=True)

    frame["sequence"] = frame["sequence"].astype(str).str.strip()
    empty = frame["sequence"].eq("")
    if empty.any():
        counts = frame.loc[empty, "chain_type"].value_counts().to_dict()
        raise ValueError(f"Antibody scoring found empty chain sequences: {counts}")

    # Remove paired-input columns so downstream scorers cannot accidentally use
    # one chain as context for another.
    return frame.drop(
        columns=["light_sequence", "light_length"], errors="ignore"
    ).reset_index(drop=True)
