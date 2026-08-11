"""Select a deterministic, auditable cohort of successful heavy-chain designs."""

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED = {"embedder", "run_id", "target_id", "design_index", "sequence", "status"}


def _rank(seed: int, target_id: str, design_index: int) -> str:
    payload = f"{seed}\0{target_id}\0{int(design_index)}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_cohort(design_csvs, records_csv, tag, n_per_target, seed, out_csv):
    records = pd.read_csv(records_csv, dtype=str, keep_default_na=False)
    light_by_id = dict(zip(records["id"], records["resolved_L_seq"]))
    frames = []

    for path in design_csvs:
        df = pd.read_csv(path, dtype={"sequence": str}, keep_default_na=False)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing cohort columns: {sorted(missing)}")
        df = df.loc[df["status"] == "ok"].copy()
        if df.empty:
            raise RuntimeError(f"{path} contains no successful designs")
        if set(df["embedder"]) != {tag}:
            raise ValueError(f"{path} does not contain only embedder {tag!r}")

        df["design_index"] = pd.to_numeric(df["design_index"], errors="raise").astype(int)
        if df.duplicated(["target_id", "design_index"]).any():
            raise ValueError(f"{path} has duplicate target_id/design_index rows")
        df["selection_key"] = [
            _rank(seed, target, index)
            for target, index in zip(df["target_id"], df["design_index"])
        ]
        df = df.sort_values(["target_id", "selection_key", "design_index"])
        if n_per_target is not None:
            df = df.groupby("target_id", sort=True, group_keys=False).head(int(n_per_target))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"[{tag}] no design files were supplied")
    cohort = pd.concat(frames, ignore_index=True)
    cohort["light_sequence"] = cohort["target_id"].map(light_by_id).fillna("")
    if (cohort["light_sequence"] == "").any():
        targets = sorted(cohort.loc[cohort["light_sequence"] == "", "target_id"].unique())
        raise ValueError(f"missing light-chain context for cohort target(s): {targets}")
    cohort["cohort_seed"] = int(seed)
    cohort["cohort_rank"] = cohort.groupby("target_id").cumcount()
    cohort = cohort.sort_values(["target_id", "cohort_rank"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_csv, index=False)
    counts = cohort.groupby("target_id").size().to_dict()
    print(f"[{tag}] selected {len(cohort)} designs: {counts} -> {out_csv}", flush=True)
    return cohort


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        limit = smk.params.n_per_target
        select_cohort(
            list(smk.input.designs), smk.input.records, smk.params.tag,
            None if limit is None else int(limit), int(smk.params.seed), smk.output.cohort,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-per-target", type=int)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    select_cohort(a.designs, a.records, a.tag, a.n_per_target, a.seed, a.out)


if __name__ == "__main__":
    main()
