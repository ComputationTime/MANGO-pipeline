#!/usr/bin/env python
"""Build the MANGO dataset: download -> standardize -> filter + train/val/test split.

    python build_dataset.py                    # run every stage
    python build_dataset.py --stages process   # rerun only the split stage

Stages (each can be selected with --stages):

  fetch        download and stage SAbDab2 (Zenodo) incl. the antigen-aware
               train/test split table                    scripts/data/fetch_sabdab2.py
  standardize  raw split table -> one dataset-agnostic CSV
                                                         scripts/data/standardize_sabdab2.py
  process      apply the configured filters, resolve sequences from the
               structures and carve a cluster-disjoint validation set out of
               train; writes records.csv (split = train | val | test)
                                                         scripts/data/process_records.py

The held-out *test* split comes from the dataset's antigen-aware split table;
the *validation* split is carved from train by whole antibody-antigen clusters
(config: processing.val). Nothing here needs a GPU.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS)] + [
    str(SCRIPTS / d) for d in ("lib", "data", "embedding", "training")
]

import pipeline as pl  # noqa: E402

STAGES = ["fetch", "standardize", "process"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pl.add_common_args(parser)
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=STAGES,
        help="stages to run (default: all, in pipeline order)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild a stage even when its output already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = pl.Pipeline(pl.load_config(*args.config), args.artifact_root)
    cfg = pipe.config
    ds = cfg["dataset"]
    selected = [stage for stage in STAGES if stage in args.stages]

    if "fetch" in selected:
        import fetch_sabdab2

        if pipe.fetch_marker.is_file() and not args.force:
            print(f"[fetch] already staged: {pipe.splits_dir}", flush=True)
        else:
            structures = fetch_sabdab2.ensure_sabdab2(
                artifact_root=str(pipe.artifact_root),
                version=pipe.dataset_version,
                url=ds["zenodo_url"],
            )
            split_csv = Path(structures) / ds["split_file"]
            if not split_csv.is_file():
                raise SystemExit(f"expected split table is missing: {split_csv}")
            pipe.fetch_marker.write_text(f"{structures}\n", encoding="utf-8")

    if "standardize" in selected:
        import standardize_sabdab2

        pl.require_file(pipe.split_csv, "run the fetch stage first")
        if pipe.standardized_csv.is_file() and not args.force:
            print(f"[standardize] already built: {pipe.standardized_csv}", flush=True)
        else:
            standardize_sabdab2.standardize(
                split_csv=str(pipe.split_csv),
                splits_dir=str(pipe.splits_dir),
                split_column=ds["split_column"],
                out_csv=str(pipe.standardized_csv),
            )

    if "process" in selected:
        import process_records

        pl.require_file(pipe.standardized_csv, "run the standardize stage first")
        pipe.records_csv.parent.mkdir(parents=True, exist_ok=True)
        # Splits are cheap and depend on config; always rebuild unless unchanged
        # inputs are explicitly trusted via --stages skipping this stage.
        process_records.process(
            standardized_csv=str(pipe.standardized_csv),
            processing=dict(cfg["processing"]),
            records_out=str(pipe.records_csv),
        )

    print(f"dataset ready: {pipe.records_csv}", flush=True)


if __name__ == "__main__":
    main()
