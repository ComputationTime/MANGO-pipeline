"""Prepare HER2, PD-1, and TIGIT AlphaSeq tables for FLAb scoring.

Keeps rows with sufficient replicate observations, splits each scFv on the
(GGGGS)3 linker, and averages AlphaSeq log10(KD) across replicate rows.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flab_common import TARGETS, prepare_frame


DATA_FILES = {
    "HER2": "HER2_data.csv",
    "PD1": "PD1_data.csv",
    "TIGIT": "TIGIT_data.csv",
}


def prepare_targets(data_dir: Path, out_dir: Path, targets) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in targets:
        source = data_dir / DATA_FILES[target]
        frame, skipped = prepare_frame(pd.read_csv(source), target)
        prepared_path = out_dir / f"{target}.csv"
        skipped_path = out_dir / f"{target}_skipped.csv"
        frame.to_csv(prepared_path, index=False)
        skipped.to_csv(skipped_path, index=False)
        summaries.append({
            "target": target,
            "source": str(source),
            "antibodies": int(len(frame)),
            "skipped": int(len(skipped)),
            "prepared": str(prepared_path),
        })
        print(
            f"{target}: {len(frame)} antibodies, {len(skipped)} skipped -> {prepared_path}",
            flush=True,
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "prepare_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    args = parser.parse_args()
    prepare_targets(args.data_dir, args.out_dir, args.targets)


if __name__ == "__main__":
    main()
