"""Fold designs in complex with the intended target and score them.  STUB.

Grant Aim 3 / handbook figure 2. For each design, build a complex with the
target antigen, predict it with AF3 / Boltz2 / Chai, and lift pTM, ipTM and PAE
from the predictor's output.

Output schema (what analysis_complex_metrics and fig2 expect)
------------------------------------------------------------
    embedder, run_id, target_id, sp_method, design_index, sequence,
    ptm, iptm, pae, status, structure_path

Why this is a stub
------------------
All three predictors need infrastructure that cannot be assumed present:
  * AF3    -- weights are access-gated; needs GPU and sequence databases
  * Boltz2 -- pip-installable but needs GPU + downloaded weights
  * Chai   -- pip-installable but needs GPU + downloaded weights

The tractable first step is Boltz2 or Chai (no gated weights), which would let
figure 2 render with two of its three panels while AF3 access is arranged.

Recommended design when implementing: run each predictor OUTSIDE this rule and
have this script only INGEST its outputs. Prediction runs are long and fail in
ways Snakemake should not have to retry; ingesting a directory of results keeps
the DAG restartable and makes it trivial to swap a predictor version.

Note the scale: config caps this at structure_prediction.n_designs (30 per
handbook figure 2) rather than all 10K designs per target -- folding everything
is not affordable. The cap is applied here, and which designs were selected is
recorded in the output so the sampling is auditable.
"""

import sys
from pathlib import Path

import pandas as pd  # noqa: F401  (kept so the output contract is visible)

OUTPUT_COLUMNS = [
    "embedder", "run_id", "target_id", "sp_method", "design_index",
    "sequence", "ptm", "iptm", "pae", "status", "structure_path",
]

SUPPORTED = {"af3", "boltz2", "chai"}


def predict_structures(
    designs_csv: str,
    target_cif: str,
    method: str,
    tag: str,
    n_designs: int,
    scores: list,
    out_csv: str,
) -> None:
    """Predict complexes for the first `n_designs` designs and score them."""
    if method not in SUPPORTED:
        raise ValueError(f"unknown structure predictor {method!r}; expected {sorted(SUPPORTED)}")

    raise NotImplementedError(
        f"Structure prediction backend {method!r} is not implemented yet.\n"
        f"Would fold {n_designs} designs from {designs_csv} against {target_cif} "
        f"and report {scores}.\n"
        "Blocked on: predictor weights, GPU, and the ingest-vs-orchestrate "
        "decision (see this file's docstring).\n"
        "Disable analysis.plots.fig2_ptm in config/config.yaml to build the "
        "other figures."
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        predict_structures(
            designs_csv=smk.input.designs,
            target_cif=smk.input.target,
            method=smk.params.method,
            tag=smk.params.tag,
            n_designs=int(smk.params.n_designs),
            scores=list(smk.params.scores),
            out_csv=smk.output.scores,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--designs", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--method", required=True, choices=sorted(SUPPORTED))
    p.add_argument("--tag", required=True)
    p.add_argument("--n-designs", type=int, default=30)
    p.add_argument("--scores", nargs="+", default=["ptm", "iptm", "pae"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    predict_structures(
        a.designs, a.target, a.method, a.tag, a.n_designs, a.scores, a.out
    )


if __name__ == "__main__":
    main()
