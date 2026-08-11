"""Interface and developability metrics on predicted complexes.  STUB.

Grant Aim 3: "similar to Germinal, I will also quantify these predictions by
interface hydrophobicity, pDockQ2, CDR-specific SAP score, and interface dG
energy relative to native."

Output schema (one row per scored complex)
------------------------------------------
    embedder, run_id, target_id, design_index, sp_method,
    pdockq2, interface_hydrophobicity, cdr_sap, interface_ddg, status

Per-metric notes for whoever implements this
--------------------------------------------
pdockq2                  computable from the predictor's PAE + interface
                         contacts; no extra tooling needed, so this is the
                         cheapest one to land first.
interface_hydrophobicity BioPython/FreeSASA over interface residues; needs a
                         contact-distance cutoff decision (5 A is conventional).
cdr_sap                  spatial aggregation propensity over CDR residues;
                         needs CDR numbering (ANARCI) plus SASA.
interface_ddg            PyRosetta InterfaceAnalyzerMover; licence-gated, and
                         "relative to native" needs the reference complex, so
                         decide whether native means the crystal structure or
                         the predicted wild-type complex.
"""

import sys
from pathlib import Path

import pandas as pd  # noqa: F401  (kept so the output contract is visible)

OUTPUT_COLUMNS = [
    "embedder", "run_id", "target_id", "design_index", "sp_method",
    "pdockq2", "interface_hydrophobicity", "cdr_sap", "interface_ddg", "status",
]


def complex_metrics(scores_csvs, tag: str, metrics: list, out_csv: str) -> None:
    """Score predicted complexes; one row per design per predictor."""
    raise NotImplementedError(
        "Complex-level metrics are not implemented yet.\n"
        f"Requested metrics: {metrics}.\n"
        "Depends on analysis_predict_structures.py producing complexes first.\n"
        "Suggested order: pdockq2 (no new deps) -> interface_hydrophobicity -> "
        "cdr_sap -> interface_ddg."
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        complex_metrics(
            scores_csvs=list(smk.input.scores),
            tag=smk.params.tag,
            metrics=list(smk.params.metrics),
            out_csv=smk.output.metrics,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--metrics", nargs="+", default=["pdockq2"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    complex_metrics(a.scores, a.tag, a.metrics, a.out)


if __name__ == "__main__":
    main()
