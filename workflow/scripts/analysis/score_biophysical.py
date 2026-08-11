"""Compute handbook Figure 5 GRAVY and charge-at-pH metrics per design."""

from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def _clean(sequence):
    return "".join(aa for aa in str(sequence).upper() if aa in CANONICAL)


def score_sequence(sequence, charge_ph):
    clean = _clean(sequence)
    if not clean:
        raise ValueError("sequence contains no canonical amino acids")
    analysis = ProteinAnalysis(clean)
    return analysis.gravy(), analysis.charge_at_pH(float(charge_ph))


def score_biophysical(cohort_csv, charge_ph, out_csv):
    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    rows = []
    for row in df.itertuples(index=False):
        base = row._asdict()
        try:
            gravy, charge = score_sequence(row.sequence, charge_ph)
            base.update(gravy=gravy, charge_at_pH=charge, charge_ph=float(charge_ph),
                        metric_status="ok")
        except Exception as exc:
            base.update(gravy=float("nan"), charge_at_pH=float("nan"),
                        charge_ph=float(charge_ph),
                        metric_status=f"error: {type(exc).__name__}: {exc}")
        rows.append(base)
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} biophysical rows -> {out_csv}", flush=True)
    return out


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_biophysical(smk.input.cohort, smk.params.charge_ph, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--charge-ph", type=float, default=7.4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_biophysical(a.cohort, a.charge_ph, a.out)


if __name__ == "__main__":
    main()
