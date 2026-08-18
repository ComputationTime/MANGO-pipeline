"""Score generated heavy chains with IgLM mean sequence log likelihood."""

from pathlib import Path
import sys

import pandas as pd
from iglm import IgLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import device_common as dc


def score_iglm(cohort_csv, chain_token, species_token, out_csv):
    dc.get_device("IgLM scoring")
    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    model = IgLM()
    scores, statuses = [], []
    for sequence in df["sequence"]:
        try:
            scores.append(float(model.log_likelihood(sequence, chain_token, species_token)))
            statuses.append("ok")
        except Exception as exc:
            scores.append(float("nan"))
            statuses.append(f"error: {type(exc).__name__}: {exc}")
    df["iglm_log_likelihood"] = scores
    df["iglm_status"] = statuses
    df["iglm_chain_token"] = chain_token
    df["iglm_species_token"] = species_token
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"IgLM scored {sum(s == 'ok' for s in statuses)}/{len(df)} -> {out_csv}", flush=True)
    if any(status != "ok" for status in statuses):
        raise RuntimeError(
            "IgLM did not score every selected sequence; diagnostics were "
            f"written to {out_csv}"
        )
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_iglm(smk.input.cohort, smk.params.chain_token,
                   smk.params.species_token, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--chain-token", default="[HEAVY]")
    p.add_argument("--species-token", default="[HUMAN]")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_iglm(a.cohort, a.chain_token, a.species_token, a.out)


if __name__ == "__main__":
    main()
