"""Score generated heavy chains with IgLM mean sequence log likelihood."""

from pathlib import Path
import sys

from iglm import IgLM

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
import chain_common
import device_common as dc


def score_iglm(
    cohort_csv,
    chain_token,
    species_token,
    out_csv,
    light_chain_token="[LIGHT]",
):
    dc.get_device("IgLM scoring")
    df = chain_common.read_chain_rows(cohort_csv)
    model = IgLM()
    scores, statuses, chain_tokens = [], [], []
    for sequence, chain_type in zip(df["sequence"], df["chain_type"]):
        selected_token = chain_token if chain_type == "Heavy" else light_chain_token
        chain_tokens.append(selected_token)
        try:
            scores.append(
                float(model.log_likelihood(sequence, selected_token, species_token))
            )
            statuses.append("ok")
        except Exception as exc:
            scores.append(float("nan"))
            statuses.append(f"error: {type(exc).__name__}: {exc}")
    df["iglm_log_likelihood"] = scores
    df["iglm_status"] = statuses
    df["iglm_chain_token"] = chain_tokens
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
    p.add_argument("--light-chain-token", default="[LIGHT]")
    p.add_argument("--species-token", default="[HUMAN]")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_iglm(
        a.cohort,
        a.chain_token,
        a.species_token,
        a.out,
        light_chain_token=a.light_chain_token,
    )


if __name__ == "__main__":
    main()
