"""Score generated chains with AntiBERTy's library pseudo-log-likelihood."""

import inspect
import sys
from pathlib import Path

from antiberty import AntiBERTyRunner

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
import chain_common
import device_common as dc


def _runner(device):
    """Load AntiBERTy on the selected device when the installed API allows it."""
    parameters = inspect.signature(AntiBERTyRunner).parameters
    if "device" in parameters:
        return AntiBERTyRunner(device=device)
    return AntiBERTyRunner()


def _library_pseudo_log_likelihood(model, sequence, batch_size):
    """Return AntiBERTyRunner.pseudo_log_likelihood for one sequence."""
    values = model.pseudo_log_likelihood([sequence], batch_size=int(batch_size))
    return float(values.detach().reshape(-1)[0].cpu())


def score_antiberty(cohort_csv, batch_size, out_csv):
    device = dc.get_device("AntiBERTy scoring")
    df = chain_common.read_chain_rows(cohort_csv)
    model = _runner(device)
    scores, statuses = [], []
    for sequence in df["sequence"]:
        try:
            scores.append(_library_pseudo_log_likelihood(model, sequence, batch_size))
            statuses.append("ok")
        except Exception as exc:
            scores.append(float("nan"))
            statuses.append(f"error: {type(exc).__name__}: {exc}")
    df["antiberty_pseudo_log_likelihood"] = scores
    df["antiberty_status"] = statuses
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"AntiBERTy scored {sum(s == 'ok' for s in statuses)}/{len(df)} "
          f"sequences on {model.device} -> {out_csv}", flush=True)
    if any(status != "ok" for status in statuses):
        raise RuntimeError(
            "AntiBERTy did not score every selected sequence; diagnostics were "
            f"written to {out_csv}"
        )
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_antiberty(smk.input.cohort, smk.params.batch_size, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_antiberty(a.cohort, a.batch_size, a.out)


if __name__ == "__main__":
    main()
