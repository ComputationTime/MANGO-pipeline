"""LEGACY combined analysis prototype; not included by the active Snakefile.

Per-design sequence screening metrics (grant Aim 2, handbook figs 3-5).

Reads every designs CSV for one run and emits one row per design with the
in-silico screening metrics. Metrics split into two tiers:

CORE (always computed, pure BioPython)
    gravy, isoelectric_point, charge_at_pH, aromaticity, instability_index,
    aliphatic_index, molecular_weight, frac_charged, frac_hydrophobic

OPTIONAL (need extra models/tools; controlled by config.analysis.seq_metrics)
    ld_germline        ANARCI + germline reference   -> fig 4
    ablang2_confidence AbLang2                       -> fig 3
    iglm_confidence    IgLM                          -> fig 3
    tap_*              therapeutic-antibody-profiler -> fig 5

An unavailable optional metric does NOT fail the run: the column is emitted as
NaN and a `<metric>_status` column records why, with a loud warning in the log.
That keeps the core figures reproducible on a laptop while making it impossible
to mistake a missing metric for a computed zero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
CHARGED = set("DEKR")            # net-charge carriers at physiological pH
HYDROPHOBIC = set("AVILMFWC")    # Kyte-Doolittle positive set

CORE_COLUMNS = [
    "gravy", "isoelectric_point", "charge_at_pH", "aromaticity",
    "instability_index", "aliphatic_index", "molecular_weight",
    "frac_charged", "frac_hydrophobic",
]


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in CANONICAL)


def _aliphatic_index(seq: str) -> float:
    """Ikai's aliphatic index: A + 2.9*V + 3.9*(I + L), in mole percent.

    Not provided by BioPython, but it is one of the grant's named screening
    metrics, so it is computed here rather than dropped.
    """
    n = len(seq)
    if n == 0:
        return float("nan")
    mol = {aa: 100.0 * seq.count(aa) / n for aa in "AVIL"}
    return mol["A"] + 2.9 * mol["V"] + 3.9 * (mol["I"] + mol["L"])


def _core_metrics(seq: str, charge_ph: float) -> dict:
    clean = _clean(seq)
    if not clean:
        return {c: float("nan") for c in CORE_COLUMNS}

    x = ProteinAnalysis(clean)
    n = len(clean)
    return {
        "gravy": x.gravy(),
        "isoelectric_point": x.isoelectric_point(),
        "charge_at_pH": x.charge_at_pH(charge_ph),
        "aromaticity": x.aromaticity(),
        "instability_index": x.instability_index(),
        "aliphatic_index": _aliphatic_index(clean),
        "molecular_weight": x.molecular_weight(),
        "frac_charged": sum(clean.count(a) for a in CHARGED) / n,
        "frac_hydrophobic": sum(clean.count(a) for a in HYDROPHOBIC) / n,
    }


# --- optional metrics --------------------------------------------------------
def _add_optional(df: pd.DataFrame, seq_cfg: dict) -> pd.DataFrame:
    """Attach optional metric columns, degrading loudly when unavailable."""

    def unavailable(name: str, reason: str):
        print(
            f"  !! {name}: UNAVAILABLE -- {reason}\n"
            f"     column emitted as NaN; figures depending on it will be empty.",
            flush=True,
        )
        df[name] = np.nan
        df[f"{name}_status"] = reason

    germline_cfg = seq_cfg.get("germline", {}) or {}
    if germline_cfg.get("enabled", False):
        try:
            import anarci  # noqa: F401

            raise NotImplementedError(
                "ANARCI is installed but germline assignment is not wired up yet"
            )
        except ImportError:
            unavailable("ld_germline", "ANARCI not installed (pip install anarci)")
        except NotImplementedError as e:
            unavailable("ld_germline", str(e))

    if seq_cfg.get("ablang2_confidence", False):
        try:
            import ablang2  # noqa: F401

            raise NotImplementedError(
                "AbLang2 is installed but per-design confidence is not wired up yet"
            )
        except ImportError:
            unavailable("ablang2_confidence", "ablang2 not installed")
        except NotImplementedError as e:
            unavailable("ablang2_confidence", str(e))

    if seq_cfg.get("iglm_confidence", False):
        try:
            import iglm  # noqa: F401

            raise NotImplementedError(
                "IgLM is installed but per-design confidence is not wired up yet"
            )
        except ImportError:
            unavailable("iglm_confidence", "iglm not installed (pip install iglm)")
        except NotImplementedError as e:
            unavailable("iglm_confidence", str(e))

    if seq_cfg.get("tap", False):
        unavailable("tap_score", "therapeutic-antibody-profiler not wired up yet")

    return df


# --- main --------------------------------------------------------------------
def seq_metrics(designs_csvs, tag: str, label: str, seq_cfg: dict, out_csv: str):
    frames = [pd.read_csv(p, dtype={"sequence": str}, keep_default_na=False)
              for p in designs_csvs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise RuntimeError(f"[{tag}] no designs found in {list(designs_csvs)}")

    n_total = len(df)
    df = df[df["status"] == "ok"].copy()
    print(
        f"[{tag}] scoring {len(df)}/{n_total} successful designs "
        f"across {len(designs_csvs)} target(s)",
        flush=True,
    )
    if df.empty:
        raise RuntimeError(f"[{tag}] every design failed generation; nothing to score")

    charge_ph = float(seq_cfg.get("charge_ph", 7.4))
    core = pd.DataFrame(
        [_core_metrics(s, charge_ph) for s in df["sequence"]], index=df.index
    )
    df = pd.concat([df, core], axis=1)
    df["embedder_label"] = label
    df["charge_ph"] = charge_ph

    df = _add_optional(df, seq_cfg)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"[{tag}] wrote {len(df)} scored designs -> {out_csv}\n"
        f"        median GRAVY={df['gravy'].median():.3f} "
        f"charge@{charge_ph}={df['charge_at_pH'].median():.2f} "
        f"frac_charged={df['frac_charged'].median():.3f}",
        flush=True,
    )
    return df


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        seq_metrics(
            designs_csvs=list(smk.input.designs),
            tag=smk.params.tag,
            label=smk.params.label,
            seq_cfg=dict(smk.params.seq_cfg),
            out_csv=smk.output.metrics,
        )
        return

    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--seq-cfg", default="{}")
    a = p.parse_args()
    cfg = {"charge_ph": 7.4}
    cfg.update(json.loads(a.seq_cfg))
    seq_metrics(a.designs, a.tag, a.label or a.tag, cfg, a.out)


if __name__ == "__main__":
    main()
