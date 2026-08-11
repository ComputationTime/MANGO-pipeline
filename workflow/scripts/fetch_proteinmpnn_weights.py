"""Download a ProteinMPNN weight file from the official GitHub repo.

Maps the config model name to the repo's weight directory and pulls
``v_48_<NNN>.pt`` for the requested noise level (2/10/20/30 -> 002/010/020/030).
Atomic + idempotent.
"""

import sys
import urllib.request
from pathlib import Path

MODEL_DIR = {
    "vanilla_models": "vanilla_model_weights",
    "ca_models": "ca_model_weights",
    "soluble_models": "soluble_model_weights",
}


def fetch(model: str, noise: int, repo_raw: str, out: str) -> None:
    if model not in MODEL_DIR:
        raise ValueError(f"unknown ProteinMPNN model {model!r}")
    dest = Path(out)
    if dest.is_file():
        print(f"ProteinMPNN weights already present: {dest}", flush=True)
        return

    url = f"{repo_raw}/{MODEL_DIR[model]}/v_48_{int(noise):03d}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.unlink(missing_ok=True)

    print(f"Downloading ProteinMPNN weights: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MANGO/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, partial.open("wb") as f:
            f.write(r.read())
        partial.replace(dest)
        print(f"saved -> {dest}", flush=True)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        fetch(smk.params.model, smk.params.noise, smk.params.repo_raw, smk.output[0])
        return
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vanilla_models")
    p.add_argument("--noise", type=int, default=2)
    p.add_argument(
        "--repo-raw",
        default="https://raw.githubusercontent.com/dauparas/ProteinMPNN/main",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fetch(a.model, a.noise, a.repo_raw, a.out)


if __name__ == "__main__":
    main()
