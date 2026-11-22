#!/usr/bin/env python
"""Master embedding script: compute antigen embeddings for any/all embedders.

    python run_embedding.py --embedders one_hot esm2     # a chosen subset
    python run_embedding.py --esm2 --proteinmpnn          # per-embedder flags
    python run_embedding.py --all                         # every embedder

Embedders: one_hot, biopython, pyrosetta_pre, esm2, esm3, esm3_structure, esmif,
proteinmpnn.
With no selection flag the config's ``active_embedders`` list is used.

Each embedder writes one ``.pt`` per structure beneath
``<embeddings>/antigen/<tag>/{train,val,test}/`` plus ``embedder_config.json``.
The implementations live in ``scripts/embedding/`` and the dataset
(``records.csv``) must already exist -- run ``build_dataset.py`` first.

Every embedder runs in its own subprocess so one failure cannot stop the rest,
and so conflicting dependency stacks (ESM2 vs ESM3, PyRosetta, ...) can use
different Python environments:

    python run_embedding.py --all \\
        --python esm3=/path/to/envs/embed_esm3/bin/python \\
        --python pyrosetta_pre=/path/to/envs/embed_pyrosetta/bin/python

Embedders without a ``--python`` override use the current interpreter. ESM3
needs ``HF_TOKEN`` (gated weights) and PyRosetta its own licensed install.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS)] + [
    str(SCRIPTS / d) for d in ("lib", "data", "embedding", "training")
]

import pipeline as pl  # noqa: E402

EMBED_DIR = SCRIPTS / "embedding"
# Methods whose pretrained weights are cached by a package, not a file we own.
PREFETCH = {"esm2", "esm3", "esmif", "pyrosetta_pre"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pl.add_common_args(parser)
    pl.add_embedder_args(parser)
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="splits to embed (default: embedding.splits from the config)",
    )
    parser.add_argument(
        "--python", action="append", default=[], metavar="TAG=PATH",
        help="interpreter for one embedder's environment (repeatable)",
    )
    parser.add_argument(
        "--skip-weights", action="store_true",
        help="do not pre-download pretrained weights before embedding",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running",
    )
    parser.add_argument(
        "--list", action="store_true", help="list registered embedders and exit",
    )
    return parser.parse_args()


def _model_name(method, spec):
    if method == "esm2":
        return f"esm2_{spec['size']}_UR50D"
    if method in {"esm3", "esmif"}:
        return spec["model_name"]
    if method == "pyrosetta_pre":
        return "pyrosetta-quarterly"
    return method


def _run(command, dry_run):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], check=True)


def embed_one(pipe, tag, python, splits, skip_weights, dry_run):
    import embed_common as ec

    spec = dict(pipe.spec(tag))
    method = spec["method"]
    if method not in pl.EMBEDDABLE_METHODS:
        raise SystemExit(f"{tag}: method {method!r} is not implemented")
    out_dir = pipe.antigen_emb_dir(tag)
    weights = None

    if method == "proteinmpnn":
        import fetch_proteinmpnn_weights

        weights = pipe.proteinmpnn_weights(tag)
        if not dry_run:
            fetch_proteinmpnn_weights.fetch(
                spec["model"], int(spec["noise"]),
                pipe.config["weights"]["proteinmpnn_repo_raw"], str(weights),
            )
    elif method in PREFETCH and not skip_weights:
        _run([
            python, EMBED_DIR / "prefetch_embedder_weights.py",
            "--method", method, "--model-name", _model_name(method, spec),
            "--tag", tag, "--out", pipe.pretrained_marker(tag),
            "--representation", spec.get("representation", "sequence"),
        ], dry_run)

    if not dry_run:
        ec.write_embedder_config(
            str(out_dir / "embedder_config.json"), tag, spec, pipe.seq_source
        )

    # PRE, ESM-IF and ProteinMPNN read the structure, not the sequence column.
    seq_source = "structure" if method in {"pyrosetta_pre", "esmif", "proteinmpnn"} \
        else pipe.seq_source
    command = [
        python, EMBED_DIR / "embed_batch.py",
        "--records", pipe.records_csv,
        "--output-dir", out_dir,
        "--tag", tag,
        "--method", method,
        "--spec", json.dumps(spec, sort_keys=True),
        "--seq-source", seq_source,
        "--splits", ",".join(splits),
    ]
    if weights is not None:
        command += ["--weights", weights]
    _run(command, dry_run)


def main():
    args = parse_args()
    pipe = pl.Pipeline(pl.load_config(*args.config), args.artifact_root)

    if args.list:
        for tag, spec in pipe.embedders.items():
            ok = "" if spec["method"] in pl.EMBEDDABLE_METHODS else "  (not implemented)"
            print(f"{tag:<16} {spec['method']:<14} {spec.get('class', '')}{ok}")
        return

    tags = pipe.select_embedders(args)
    pythons = {}
    for item in args.python:
        key, _, path = item.partition("=")
        if key not in pipe.embedders or not path:
            raise SystemExit(f"--python expects TAG=PATH with a known tag, got {item!r}")
        pythons[key] = path
    splits = args.splits or pipe.embed_splits
    if not args.dry_run:
        pl.require_file(pipe.records_csv, "run `python build_dataset.py` first")

    print(f"embedders: {', '.join(tags)}", flush=True)
    failed = {}
    for tag in tags:
        started = time.time()
        print(f"\n=== embedding {tag} ===", flush=True)
        try:
            embed_one(
                pipe, tag, pythons.get(tag, sys.executable), splits,
                args.skip_weights, args.dry_run,
            )
            print(f"[{tag}] done in {time.time() - started:.0f}s", flush=True)
        except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
            failed[tag] = f"{type(exc).__name__}: {exc}"
            print(f"[{tag}] FAILED: {failed[tag]}", flush=True)

    print("\n=== embedding summary ===")
    for tag in tags:
        print(f"{tag:<16} {'FAILED  ' + failed[tag] if tag in failed else 'ok'}")
    if failed:
        raise SystemExit(f"{len(failed)} embedder(s) failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
