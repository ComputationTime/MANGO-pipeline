#!/usr/bin/env python
"""Master analysis script: score the generated designs and render every plot.

    python run_analysis.py --all                       # score + plot everything
    python run_analysis.py --embedders one_hot esm2    # a chosen subset
    python run_analysis.py --stages plots              # re-plot from saved scores
    python run_analysis.py --stages plots --only fig4  # a single figure

Stages (``--stages`` picks a subset; default is all of them):

  cohort     deterministic, leakage-safe design subset per embedder
                                                  analysis/select_cohort.py
  score      sequence scorers on that cohort (``--scorers`` picks a subset):
               biophysical  six developability properties   score_biophysical.py
               germline     ANARCI nearest germline + LD    score_germline.py
               iglm / antiberty / ablang2  antibody likeness
  structure  Boltz-2 / Chai-1 confidence + Rosetta interface QC
                                                  run_boltz/chai_confidence.py,
                                                  run_rosetta_interface.py
  plots      every figure, written to artifacts/analysis/figures/
                                                  analysis/generate_figures.py

Figures: fig1 NLL | fig2 structure confidence | fig3 antibody likeness |
fig4 germline distance (single violin plot, all embedders) | fig5 developability
violins | fig6 V/J gene families + species | rosetta interface.

Scorers with conflicting dependencies run as subprocesses, so each can use its
own environment, e.g. ANARCI usually lives apart from the plotting stack:

    python run_analysis.py --all \\
        --python germline=/opt/anaconda3/bin/python \\
        --python ablang2=/path/to/envs/analysis_ablang2/bin/python

Scorers without an override use the current interpreter. Training and design
generation (``run_training.py``) must have finished first.
"""

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS)] + [
    str(SCRIPTS / d) for d in ("lib", "data", "embedding", "training", "analysis")
]

import pipeline as pl  # noqa: E402

ANALYSIS = SCRIPTS / "analysis"
STAGES = ["cohort", "score", "structure", "plots"]
SCORERS = ["biophysical", "germline", "iglm", "antiberty", "ablang2"]
PREDICTORS = {"boltz2": "run_boltz_confidence.py", "chai": "run_chai_confidence.py"}
# Python overrides may also name the structure tools.
PYTHON_KEYS = SCORERS + ["boltz2", "chai", "rosetta"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pl.add_common_args(parser)
    pl.add_embedder_args(parser)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--scorers", nargs="+", choices=SCORERS, default=SCORERS)
    parser.add_argument("--skip-structure", action="store_true",
                        help="shortcut for dropping the structure stage")
    parser.add_argument("--only", nargs="+",
                        choices=["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "rosetta"],
                        help="render only these figures")
    parser.add_argument("--python", action="append", default=[], metavar="NAME=PATH",
                        help=f"interpreter for a tool ({', '.join(PYTHON_KEYS)}); repeatable")
    parser.add_argument("--ncpu", type=int, default=4, help="ANARCI worker processes")
    parser.add_argument("--force", action="store_true",
                        help="recompute scores even if their CSV already exists")
    parser.add_argument("--allow-missing-ablang2", action="store_true",
                        help="draw an explicit empty AbLang2 panel if it was not scored")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="figure directory (default: <artifacts>/analysis/figures)")
    return parser.parse_args()


def _run(command, python):
    command = [str(python), *[str(part) for part in command]]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage_cohort(pipe, tags):
    import select_cohort

    cohort_cfg = pipe.config["analysis"]["cohort"]
    instances = pipe.generation_instances()
    for tag in tags:
        designs = [pipe.designs_csv(tag, i) for i in instances]
        for path in designs:
            pl.require_file(path, f"run `python run_training.py --embedders {tag}` first")
        select_cohort.select_cohort(
            [str(p) for p in designs], str(pipe.records_csv), tag,
            cohort_cfg.get("n_per_target"),
            int(cohort_cfg.get("seed", pipe.config["experiment"]["seed"])),
            str(pipe.analysis_run_dir(tag) / "cohort.csv"),
        )


def stage_score(pipe, tags, scorers, pythons, ncpu, force):
    cfg = pipe.config["analysis"]
    failures = {}
    for scorer in scorers:
        python = pythons.get(scorer, sys.executable)
        if scorer == "ablang2":
            _run([
                ANALYSIS / "fetch_ablang2_weights.py",
                "--url", pipe.config["weights"]["ablang2_url"],
                "--weights-dir", pipe.ablang2_weights_dir,
                "--marker", pipe.ablang2_weights_dir / ".ready",
            ], python)
        for tag in tags:
            run_dir = pipe.analysis_run_dir(tag)
            cohort = pl.require_file(run_dir / "cohort.csv", "run the cohort stage first")
            out = run_dir / "metrics" / f"{scorer}.csv"
            if out.is_file() and not force:
                print(f"[{tag}] {scorer}: reusing {out}", flush=True)
                continue
            if scorer == "biophysical":
                command = [ANALYSIS / "score_biophysical.py", "--cohort", cohort,
                           "--charge-ph", cfg["developability"]["charge_ph"], "--out", out]
            elif scorer == "germline":
                command = [ANALYSIS / "score_germline.py", "--cohort", cohort,
                           "--scheme", cfg["germline"]["scheme"],
                           "--ncpu", ncpu, "--out", out]
                if cfg["germline"].get("allowed_species"):
                    command += ["--allowed-species", *cfg["germline"]["allowed_species"]]
            elif scorer == "iglm":
                command = [ANALYSIS / "score_iglm.py", "--cohort", cohort,
                           "--chain-token", cfg["ab_likeness"]["iglm"]["chain_token"],
                           "--species-token", cfg["ab_likeness"]["iglm"]["species_token"],
                           "--out", out]
            elif scorer == "antiberty":
                command = [ANALYSIS / "score_antiberty.py", "--cohort", cohort,
                           "--batch-size", cfg["ab_likeness"]["antiberty"]["batch_size"],
                           "--out", out]
            else:
                command = [ANALYSIS / "score_ablang2.py", "--cohort", cohort,
                           "--model-dir", pipe.ablang2_weights_dir, "--out", out]
            try:
                _run(command, python)
            except subprocess.CalledProcessError as exc:
                failures[(tag, scorer)] = str(exc)
                print(f"[{tag}] {scorer} FAILED: {exc}", flush=True)
    return failures


def stage_structure(pipe, tags, pythons, force):
    sp = pipe.config["structure_prediction"]
    rosetta_on = bool(pipe.config.get("rosetta_interface", {}).get("enabled", False))
    scorefxn = pipe.config.get("rosetta_interface", {}).get("scorefxn", "ref2015")
    unknown = sorted(set(sp["methods"]) - set(PREDICTORS))
    if unknown:
        raise SystemExit(f"unsupported structure method(s) {unknown}; AF3 is deferred")
    failures = {}
    for tag in tags:
        for instance in pipe.structure_targets():
            base = pipe.analysis_run_dir(tag) / "structures" / instance
            designs = pl.require_file(pipe.designs_csv(tag, instance), "generate designs first")
            for method in sp["methods"]:
                scores = base / f"{method}_confidence.csv"
                try:
                    if force or not scores.is_file():
                        _run([
                            ANALYSIS / PREDICTORS[method],
                            "--designs", designs, "--records", pipe.records_csv,
                            "--target-id", instance, "--n-designs", sp["n_designs"],
                            "--seed", sp.get("selection_seed", pipe.config["experiment"]["seed"]),
                            "--samples", sp.get("diffusion_samples", 1),
                            "--recycles", sp.get("recycling_steps", 3),
                            "--cache", pipe.artifact_root / "cache" / ("boltz" if method == "boltz2" else "chai"),
                            "--raw", base / f"{method}_raw", "--out", scores,
                            *(["--use-msa-server"] if sp.get("use_msa_server") else []),
                        ], pythons.get(method, sys.executable))
                    rosetta = base / f"{method}_rosetta.csv"
                    if rosetta_on and (force or not rosetta.is_file()):
                        _run([ANALYSIS / "run_rosetta_interface.py", "--scores", scores,
                              "--out", rosetta, "--scorefxn", scorefxn],
                             pythons.get("rosetta", sys.executable))
                except subprocess.CalledProcessError as exc:
                    failures[(tag, instance, method)] = str(exc)
                    print(f"[{tag}/{instance}/{method}] FAILED: {exc}", flush=True)
    return failures


def stage_plots(pipe, tags, args):
    import generate_figures

    labels = {tag: pipe.label(tag) for tag in tags}
    runs = {tag: pipe.run_dir(tag) for tag in tags}
    analysis = {tag: pipe.analysis_run_dir(tag) for tag in tags}
    only = set(args.only or generate_figures.FIGURES)

    def per_tag(root, *parts):
        return [pl.require_file(root[tag].joinpath(*parts),
                                "run the earlier pipeline stage first") for tag in tags]

    needs = {"fig3": ("iglm", "antiberty", "ablang2"), "fig4": ("germline",),
             "fig5": ("biophysical",), "fig6": ("germline",)}
    metrics = {}
    for metric in ("iglm", "antiberty", "ablang2", "germline", "biophysical"):
        if any(metric in needs.get(fig, ()) for fig in only):
            metrics[metric] = per_tag(analysis, "metrics", f"{metric}.csv")
    structures = [p for tag in tags
                  for p in sorted(analysis[tag].glob("structures/*/*_confidence.csv"))]
    rosetta = [p for tag in tags
               for p in sorted(analysis[tag].glob("structures/*/*_rosetta.csv"))]
    return generate_figures.generate_all(
        tags=tags, labels=labels,
        output_dir=args.output_dir or pipe.figures_dir,
        dpi=pipe.config["analysis"]["dpi"],
        evals=per_tag(runs, "eval.json") if "fig1" in only else [],
        metrics={k: metrics.get(k, []) for k in ("iglm", "antiberty", "ablang2")},
        germline=metrics.get("germline", []),
        biophysical=metrics.get("biophysical", []),
        confidence=structures, rosetta=rosetta,
        only=args.only, allow_missing_ablang2=args.allow_missing_ablang2,
    )


def main():
    args = parse_args()
    pipe = pl.Pipeline(pl.load_config(*args.config), args.artifact_root)
    flagged = bool(args.all_embedders or args.embedders or args.flag_embedders)
    configured = pipe.config["analysis"].get("embedders")
    if configured and not flagged:
        args.embedders = list(configured)
    tags = pipe.select_embedders(args)

    pythons = {}
    for item in args.python:
        key, _, path = item.partition("=")
        if key not in PYTHON_KEYS or not path:
            raise SystemExit(f"--python expects NAME=PATH with NAME in {PYTHON_KEYS}")
        pythons[key] = path

    stages = [s for s in STAGES if s in args.stages]
    structure_on = pipe.config["structure_prediction"].get("enabled", False)
    if args.skip_structure or not structure_on:
        stages = [s for s in stages if s != "structure"]
    print(f"embedders: {', '.join(tags)}\nstages: {', '.join(stages)}", flush=True)

    failures = {}
    started = time.time()
    try:
        if "cohort" in stages:
            print("\n=== cohort ===", flush=True)
            stage_cohort(pipe, tags)
        if "score" in stages:
            print("\n=== score ===", flush=True)
            failures.update(stage_score(pipe, tags, args.scorers, pythons,
                                        args.ncpu, args.force))
        if "structure" in stages:
            print("\n=== structure ===", flush=True)
            failures.update(stage_structure(pipe, tags, pythons, args.force))
        manifest = None
        if "plots" in stages:
            print("\n=== plots ===", flush=True)
            manifest = stage_plots(pipe, tags, args)
            failures.update({("plot", name): msg for name, msg in manifest["failed"].items()})
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit("analysis failed")

    print(f"\nanalysis finished in {time.time() - started:.0f}s", flush=True)
    if failures:
        for key, message in failures.items():
            print(f"FAILED {key}: {message}")
        raise SystemExit(f"{len(failures)} analysis step(s) failed")


if __name__ == "__main__":
    main()
