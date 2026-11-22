"""Render every MANGO comparison figure from already-scored inputs.

Used by ``run_analysis.py`` (its plotting stage). Each figure comes from its own
``plot_*.py`` module in this folder; this module only wires inputs to them. It
never runs sequence scorers or structure predictors, so every input must exist.

Figures
-------
fig1  NLL / perplexity by split and embedder
fig2  Boltz-2 / Chai-1 structure confidence       (needs structure tables)
fig3  antibody likeness (IgLM, AntiBERTy, AbLang2)
fig4  Levenshtein distance from germline - ONE violin plot, all embedders
fig5  developability: six biophysical properties as violins
fig6  heavy-chain V/J gene families and species
rosetta  fixed-backbone interface QC               (needs rosetta tables)
"""

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_ab_likeness  # noqa: E402
import plot_developability  # noqa: E402
import plot_genes  # noqa: E402
import plot_germline  # noqa: E402
import plot_nll  # noqa: E402
import plot_rosetta_interface  # noqa: E402
import plot_structure_confidence  # noqa: E402

FIGURES = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "rosetta"]


def _preview_ablang2_inputs(paths, output_dir):
    """Replace legacy confidence inputs with explicit empty preview inputs.

    AbLang2 confidence is never relabelled as pseudo-perplexity; legacy tables
    become an empty, clearly-marked panel instead.
    """
    preview_dir = Path(output_dir) / "_preview_inputs"
    preview_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for path in paths:
        frame = pd.read_csv(path)
        modes = set(frame.get("ablang2_mode", pd.Series(dtype=str)).astype(str))
        if modes == {"pseudo_log_likelihood"} and "ablang2_pseudo_log_likelihood" in frame:
            prepared.append(path)
            continue
        frame["ablang2_pseudo_log_likelihood"] = float("nan")
        frame["ablang2_mode"] = "pseudo_log_likelihood"
        destination = preview_dir / f"{Path(path).parent.parent.name}_ablang2.csv"
        frame.to_csv(destination, index=False)
        prepared.append(destination)
    return prepared


def _structure_embedders(paths):
    seen = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["embedder"])
        for tag in frame["embedder"].dropna().astype(str):
            if tag not in seen:
                seen.append(tag)
    return seen


def _structure_methods(paths):
    methods = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["predictor"])
        for method in frame["predictor"].dropna().astype(str):
            if method not in methods:
                methods.append(method)
    return methods


def generate_all(
    tags,
    labels,
    output_dir,
    dpi,
    evals,
    metrics,
    germline,
    biophysical,
    confidence=(),
    rosetta=(),
    only=None,
    allow_missing_ablang2=False,
):
    """Render the requested figures; return the manifest dict.

    ``evals``        eval.json paths, one per tag (same order as ``tags``)
    ``metrics``      {"iglm" | "antiberty" | "ablang2": [csv per tag]}
    ``germline``     germline.csv paths, one per tag (also feeds fig6)
    ``biophysical``  biophysical.csv paths, one per tag
    ``confidence``   *_confidence.csv paths (optional)
    ``rosetta``      *_rosetta.csv paths (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = list(tags)
    metrics = {name: [Path(p) for p in paths] for name, paths in metrics.items()}
    if allow_missing_ablang2 and metrics.get("ablang2"):
        metrics["ablang2"] = _preview_ablang2_inputs(metrics["ablang2"], output_dir)
    confidence = [Path(p) for p in confidence]
    rosetta = [Path(p) for p in rosetta]

    def fig1():
        plot_nll.plot_nll(
            evals, tags, labels, dpi,
            output_dir / "fig1_nll.png", output_dir / "fig1_nll_data.csv",
        )

    def fig2():
        if not confidence:
            raise FileNotFoundError("no structure-confidence tables were produced")
        plot_structure_confidence.plot(
            confidence, _structure_embedders(confidence),
            {t: labels.get(t, t) for t in _structure_embedders(confidence)},
            _structure_methods(confidence), dpi,
            output_dir / "fig2_structure_confidence.png",
            output_dir / "fig2_structure_confidence_data.csv",
        )

    def fig3():
        plot_ab_likeness.plot_ab_likeness(
            {name: metrics[name] for name in ("iglm", "antiberty", "ablang2")},
            tags, labels, dpi,
            output_dir / "fig3_ablikeness.png",
            output_dir / "fig3_ablikeness_data.csv",
        )

    def fig4():
        plot_germline.plot_germline(
            germline, tags, labels, dpi,
            output_dir / "fig4_ld_germline.png",
            output_dir / "fig4_ld_germline_data.csv",
        )

    def fig5():
        plot_developability.plot_developability(
            biophysical, tags, labels, dpi,
            output_dir / "fig5_developability.png",
            output_dir / "fig5_developability_data.csv",
        )

    def fig6():
        plot_genes.plot_all_embedder_genes(
            germline, tags, labels,
            output_dir / "fig6_gene_families.png",
            output_dir / "fig6_gene_families_data.csv",
            dpi, species_paths=germline,
        )

    def rosetta_figure():
        if not rosetta:
            raise FileNotFoundError("no Rosetta interface tables were produced")
        plot_rosetta_interface.plot(
            rosetta,
            output_dir / "rosetta_interface_plumbing.png",
            output_dir / "rosetta_interface_plumbing_data.csv",
            dpi,
        )

    steps = {
        "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
        "fig5": fig5, "fig6": fig6, "rosetta": rosetta_figure,
    }
    selected = list(only) if only else list(steps)
    unknown = sorted(set(selected) - set(steps))
    if unknown:
        raise ValueError(f"unknown figure(s) {unknown}; choose from {FIGURES}")

    failed, skipped = {}, {}
    for name in selected:
        # Structure-derived figures are optional when the user did not run them.
        if name == "fig2" and not confidence and not only:
            skipped[name] = "structure confidence was not run"
            continue
        if name == "rosetta" and not rosetta and not only:
            skipped[name] = "Rosetta interface scoring was not run"
            continue
        print(f"[{name}] rendering", flush=True)
        try:
            steps[name]()
        except Exception as exc:  # one bad input must not hide the other figures
            failed[name] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    manifest = {
        "embedders": tags,
        "figures": sorted(
            str(path.relative_to(output_dir)) for path in output_dir.rglob("*.png")
        ),
        "skipped": skipped,
        "failed": failed,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rendered {len(manifest['figures'])} plots in {output_dir}", flush=True)
    for name, reason in skipped.items():
        print(f"[{name}] skipped: {reason}", flush=True)
    return manifest
