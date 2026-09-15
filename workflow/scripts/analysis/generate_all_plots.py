"""Render every MANGO analysis plot from existing metric artifacts.

This is a lightweight preview entry point: it calls the same plotting functions
as Snakemake but never runs sequence scorers or structure-prediction models.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import plot_ab_likeness
import plot_developability
import plot_genes
import plot_germline
import plot_nll
import plot_rosetta_interface
import plot_structure_confidence


DEFAULT_LABELS = {
    "one_hot": "One-hot",
    "biopython": "BioPython",
    "pyrosetta_pre": "PyRosetta PRE",
    "esm2": "ESM2",
    "esm3": "ESM3 (sequence)",
    "esmif": "ESM-IF",
    "proteinmpnn": "ProteinMPNN",
    "afm": "AF-M",
}


def _existing(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required plot input does not exist: {path}")
    return path


def _discover_embedders(input_root):
    embedders_dir = input_root / "embedders"
    if not embedders_dir.is_dir():
        raise FileNotFoundError(
            f"expected an embedders directory beneath {input_root}"
        )
    tags = sorted(
        path.name
        for path in embedders_dir.iterdir()
        if path.is_dir() and (path / "eval.json").is_file()
    )
    if not tags:
        raise RuntimeError(f"no embedder artifacts found beneath {embedders_dir}")
    return tags


def _ordered_embedders(input_root, requested):
    available = _discover_embedders(input_root)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested embedders have no artifacts: {missing}")
        return requested
    preferred = [tag for tag in DEFAULT_LABELS if tag in available]
    return preferred + [tag for tag in available if tag not in preferred]


def _structure_inputs(structure_root, suffix):
    paths = sorted(structure_root.glob(f"*_{suffix}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no *_{suffix}.csv inputs found beneath {structure_root}"
        )
    return paths


def _structure_embedders(paths):
    seen = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["embedder"])
        for tag in frame["embedder"].dropna().astype(str):
            if tag not in seen:
                seen.append(tag)
    if not seen:
        raise RuntimeError("structure inputs contain no embedder values")
    return seen


def _preview_ablang2_inputs(paths, output_dir):
    """Replace legacy confidence inputs with explicit empty preview inputs."""
    preview_dir = output_dir / "_preview_inputs"
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


def generate_all(
    input_root,
    structure_root,
    predictions_csv,
    output_dir,
    embedders,
    dpi,
    allow_missing_ablang2=False,
):
    input_root = Path(input_root)
    structure_root = Path(structure_root)
    predictions_csv = _existing(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_tags = pd.read_csv(
        predictions_csv, usecols=["embedder"]
    )["embedder"].dropna().astype(str).unique().tolist()
    artifact_tags = _ordered_embedders(input_root, embedders)
    tags = [tag for tag in artifact_tags if tag in prediction_tags]
    if embedders and len(tags) != len(embedders):
        missing = sorted(set(embedders) - set(tags))
        raise ValueError(f"requested embedders are absent from predictions CSV: {missing}")
    if not tags:
        raise RuntimeError("no embedders are shared by artifacts and predictions CSV")
    labels = {tag: DEFAULT_LABELS.get(tag, tag) for tag in tags}
    embedder_root = input_root / "embedders"

    evals = [_existing(embedder_root / tag / "eval.json") for tag in tags]
    metrics = {
        metric: [
            _existing(embedder_root / tag / "metrics" / f"{metric}.csv")
            for tag in tags
        ]
        for metric in ("iglm", "antiberty", "ablang2", "germline")
    }
    if allow_missing_ablang2:
        metrics["ablang2"] = _preview_ablang2_inputs(
            metrics["ablang2"], output_dir
        )

    plot_nll.plot_nll(
        evals,
        tags,
        labels,
        dpi,
        output_dir / "fig1_nll.png",
        output_dir / "fig1_nll_data.csv",
    )

    confidence_paths = _structure_inputs(structure_root, "confidence")
    structure_tags = _structure_embedders(confidence_paths)
    structure_labels = {
        tag: DEFAULT_LABELS.get(tag, tag) for tag in structure_tags
    }
    methods = []
    for path in confidence_paths:
        frame = pd.read_csv(path, usecols=["predictor"])
        for method in frame["predictor"].dropna().astype(str):
            if method not in methods:
                methods.append(method)
    plot_structure_confidence.plot(
        confidence_paths,
        structure_tags,
        structure_labels,
        methods,
        dpi,
        output_dir / "fig2_structure_confidence.png",
        output_dir / "fig2_structure_confidence_data.csv",
    )

    plot_ab_likeness.plot_ab_likeness(
        {name: metrics[name] for name in ("iglm", "antiberty", "ablang2")},
        tags,
        labels,
        dpi,
        output_dir / "fig3_ablikeness.png",
        output_dir / "fig3_ablikeness_data.csv",
    )
    plot_germline.plot_germline(
        [predictions_csv],
        tags,
        labels,
        dpi,
        output_dir / "fig4_ld_germline.png",
        output_dir / "fig4_ld_germline_data.csv",
    )
    plot_developability.plot_developability(
        [predictions_csv],
        tags,
        labels,
        dpi,
        output_dir / "fig5_developability.png",
        output_dir / "fig5_developability_data.csv",
    )
    plot_genes.plot_all_embedder_genes(
        predictions_csv,
        tags,
        labels,
        output_dir / "fig6_gene_families.png",
        output_dir / "fig6_gene_families_data.csv",
        dpi,
        species_paths=metrics["germline"],
    )

    rosetta_paths = _structure_inputs(structure_root, "rosetta")
    plot_rosetta_interface.plot(
        rosetta_paths,
        output_dir / "rosetta_interface_plumbing.png",
        output_dir / "rosetta_interface_plumbing_data.csv",
        dpi,
    )

    manifest = {
        "input_root": str(input_root),
        "structure_root": str(structure_root),
        "predictions_csv": str(predictions_csv),
        "embedders": tags,
        "structure_embedders": structure_tags,
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"rendered {len(manifest['figures'])} plots in {output_dir}", flush=True)


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "review" / "smoke_artifacts" / "github",
        help="Root containing embedders/<tag>/eval.json and metrics/*.csv",
    )
    parser.add_argument(
        "--structure-root",
        type=Path,
        default=(
            repo_root
            / "review"
            / "structure_confidence_sample"
            / "github"
            / "predictions"
        ),
        help="Directory containing existing *_confidence.csv and *_rosetta.csv",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help=(
            "Flat predictions table containing embedder, gene, germline-distance, "
            "and developability columns"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "plot_preview",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        help="Optional ordered subset; defaults to every available embedder",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--allow-missing-ablang2",
        action="store_true",
        help=(
            "Render an explicit empty AbLang2 panel when only legacy confidence "
            "inputs are available; never converts confidence to perplexity"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_all(
        args.input_root,
        args.structure_root,
        args.predictions_csv,
        args.output_dir,
        args.embedders,
        args.dpi,
        args.allow_missing_ablang2,
    )


if __name__ == "__main__":
    main()
