"""Shared configuration, paths and run identities for the master scripts.

This is the plain-Python equivalent of the old Snakemake ``common.smk``: it
turns ``config/config.yaml`` (the single global state file) into the directory
layout every stage reads and writes, so ``build_dataset.py``,
``run_embedding.py``, ``run_training.py`` and ``run_analysis.py`` always agree.

    artifacts/data/<dataset>_v<version>/splits_final/        raw dataset
    artifacts/data/<dataset>_v<version>/standardized/        standardized.csv
    artifacts/data/<dataset>_v<version>/processed/records.csv  train/val/test
    artifacts/data/<dataset>_v<version>/embeddings/antigen/<tag>/<split>/<id>.pt
    artifacts/runs/<tag>__<hash>/                            trained models
    artifacts/designs/<tag>__<hash>/<target>/designs.csv     generated designs
    artifacts/analysis/<tag>__<hash>/                        cohort + metrics
    artifacts/analysis/figures/                              every plot
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

DEFAULT_LABELS = {
    "one_hot": "One-hot",
    "biopython": "BioPython",
    "pyrosetta_pre": "PyRosetta PRE",
    "esm2": "ESM2",
    "esm3": "ESM3 (sequence)",
    "esm3_structure": "ESM3 (sequence + structure)",
    "esmif": "ESM-IF",
    "proteinmpnn": "ProteinMPNN",
}

# Methods the batch embedder can build today.
EMBEDDABLE_METHODS = {
    "one_hot", "biopython", "pyrosetta_pre", "esm2", "esm3", "esmif",
    "proteinmpnn", "antigen_control",
}


def _deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(*paths) -> dict:
    """Load config.yaml, then layer any overlay files on top of it."""
    import yaml

    paths = [Path(p) for p in paths] or [DEFAULT_CONFIG]
    config: dict = {}
    for path in paths:
        with open(path) as handle:
            _deep_update(config, yaml.safe_load(handle) or {})
    return config


def add_common_args(parser):
    """Flags shared by every master script."""
    parser.add_argument(
        "--config", nargs="+", type=Path, default=[DEFAULT_CONFIG],
        help="config file(s); later files override earlier ones",
    )
    parser.add_argument(
        "--artifact-root", default=None,
        help="override dataset.artifact_root from the config",
    )


def add_embedder_args(parser):
    group = parser.add_argument_group("embedder selection")
    group.add_argument(
        "--embedders", nargs="+", metavar="TAG",
        help="embedder tag(s) to use, e.g. --embedders one_hot esm2",
    )
    group.add_argument(
        "--all", action="store_true", dest="all_embedders",
        help="use every implemented embedder registered in the config",
    )
    for tag in DEFAULT_LABELS:
        group.add_argument(
            f"--{tag.replace('_', '-')}", action="append_const",
            dest="flag_embedders", const=tag,
            help=f"include the {tag} embedder",
        )


class Pipeline:
    """Resolved paths and identities for one configuration."""

    def __init__(self, config: dict, artifact_root: str | None = None):
        self.config = copy.deepcopy(config)
        if artifact_root:
            self.config["dataset"]["artifact_root"] = str(artifact_root)
            self.config["weights"]["root"] = f"{artifact_root}/weights"
        ds = self.config["dataset"]
        self.dataset_name = ds["name"]
        self.dataset_version = ds["version"]
        self.artifact_root = Path(ds["artifact_root"])
        if not self.artifact_root.is_absolute():
            self.artifact_root = REPO_ROOT / self.artifact_root
        root = self.artifact_root
        self.dataset_root = root / "data" / f"{self.dataset_name}_v{self.dataset_version}"
        self.splits_dir = self.dataset_root / "splits_final"
        self.split_csv = self.splits_dir / ds["split_file"]
        self.fetch_marker = self.dataset_root / f".{self.dataset_name}_ready"
        self.standardized_csv = self.dataset_root / "standardized" / "standardized.csv"
        self.records_csv = self.dataset_root / "processed" / "records.csv"
        self.emb_dir = self.dataset_root / "embeddings"
        self.runs_dir = root / "runs"
        self.designs_dir = root / "designs"
        self.analysis_dir = root / "analysis"
        self.figures_dir = self.analysis_dir / "figures"
        self.log_dir = root / "logs"
        weights_root = Path(self.config["weights"]["root"])
        self.weights_root = (
            weights_root if weights_root.is_absolute() else REPO_ROOT / weights_root
        )
        self.ablang2_weights_dir = self.weights_root / "ABLANG-ablang2-paired"
        self.embedders = self.config["embedders"]
        self.seq_source = self.config["embedding"]["seq_source"]
        self.embed_splits = list(self.config["embedding"]["splits"])

    # -- embedder selection ---------------------------------------------------
    def select_embedders(self, args=None, default_to_active=True) -> list:
        """Resolve --embedders / --all / --<tag> flags to a validated tag list."""
        requested: list = []
        if args is not None:
            if getattr(args, "all_embedders", False):
                requested = [
                    t for t, spec in self.embedders.items()
                    if spec["method"] in EMBEDDABLE_METHODS
                ]
            else:
                requested = list(getattr(args, "embedders", None) or [])
                requested += list(getattr(args, "flag_embedders", None) or [])
        if not requested and default_to_active:
            requested = list(self.config.get("active_embedders", []))
        requested = list(dict.fromkeys(requested))
        unknown = [t for t in requested if t not in self.embedders]
        if unknown:
            raise SystemExit(
                f"unknown embedder(s) {unknown}; registered: {sorted(self.embedders)}"
            )
        if not requested:
            raise SystemExit("no embedders selected: use --embedders, --all, or --<tag>")
        return requested

    def spec(self, tag: str) -> dict:
        return self.embedders[tag]

    def method(self, tag: str) -> str:
        return self.embedders[tag]["method"]

    def label(self, tag: str) -> str:
        return self.embedders[tag].get("label", DEFAULT_LABELS.get(tag, tag))

    def embedding_tag(self, tag: str) -> str:
        source = self.embedders[tag].get("embedding_source", tag)
        if source not in self.embedders:
            raise ValueError(f"invalid embedding_source {source!r} for {tag!r}")
        return source

    # -- paths ---------------------------------------------------------------
    def structure_path(self, instance: str) -> Path:
        return self.splits_dir / f"{instance}.cif"

    def antigen_emb_dir(self, tag: str) -> Path:
        return self.emb_dir / "antigen" / self.embedding_tag(tag)

    def pretrained_marker(self, tag: str) -> Path:
        return self.weights_root / "pretrained" / f"{tag}.json"

    def proteinmpnn_weights(self, tag: str) -> Path:
        spec = self.spec(tag)
        return (self.weights_root / "proteinmpnn" / spec["model"]
                / f"v_48_{int(spec['noise']):03d}.pt")

    # -- run identity (identical to the former Snakemake hash) ----------------
    def model_config(self, tag: str) -> dict:
        return {**self.config["model"], **self.embedders[tag].get("model_overrides", {})}

    def experiment_hash(self, tag: str) -> str:
        model_config = self.model_config(tag)
        model_config.pop("retrain", None)
        model_config.pop("loss_plot_interval", None)
        payload = {
            "dataset": self.config["dataset"],
            "processing": self.config["processing"],
            "embedding": self.config["embedding"],
            "embedder": self.embedders[tag],
            "model": model_config,
            "weights": self.config["weights"],
            "experiment_seed": self.config["experiment"]["seed"],
        }
        if self.embedding_tag(tag) != tag:
            payload["embedding_source"] = self.embedders[self.embedding_tag(tag)]
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:8]

    def run_id(self, tag: str) -> str:
        return f"{tag}__{self.experiment_hash(tag)}"

    def run_dir(self, tag: str) -> Path:
        return self.runs_dir / self.run_id(tag)

    def designs_csv(self, tag: str, instance: str) -> Path:
        return self.designs_dir / self.run_id(tag) / instance / "designs.csv"

    def analysis_run_dir(self, tag: str) -> Path:
        return self.analysis_dir / self.run_id(tag)

    # -- generation targets ---------------------------------------------------
    def generation_instances(self) -> list:
        import pandas as pd
        from generation_targets import select_generation_targets

        gen = self.config["generation"]
        records = pd.read_csv(self.records_csv, dtype=str, keep_default_na=False)
        selection = gen.get("target_selection", {})
        return select_generation_targets(
            records,
            split=gen["source"],
            strategy=selection.get("strategy", "one_per_cluster"),
            cluster_column=selection.get(
                "cluster_column", self.config["processing"]["val"]["cluster_column"]
            ),
            max_targets=gen.get("max_targets"),
        )

    def structure_targets(self) -> list:
        sp = self.config["structure_prediction"]
        if sp.get("targets"):
            return list(sp["targets"])
        available = self.generation_instances()
        n = int(sp.get("n_targets", 1))
        if len(available) < n:
            raise ValueError(
                f"structure_prediction requests {n} targets but generation "
                f"supplied only {len(available)}"
            )
        return available[:n]


def require_file(path, hint: str):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing {path}\n  -> {hint}")
    return path
