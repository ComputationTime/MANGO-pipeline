# =============================================================================
# common.smk -- registry, derived paths, and helpers shared by every module.
# =============================================================================
# Nothing here runs jobs. It turns the global state file (config/config.yaml)
# into the paths and fan-out lists the rule modules consume, so a rule never
# has to know how a tag maps to a directory or which instances are in a split.

import hashlib
import json
import re
import sys
from pathlib import PurePosixPath

sys.path.insert(0, "workflow/scripts")
from generation_targets import select_generation_targets

# --- Dataset locations -------------------------------------------------------
_DS = config["dataset"]
DATASET_NAME = _DS["name"]
DATASET_VERSION = _DS["version"]
ARTIFACT_ROOT = _DS["artifact_root"]

DATASET_ROOT = f"{ARTIFACT_ROOT}/data/{DATASET_NAME}_v{DATASET_VERSION}"
SPLITS_DIR = f"{DATASET_ROOT}/splits_final"
SPLIT_CSV = f"{SPLITS_DIR}/{_DS['split_file']}"

# Module boundaries, in pipeline order.
FETCH_MARKER = f"{DATASET_ROOT}/.{DATASET_NAME}_ready"              # 1 fetch
STANDARDIZED_CSV = f"{DATASET_ROOT}/standardized/standardized.csv"  # 2 standardize
PROCESSED_DIR = f"{DATASET_ROOT}/processed"                         # 3 process
RECORDS_CSV = f"{PROCESSED_DIR}/records.csv"
EMB_DIR = f"{DATASET_ROOT}/embeddings"                              # 4 embed
RUNS_DIR = f"{ARTIFACT_ROOT}/runs"                                  # 5 train
DESIGNS_DIR = f"{ARTIFACT_ROOT}/designs"                            # 6 inference
ANALYSIS_DIR = f"{ARTIFACT_ROOT}/analysis"                          # 7 analysis
FIGURES_DIR = f"{ANALYSIS_DIR}/figures"
TARGETS_DIR = f"{ARTIFACT_ROOT}/targets"                            # Aim 2 structures
WEIGHTS_ROOT = config["weights"]["root"]
ABLANG2_WEIGHTS_DIR = f"{WEIGHTS_ROOT}/ABLANG-ablang2-paired"
ABLANG2_WEIGHTS_MARKER = f"{ABLANG2_WEIGHTS_DIR}/.ready"

LOG_DIR = f"{ARTIFACT_ROOT}/logs"


def structure_path(instance: str) -> str:
    """Path to the .cif for a dataset INSTANCE (e.g. 'pdb_00001a14_H_L')."""
    return str(PurePosixPath(SPLITS_DIR) / f"{instance}.cif")


# --- Embedder registry -------------------------------------------------------
EMBEDDERS = config["embedders"]
ACTIVE_EMBEDDERS = list(config["active_embedders"])
ANTIBODY_METHOD = config["embedding"]["antibody"]["method"]
BATCH_EMBEDDINGS = bool(config.get("execution", {}).get("batch_embeddings", False))

# The task is: antigen + light chain -> heavy chain. The antibody context
# embedding therefore covers the LIGHT chain only (heavy slot masked). The
# context is part of the directory name so a change of task can never silently
# reuse embeddings built under the old one.
ANTIBODY_CONTEXT = config["embedding"]["antibody"].get("context", "light_only")
if ANTIBODY_CONTEXT != "light_only":
    raise ValueError(
        f"embedding.antibody.context is {ANTIBODY_CONTEXT!r}; only 'light_only' "
        "is supported -- the study predicts the heavy chain from the antigen and "
        "the light chain, so the heavy chain must never enter the context."
    )
ANTIBODY_DIR = f"{ANTIBODY_METHOD}_{ANTIBODY_CONTEXT}"

SEQ_SOURCE = config["embedding"]["seq_source"]
EMBED_SPLITS = list(config["embedding"]["splits"])

_unknown = [t for t in ACTIVE_EMBEDDERS if t not in EMBEDDERS]
if _unknown:
    raise ValueError(
        f"active_embedders contains unregistered tag(s) {_unknown}; "
        f"known tags: {sorted(EMBEDDERS)}"
    )


def embedder_spec(tag: str) -> dict:
    """Full config block for an embedder tag."""
    return EMBEDDERS[tag]


def embedder_method(tag: str) -> str:
    """Which embedding method (and therefore which rule/script) a tag uses."""
    return EMBEDDERS[tag]["method"]


def embedder_label(tag: str) -> str:
    """Human-readable name for figures."""
    return EMBEDDERS[tag].get("label", tag)


def tags_for_method(method: str) -> list:
    """Active tags built by a given method -- used for wildcard constraints.

    Several tags can share a method (e.g. two ESM2 sizes), and every method
    rule writes the same output pattern, so each rule constrains {embedder} to
    exactly its own tags. That keeps the DAG unambiguous without ruleorder.
    """
    return [t for t in ACTIVE_EMBEDDERS if embedder_method(t) == method]


def method_constraint(method: str) -> str:
    """Regex matching only the active tags of `method` (matches nothing if none)."""
    tags = tags_for_method(method)
    if not tags:
        return r"(?!x)x"  # inactive method: unsatisfiable, so the rule is inert
    return "(" + "|".join(re.escape(t) for t in tags) + ")"


# --- Embedding paths (train/val/test subfolders, one .pt per id) -------------
def antigen_emb_dir(tag: str) -> str:
    return f"{EMB_DIR}/antigen/{tag}"


def antigen_emb_config(tag: str) -> str:
    return f"{antigen_emb_dir(tag)}/embedder_config.json"


def antigen_emb_path(tag: str, split: str, instance: str) -> str:
    return f"{antigen_emb_dir(tag)}/{split}/{instance}.pt"


def antibody_emb_dir() -> str:
    return f"{EMB_DIR}/antibody/{ANTIBODY_DIR}"


def antibody_emb_config() -> str:
    return f"{antibody_emb_dir()}/embedder_config.json"


def antibody_emb_path(split: str, instance: str) -> str:
    return f"{antibody_emb_dir()}/{split}/{instance}.pt"


def antigen_batch_marker(tag: str) -> str:
    return f"{antigen_emb_dir(tag)}/.batch_complete.json"


def antibody_batch_marker() -> str:
    return f"{antibody_emb_dir()}/.batch_complete.json"


def pretrained_weight_marker(tag: str) -> str:
    """Proof that an upstream package successfully cached a tag's model."""
    return f"{WEIGHTS_ROOT}/pretrained/{tag}.json"


# --- Instance lists (resolved from the process checkpoint) -------------------
def _records_df():
    """Read records.csv via the process checkpoint (forces it to run first)."""
    import pandas as pd

    records = checkpoints.process_records.get().output.records
    return pd.read_csv(records, dtype=str, keep_default_na=False)


def instances_by_split(splits) -> list:
    """[(split, instance), ...] for the requested splits, in records order."""
    df = _records_df()
    wanted = set(splits)
    sub = df.loc[df["split"].isin(wanted), ["split", "id"]]
    return list(sub.itertuples(index=False, name=None))


def antigen_embedding_targets(tag: str, splits=None) -> list:
    splits = EMBED_SPLITS if splits is None else splits
    return [antigen_emb_path(tag, s, i) for s, i in instances_by_split(splits)]


def antibody_embedding_targets(splits=None) -> list:
    splits = EMBED_SPLITS if splits is None else splits
    return [antibody_emb_path(s, i) for s, i in instances_by_split(splits)]


def antigen_embedding_dependencies(tag: str, splits=None) -> list:
    """Declared readiness inputs; GPU mode uses a persistent-model batch marker."""
    if BATCH_EMBEDDINGS:
        return [antigen_batch_marker(tag)]
    return antigen_embedding_targets(tag, splits)


def antibody_embedding_dependencies(splits=None) -> list:
    if BATCH_EMBEDDINGS:
        return [antibody_batch_marker()]
    return antibody_embedding_targets(splits)


# --- Runs (one trained model per embedder tag) -------------------------------
def experiment_hash(tag: str) -> str:
    """Stable short hash of everything that defines this model's training.

    Covers the dataset, the processing filters, the embedding choices, the
    embedder's own parameters, and the model hyperparameters -- so changing any
    of them yields a different run id, and re-running with the same global state
    file reuses the existing one.
    """
    model_config = dict(config["model"])
    # Whether Snakemake should rerun a model is execution policy, not part of
    # the scientific model identity.
    model_config.pop("retrain", None)
    model_config.pop("loss_plot_interval", None)
    payload = {
        "dataset": config["dataset"],
        "processing": config["processing"],
        "embedding": config["embedding"],
        "embedder": EMBEDDERS[tag],
        "model": model_config,
        "weights": config["weights"],
        "experiment_seed": config["experiment"]["seed"],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def run_id(tag: str) -> str:
    """Run directory name: readable tag + config hash (signature: runs/<hash>/)."""
    return f"{tag}__{experiment_hash(tag)}"


def run_dir(tag: str) -> str:
    return f"{RUNS_DIR}/{run_id(tag)}"


def run_ckpt(tag: str) -> str:
    return f"{run_dir(tag)}/checkpoints/best.pt"


def run_model_config(tag: str) -> str:
    return f"{run_dir(tag)}/model_config.json"


def run_metrics(tag: str) -> str:
    return f"{run_dir(tag)}/metrics.jsonl"


def run_training_curve_csv(tag: str) -> str:
    return f"{run_dir(tag)}/training_curve.csv"


def run_training_curve_png(tag: str) -> str:
    return f"{run_dir(tag)}/training_curve.png"


# Map run_id -> tag so rules carrying a {run} wildcard can recover their embedder.
RUN_ID_TO_TAG = {run_id(t): t for t in ACTIVE_EMBEDDERS}


def tag_for_run(run: str) -> str:
    return RUN_ID_TO_TAG[run]


# --- Inference / analysis outputs -------------------------------------------
def predictions_csv(tag: str) -> str:
    """Test-split reconstruction predictions (one row per test structure)."""
    return f"{run_dir(tag)}/predictions_test.csv"


def eval_json(tag: str) -> str:
    """Split NLL/perplexity for this run -- the data behind figure 1."""
    return f"{run_dir(tag)}/eval.json"


def designs_csv(tag: str, instance: str) -> str:
    """De novo heavy-chain designs against one held-out structure (Aim 2)."""
    return f"{DESIGNS_DIR}/{run_id(tag)}/{instance}/designs.csv"


def seq_metrics_csv(tag: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/seq_metrics.csv"


def analysis_cohort_csv(tag: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/cohort.csv"


def analysis_metric_csv(tag: str, metric: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/metrics/{metric}.csv"


def analysis_design_inputs(run: str) -> list:
    """Generated-design tables feeding one run's deterministic cohort."""
    tag = tag_for_run(run)
    return [designs_csv(tag, instance) for instance in generation_instances()]


def figure_data_path(name: str) -> str:
    return f"{FIGURES_DIR}/{name}_data.csv"


def struct_scores_csv(tag: str, instance: str, method: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/structures/{instance}/{method}_confidence.csv"


def struct_raw_dir(tag: str, instance: str, method: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/structures/{instance}/{method}_raw"


def complex_metrics_csv(tag: str, instance: str) -> str:
    return f"{ANALYSIS_DIR}/{run_id(tag)}/structures/{instance}/complex_metrics.csv"


def figure_path(name: str) -> str:
    return f"{FIGURES_DIR}/{name}.{config['analysis']['format']}"


# --- Analysis fan-in ---------------------------------------------------------
def analysis_embedders() -> list:
    """Embedder tags compared in the figures (null => active_embedders)."""
    configured = config["analysis"].get("embedders")
    return list(configured) if configured else list(ACTIVE_EMBEDDERS)


ANALYSIS_EMBEDDERS = analysis_embedders()

_bad = [t for t in ANALYSIS_EMBEDDERS if t not in EMBEDDERS]
if _bad:
    raise ValueError(f"analysis.embedders contains unregistered tag(s) {_bad}")


def enabled_figures() -> list:
    return [n for n, c in config["analysis"]["plots"].items() if c.get("enabled", True)]


# --- Aim 2 target structures -------------------------------------------------
GEN_TARGETS = config["generation"].get("targets", [])
TARGET_PDBS = [t["pdb"] for t in GEN_TARGETS]

# The therapeutic targets are described by a records-shaped table with
# split="target", so the SAME eight embedder rules serve them -- their
# embeddings land in <tag>/target/<pdb>.pt alongside train/val/test.
TARGETS_RECORDS = f"{TARGETS_DIR}/targets_records.csv"
TARGET_SPLIT = "target"


def target_structure(pdb: str) -> str:
    return f"{TARGETS_DIR}/structures/{pdb}.cif"


def target_emb_path(tag: str, pdb: str) -> str:
    return antigen_emb_path(tag, TARGET_SPLIT, pdb)


def records_for_split(split: str) -> str:
    """Which records table backs a given split (targets live in their own)."""
    return TARGETS_RECORDS if split == TARGET_SPLIT else RECORDS_CSV


# --- Aim 2 generation source -------------------------------------------------
# Which structures generation designs against. `test` uses held-out dataset
# complexes; `target` uses the therapeutic panel above. Either way the source
# must supply an antigen AND a light chain, because that pair is the entire
# conditioning signal for heavy-chain generation.
GENERATION_SPLIT = config["generation"]["source"]

if GENERATION_SPLIT not in set(EMBED_SPLITS):
    raise ValueError(
        f"generation.source is {GENERATION_SPLIT!r}, which is not one of the "
        f"active embedded splits {EMBED_SPLITS}. External therapeutic targets "
        "are deferred from the first milestone."
    )


def generation_instances() -> list:
    """Leakage-safe, deterministic ids used by generation and Figures 3-5."""
    if GENERATION_SPLIT == TARGET_SPLIT:
        ids = sorted(TARGET_PDBS)
        limit = config["generation"].get("max_targets")
        return ids[: int(limit)] if limit else ids

    selection = config["generation"].get("target_selection", {})
    return select_generation_targets(
        _records_df(), split=GENERATION_SPLIT,
        strategy=selection.get("strategy", "one_per_cluster"),
        cluster_column=selection.get(
            "cluster_column", config["processing"]["val"]["cluster_column"]),
        max_targets=config["generation"].get("max_targets"),
    )


def generation_antigen_emb(tag: str, instance: str) -> str:
    return antigen_emb_path(tag, GENERATION_SPLIT, instance)


def generation_antibody_emb(instance: str) -> str:
    return antibody_emb_path(GENERATION_SPLIT, instance)


def generation_structure(instance: str) -> str:
    """The .cif backing a generation instance, in whichever tree it lives."""
    if GENERATION_SPLIT == TARGET_SPLIT:
        return target_structure(instance)
    return structure_path(instance)


def sp_targets() -> list:
    """Structures whose designs go through structure prediction (Aim 3).

    Defaults to the head of the generation set so Aim 3 always folds designs
    that Aim 2 actually produced. Resolves the process checkpoint, so only call
    it from an input function, never at parse time.
    """
    configured = config["structure_prediction"].get("targets")
    if configured:
        return list(configured)
    n = int(config["structure_prediction"].get("n_targets", 1))
    return generation_instances()[:n]


def sp_methods() -> list:
    methods = list(config["structure_prediction"]["methods"])
    supported = {"boltz2", "chai"}
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise ValueError(
            f"local structure-confidence method(s) {unknown} are not supported; "
            "AF3 remains deferred to cluster-side ingestion"
        )
    return methods


# --- Wildcard hygiene --------------------------------------------------------
# `instance` must not swallow path separators; `split`, `embedder` and `run` are
# drawn from closed sets so they can never eat each other's path segment.
wildcard_constraints:
    instance=r"[^/]+",
    split=r"train|val|test|target",
    embedder="|".join(re.escape(t) for t in EMBEDDERS),
    run="|".join(re.escape(r) for r in RUN_ID_TO_TAG) or r"(?!x)x",
    pdb=r"[A-Za-z0-9]{4}",
    sp_method=r"[a-z0-9_]+",
    model=r"[a-z_]+models",
    noise3=r"\d{3}",
