#!/usr/bin/env bash
set -euo pipefail

# One-command local NVIDIA GPU runner.
#   ./run_gpu.sh smoke        five-record plumbing run (default)
#   ./run_gpu.sh small        bounded multi-record validation run
#   ./run_gpu.sh study        full seven-embedder SAbDab2 study
#   ./run_gpu.sh smoke-esm3   smoke + gated ESM3
#   ./run_gpu.sh study-esm3   full study + gated ESM3
#   ./run_gpu.sh smoke-pyrosetta / study-pyrosetta
#   ./run_gpu.sh smoke-all / study-all (ESM3 + PyRosetta)
#   ./run_gpu.sh weights[-esm3|-pyrosetta|-all] (downloads only)
#   ./run_gpu.sh esm2-ablation (8M/35M/150M plus the cached 650M evaluation)
#   ./run_gpu.sh antigen-controls (constant, random, pooled, and shuffled controls)

MODE="${1:-smoke}"
case "$MODE" in
  smoke|small|study|smoke-esm3|small-esm3|study-esm3|smoke-pyrosetta|small-pyrosetta|study-pyrosetta|smoke-all|small-all|study-all|weights|weights-esm3|weights-pyrosetta|weights-all|esm2-ablation|antigen-controls) ;;
  *) echo "usage: $0 {smoke|small|study|weights}[-esm3|-pyrosetta|-all] | esm2-ablation | antigen-controls" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is missing; install an NVIDIA driver before running MANGO." >&2
  exit 1
}
nvidia-smi >/dev/null

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" && -x "/workspace/miniforge3/bin/conda" ]]; then
  CONDA_BIN="/workspace/miniforge3/bin/conda"
fi
if [[ -z "$CONDA_BIN" ]]; then
  echo "conda is required for isolated embedder environments." >&2
  exit 1
fi
# Snakemake invokes `conda` by name when creating per-rule environments, even
# when the launcher discovered it through CONDA_EXE rather than the inherited
# PATH (the normal layout for a non-interactive Miniforge installation).
export PATH="$(dirname "$CONDA_BIN"):$PATH"

export MANGO_REQUIRE_CUDA=1
export PYTHONNOUSERSITE=1
export CONDA_CHANNEL_PRIORITY=strict
export HF_HOME="${HF_HOME:-$ROOT_DIR/artifacts/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/artifacts/cache/torch}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT_DIR/.snakemake/conda-pkgs}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CONDA_PKGS_DIRS"

if [[ "$MODE" == "study" || "$MODE" == *-esm3 || "$MODE" == *-all ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: HF_TOKEN is unset; ESM3 may fail, but other embedders will still run." >&2
  fi
fi

DRIVER="$ROOT_DIR/.snakemake/gpu-driver"
if [[ ! -x "$DRIVER/bin/snakemake" ]]; then
  "$CONDA_BIN" create -y -p "$DRIVER" -c conda-forge \
    --strict-channel-priority python=3.11 pip
  "$DRIVER/bin/pip" install --no-cache-dir \
    snakemake==9.25.1 pandas==2.3.3
fi

CONFIG_ARGS=(--configfile config/gpu.yaml)
if [[ "$MODE" == smoke* ]]; then
  CONFIG_ARGS+=(config/smoke.yaml)
elif [[ "$MODE" == small* ]]; then
  CONFIG_ARGS+=(config/small.yaml)
fi
if [[ "$MODE" == "study" ]]; then
  CONFIG_ARGS+=(config/gpu_all.yaml)
elif [[ "$MODE" == "esm2-ablation" ]]; then
  CONFIG_ARGS+=(config/esm2_ablation.yaml)
elif [[ "$MODE" == "antigen-controls" ]]; then
  CONFIG_ARGS+=(config/antigen_controls.yaml)
elif [[ "$MODE" == *-esm3 ]]; then
  CONFIG_ARGS+=(config/gpu_esm3.yaml)
elif [[ "$MODE" == *-pyrosetta ]]; then
  CONFIG_ARGS+=(config/gpu_pyrosetta.yaml)
elif [[ "$MODE" == *-all ]]; then
  CONFIG_ARGS+=(config/gpu_all.yaml)
fi

SNAKEMAKE=("$DRIVER/bin/snakemake" -s workflow/Snakefile \
  --profile workflow/profiles/gpu "${CONFIG_ARGS[@]}" --rerun-incomplete)

EMBEDDERS=(one_hot biopython esm2 esmif proteinmpnn)
if [[ "$MODE" == "study" ]]; then
  EMBEDDERS=(one_hot biopython pyrosetta_pre esm2 esm3 esmif proteinmpnn)
elif [[ "$MODE" == *-pyrosetta || "$MODE" == *-all ]]; then
  EMBEDDERS=(one_hot biopython pyrosetta_pre esm2 esmif proteinmpnn)
fi
if [[ "$MODE" == *-esm3 ]]; then
  EMBEDDERS=(one_hot biopython esm2 esm3 esmif proteinmpnn)
elif [[ "$MODE" == *-all ]]; then
  EMBEDDERS=(one_hot biopython pyrosetta_pre esm2 esm3 esmif proteinmpnn)
fi

run_for_embedder() {
  local embedder="$1"
  local target="$2"
  shift 2
  # A separate Snakemake invocation is intentional. In a combined DAG,
  # environment creation happens before execution and one invalid environment
  # can prevent otherwise independent embedder jobs from ever starting.
  "${SNAKEMAKE[@]}" \
    --config "active_embedders=[$embedder]" \
    --keep-going "$@" "$target"
}

refresh_review_bundle() {
  # Review bundles are lightweight symlink indexes. Refresh after each method
  # so partial progress is immediately easy to inspect.
  if [[ "$MODE" == smoke* && -f artifacts/gpu/results_report.json ]]; then
    "$DRIVER/bin/python" workflow/scripts/collect_review_artifacts.py \
      --tier smoke --output review/smoke_artifacts
  elif [[ "$MODE" == study* && -f artifacts/gpu/results_report.json ]]; then
    "$DRIVER/bin/python" workflow/scripts/collect_review_artifacts.py \
      --tier full --output review/full_v2_artifacts
  fi
}

requires_prefetch() {
  case "$1" in
    pyrosetta_pre|esm2|esm3|esmif|proteinmpnn) return 0 ;;
    *) return 1 ;;
  esac
}

# Force the preflight on every invocation so a manifest copied from another
# machine can never bypass validation of this GPU and driver.
"${SNAKEMAKE[@]}" --force gpu_preflight

if [[ "$MODE" == "esm2-ablation" ]]; then
  # The requested control point is an immutable input to this focused run.
  # Refuse to proceed if it is absent instead of silently retraining 650M.
  REUSED_ESM2_EVAL="artifacts/runs/esm2__c49369e9/eval.json"
  if [[ ! -f "$REUSED_ESM2_EVAL" ]]; then
    echo "The cached 650M evaluation is missing: $REUSED_ESM2_EVAL" >&2
    echo "Refusing to retrain 650M for the ESM2-size ablation." >&2
    exit 1
  fi

  RUN_FAILED=0
  for embedder in esm2_8m esm2_35m esm2_150m; do
    echo
    echo "===== MANGO ESM2 ablation: $embedder ====="
    WEIGHT_FAILED=0
    if ! run_for_embedder "$embedder" weights; then
      WEIGHT_FAILED=1
      echo "[$embedder] weight setup failed; attempting cached assets." >&2
    fi
    if ! run_for_embedder "$embedder" evaluate; then
      RUN_FAILED=1
      echo "[$embedder] embedding/training/evaluation failed; continuing." >&2
    elif [[ "$WEIGHT_FAILED" -ne 0 ]]; then
      echo "[$embedder] recovered using cached assets." >&2
    fi
  done

  if [[ "$RUN_FAILED" -eq 0 ]]; then
    # mtime-only triggering plus the ancient() 650M input in the plot rule
    # guarantees that this aggregation step consumes, but never rebuilds, the
    # completed 650M evaluation.
    if ! "${SNAKEMAKE[@]}" --rerun-triggers mtime --keep-going esm2_ablation; then
      RUN_FAILED=1
      echo "ESM2 ablation plotting failed; completed size runs remain cached." >&2
    fi
  else
    echo "Skipping the ESM2 ablation plot until every new size evaluates." >&2
  fi

  echo
  echo "ESM2 ablation figure: artifacts/analysis/figures/fig7_esm2_size_ablation.png"
  echo "ESM2 ablation data: artifacts/analysis/figures/fig7_esm2_size_ablation_data.csv"
  exit "$RUN_FAILED"
fi

if [[ "$MODE" == "antigen-controls" ]]; then
  SHUFFLED_SOURCE_EVAL="artifacts/runs/esm2__c49369e9/eval.json"
  SHUFFLED_SOURCE_DIR="artifacts/data/sabdab2_v0.1.0/embeddings/antigen/esm2"
  if [[ ! -f "$SHUFFLED_SOURCE_EVAL" || ! -d "$SHUFFLED_SOURCE_DIR" ]]; then
    echo "The cached ESM2 650M evaluation and embeddings are required for the shuffled control." >&2
    echo "Missing source: $SHUFFLED_SOURCE_EVAL or $SHUFFLED_SOURCE_DIR" >&2
    exit 1
  fi

  RUN_FAILED=0
  for embedder in control_constant control_random control_shuffled control_pooled_esm2; do
    echo
    echo "===== MANGO antigen control: $embedder ====="
    if ! run_for_embedder "$embedder" evaluate --rerun-triggers=mtime; then
      RUN_FAILED=1
      echo "[$embedder] embedding/training/evaluation failed; continuing." >&2
    fi
  done

  if [[ "$RUN_FAILED" -eq 0 ]]; then
    if ! "${SNAKEMAKE[@]}" --rerun-triggers mtime --keep-going antigen_controls; then
      RUN_FAILED=1
      echo "Antigen-control plotting failed; completed models remain cached." >&2
    fi
  else
    echo "Skipping the antigen-control plot until all four controls evaluate." >&2
  fi

  echo
  echo "Control evaluations are under artifacts/runs/control_*/eval.json"
  echo "Control figure: artifacts/analysis/figures/fig8_antigen_controls.png"
  echo "Control data: artifacts/analysis/figures/fig8_antigen_controls_data.csv"
  exit "$RUN_FAILED"
fi

RUN_FAILED=0
PREFETCH_PID=""
PREFETCH_TAG=""
for i in "${!EMBEDDERS[@]}"; do
  embedder="${EMBEDDERS[$i]}"
  echo
  echo "===== MANGO embedder: $embedder ====="

  # A future embedder's asset-only DAG may run beside the current embedder's
  # compute DAG. It uses disjoint scientific outputs and --nolock because the
  # foreground Snakemake process owns the repository lock. GPU-heavy embedding
  # and training remain exclusively in the foreground and therefore serialized.
  if [[ -n "$PREFETCH_PID" && "$PREFETCH_TAG" == "$embedder" ]]; then
    if ! wait "$PREFETCH_PID"; then
      echo "[$embedder] overlapped prefetch failed; retrying in foreground." >&2
    fi
    PREFETCH_PID=""
    PREFETCH_TAG=""
  fi

  # Resolve this embedder's model assets before its long compute jobs. Failures
  # are recorded, but never prevent the next embedder from being attempted.
  WEIGHT_FAILED=0
  if ! run_for_embedder "$embedder" weights; then
    WEIGHT_FAILED=1
    echo "[$embedder] weight/asset setup failed; continuing." >&2
  fi
  if [[ "$MODE" == weights* ]]; then
    if [[ "$WEIGHT_FAILED" -ne 0 ]]; then
      RUN_FAILED=1
    fi
  else
    # Once the current assets are safe, overlap the next asset-bearing method's
    # downloads/environment creation with this method's GPU run.
    if [[ -z "$PREFETCH_PID" ]]; then
      for ((j=i+1; j<${#EMBEDDERS[@]}; j++)); do
        candidate="${EMBEDDERS[$j]}"
        if requires_prefetch "$candidate"; then
          echo "[$embedder] prefetching future embedder $candidate in background."
          # Concurrent Conda writers must not share a package cache on network
          # filesystems: interrupted extraction can otherwise poison an
          # unrelated foreground solve. The finished rule environment remains
          # in the normal per-rule prefix and is reused by the foreground DAG.
          (
            export CONDA_PKGS_DIRS="$ROOT_DIR/.snakemake/conda-pkgs-prefetch/$candidate"
            mkdir -p "$CONDA_PKGS_DIRS"
            run_for_embedder "$candidate" weights --nolock
          ) &
          PREFETCH_PID=$!
          PREFETCH_TAG="$candidate"
          break
        fi
      done
    fi

    # Produce train/evaluate/predict outputs before optional analyses. This
    # guarantees that an analysis-only environment failure cannot hide usable
    # NLL, perplexity, predictions or the training curve.
    if ! run_for_embedder "$embedder" inference; then
      RUN_FAILED=1
      echo "[$embedder] core run is incomplete; continuing with the next embedder." >&2
    elif [[ "$WEIGHT_FAILED" -ne 0 ]]; then
      echo "[$embedder] recovered during the full run after prefetch failed." >&2
    fi

    # Refresh the all-method inventory after every foreground run. This makes
    # completed NLL/perplexity/prediction counts immediately available while
    # later embedders continue; the final strict figures still wait for all.
    if ! "${SNAKEMAKE[@]}" --force gpu_report; then
      RUN_FAILED=1
      echo "[$embedder] incremental report refresh failed; continuing." >&2
    fi

    # Analysis is modular and may fail independently after core results exist.
    if ! run_for_embedder "$embedder" gpu_embedder_result; then
      RUN_FAILED=1
      echo "[$embedder] analysis is incomplete; core results remain available." >&2
    fi
    if ! "${SNAKEMAKE[@]}" --force gpu_report; then
      RUN_FAILED=1
      echo "[$embedder] post-analysis report refresh failed; continuing." >&2
    fi
    if ! refresh_review_bundle; then
      RUN_FAILED=1
      echo "[$embedder] smoke review bundle refresh failed; continuing." >&2
    fi
  fi
done


if [[ -n "$PREFETCH_PID" ]]; then
  if ! wait "$PREFETCH_PID"; then
    RUN_FAILED=1
    echo "[$PREFETCH_TAG] final overlapped prefetch failed." >&2
  fi
fi

if [[ "$MODE" == weights* ]]; then
  echo
  if [[ "$RUN_FAILED" -eq 0 ]]; then
    echo "MANGO weights are ready under artifacts/weights and artifacts/cache."
  else
    echo "Some MANGO weights failed; every requested embedder was still attempted." >&2
  fi
  exit "$RUN_FAILED"
fi

# Cross-method plots and the strict completion manifest are meaningful only if
# every isolated method finished. All constituent files are cached at this
# point, so this final combined invocation is lightweight.
if [[ "$RUN_FAILED" -eq 0 ]]; then
  # Per-embedder jobs intentionally run with a singleton active_embedders list,
  # whereas aggregation sees the full list.  That orchestration-only change is
  # captured inside broad rule params, but it must not invalidate scientific
  # outputs.  Keep every substantive rerun trigger while excluding only params
  # for this read-mostly combined DAG.
  if ! "${SNAKEMAKE[@]}" --rerun-triggers mtime input software-env code \
      --keep-going gpu_results; then
    RUN_FAILED=1
    echo "Cross-embedder aggregation failed; writing the per-embedder report." >&2
  fi
else
  echo "At least one embedder is incomplete; skipping strict cross-method plots." >&2
fi

# This target has no scientific inputs on purpose: force it after the main DAG
# so it can inventory whatever succeeded even when another branch failed.
"${SNAKEMAKE[@]}" --force gpu_report
refresh_review_bundle

echo
"$DRIVER/bin/python" -c \
  'import json; r=json.load(open("artifacts/gpu/results_report.json")); s=r["summary"]; print("MANGO GPU report: {}/{} core runs succeeded; fully complete={}; analysis incomplete={}; failed={}".format(s["core_successful"], s["expected"], s["fully_complete_embedders"], s["analysis_incomplete_embedders"], s["failed_or_incomplete_embedders"]))'
echo "Numbers: artifacts/gpu/results_report.csv"
echo "Details: artifacts/gpu/results_report.json"
if [[ "$RUN_FAILED" -ne 0 ]]; then
  echo "MANGO finished partially; returning a failure status so automation notices." >&2
  exit 1
fi
echo "Strict completion manifest: artifacts/gpu/results_complete.json"
