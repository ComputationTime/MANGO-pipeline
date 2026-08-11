#!/usr/bin/env bash
set -euo pipefail

# One-command local NVIDIA GPU runner.
#   ./run_gpu.sh smoke        five-record plumbing run (default)
#   ./run_gpu.sh study        full configured SAbDab2 study
#   ./run_gpu.sh smoke-esm3   smoke + gated ESM3
#   ./run_gpu.sh study-esm3   full study + gated ESM3
#   ./run_gpu.sh smoke-pyrosetta / study-pyrosetta
#   ./run_gpu.sh smoke-all / study-all (ESM3 + PyRosetta)
#   ./run_gpu.sh weights[-esm3|-pyrosetta|-all] (downloads only)

MODE="${1:-smoke}"
case "$MODE" in
  smoke|study|smoke-esm3|study-esm3|smoke-pyrosetta|study-pyrosetta|smoke-all|study-all|weights|weights-esm3|weights-pyrosetta|weights-all) ;;
  *) echo "usage: $0 {smoke|study|weights}[-esm3|-pyrosetta|-all]" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is missing; install an NVIDIA driver before running MANGO." >&2
  exit 1
}
nvidia-smi >/dev/null

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  echo "conda is required for isolated embedder environments." >&2
  exit 1
fi

export MANGO_REQUIRE_CUDA=1
export PYTHONNOUSERSITE=1
export CONDA_CHANNEL_PRIORITY=strict
export HF_HOME="${HF_HOME:-$ROOT_DIR/artifacts/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/artifacts/cache/torch}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT_DIR/.snakemake/conda-pkgs}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CONDA_PKGS_DIRS"

if [[ "$MODE" == *-esm3 || "$MODE" == *-all ]]; then
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
fi
if [[ "$MODE" == *-esm3 ]]; then
  CONFIG_ARGS+=(config/gpu_esm3.yaml)
elif [[ "$MODE" == *-pyrosetta ]]; then
  CONFIG_ARGS+=(config/gpu_pyrosetta.yaml)
elif [[ "$MODE" == *-all ]]; then
  CONFIG_ARGS+=(config/gpu_all.yaml)
fi

SNAKEMAKE=("$DRIVER/bin/snakemake" -s workflow/Snakefile \
  --profile workflow/profiles/gpu "${CONFIG_ARGS[@]}" --rerun-incomplete)

EMBEDDERS=(one_hot biopython esm2 esmif proteinmpnn)
if [[ "$MODE" == *-pyrosetta || "$MODE" == *-all ]]; then
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
  # A separate Snakemake invocation is intentional. In a combined DAG,
  # environment creation happens before execution and one invalid environment
  # can prevent otherwise independent embedder jobs from ever starting.
  "${SNAKEMAKE[@]}" \
    --config "active_embedders=[$embedder]" \
    --keep-going "$target"
}

# Force the preflight on every invocation so a manifest copied from another
# machine can never bypass validation of this GPU and driver.
"${SNAKEMAKE[@]}" --force gpu_preflight
RUN_FAILED=0
for embedder in "${EMBEDDERS[@]}"; do
  echo
  echo "===== MANGO embedder: $embedder ====="
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
    if ! run_for_embedder "$embedder" gpu_embedder_result; then
      RUN_FAILED=1
      echo "[$embedder] run is incomplete; continuing with the next embedder." >&2
    elif [[ "$WEIGHT_FAILED" -ne 0 ]]; then
      echo "[$embedder] recovered during the full run after prefetch failed." >&2
    fi
  fi
done

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
  if ! "${SNAKEMAKE[@]}" --keep-going gpu_results; then
    RUN_FAILED=1
    echo "Cross-embedder aggregation failed; writing the per-embedder report." >&2
  fi
else
  echo "At least one embedder is incomplete; skipping strict cross-method plots." >&2
fi

# This target has no scientific inputs on purpose: force it after the main DAG
# so it can inventory whatever succeeded even when another branch failed.
"${SNAKEMAKE[@]}" --force gpu_report

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
