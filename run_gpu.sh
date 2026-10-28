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
#   ./run_gpu.sh antigen-controls (constant, random, pooled, shuffled, cross-attention)
#   ./run_gpu.sh antigen-length-ablation (constant/random/token-merged ESM2 length sweep)
#   ./run_gpu.sh iglm-baseline (IgLM from scratch, then the context-length figure)
#   ./run_gpu.sh abrank-transfer (audited AbRank controls, then SAbDab2 fine-tuning)
#   ./run_gpu.sh asd-transfer (audited ASD 2.0 controls, then SAbDab2 fine-tuning)
#   ./run_gpu.sh asd-loss-comparison (ESM2 CE/focal, normal/cross-attention, with/without ASD)
#   ./run_gpu.sh asd-prototype (small HER2-free ASD, one epoch per training stage)
#   ./run_gpu.sh asd-prototype-iglm (matched IgLM controls added to the prototype)
#   ./run_gpu.sh antigen-diagnostics (small paired probes of pilot checkpoints)
#   ./run_gpu.sh antigen-diagnostic-training (one-epoch controlled follow-ups)
#   ./run_gpu.sh asd-scaling (nested ASD-only 2k/4k data-size comparison)
#   ./run_gpu.sh asd-scaling-8k (extend the frozen comparison to 8k records)
#   ./run_gpu.sh asd-scaling-16k (16k extension and regular MANGO at every size)
#   ./run_gpu.sh asd-scaling-fixed-positions (add fixed-position MANGO at every size)

MODE="${1:-smoke}"
case "$MODE" in
  smoke|small|study|smoke-esm3|small-esm3|study-esm3|smoke-pyrosetta|small-pyrosetta|study-pyrosetta|smoke-all|small-all|study-all|weights|weights-esm3|weights-pyrosetta|weights-all|esm2-ablation|antigen-controls|antigen-length-ablation|iglm-baseline|abrank-transfer|asd-transfer|asd-loss-comparison|asd-prototype|asd-prototype-iglm|antigen-diagnostics|antigen-diagnostic-training|asd-scaling|asd-scaling-8k|asd-scaling-16k|asd-scaling-fixed-positions) ;;
  *) echo "usage: $0 {smoke|small|study|weights}[-esm3|-pyrosetta|-all] | esm2-ablation | antigen-controls | antigen-length-ablation | iglm-baseline | abrank-transfer | asd-transfer | asd-loss-comparison | asd-prototype | asd-prototype-iglm | antigen-diagnostics | antigen-diagnostic-training | asd-scaling | asd-scaling-8k | asd-scaling-16k | asd-scaling-fixed-positions" >&2; exit 2 ;;
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
if [[ -z "$CONDA_BIN" && -x "$ROOT_DIR/artifacts/miniforge3/bin/conda" ]]; then
  CONDA_BIN="$ROOT_DIR/artifacts/miniforge3/bin/conda"
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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/artifacts/cache}"
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

if [[ "$MODE" == "asd-scaling" || "$MODE" == "asd-scaling-8k" || "$MODE" == "asd-scaling-16k" || "$MODE" == "asd-scaling-fixed-positions" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  SCALE_CONFIG=config/asd_scaling.yaml
  SCALE_REPORT=workflow/scripts/report_asd_scaling.py
  if [[ "$MODE" == "asd-scaling-fixed-positions" ]]; then
    SCALE_CONFIG=config/asd_scaling_fixed_positions.yaml
    SCALE_REPORT=workflow/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_scaling_16k.py --config "$SCALE_CONFIG"
  elif [[ "$MODE" == "asd-scaling-16k" ]]; then
    SCALE_CONFIG=config/asd_scaling_16k.yaml
    SCALE_REPORT=workflow/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_scaling_16k.py
  elif [[ "$MODE" == "asd-scaling-8k" ]]; then
    SCALE_CONFIG=config/asd_scaling_8k.yaml
    SCALE_REPORT=workflow/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_scaling_8k.py
  else
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_scaling.py
  fi
  export MANGO_ASD_SCALING_CONFIG="$SCALE_CONFIG"
  mapfile -t SCALE_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(*c["runs"], sep="\n")' "$SCALE_CONFIG")
  SCALE_ROOT="${SCALE_SETTINGS[0]}"
  SCALE_SNAKE=("$DRIVER/bin/snakemake" -s workflow/asd_scaling.smk \
    --profile workflow/profiles/gpu --rerun-incomplete)
  for run in "${SCALE_SETTINGS[@]:1}"; do
    if ! "${SCALE_SNAKE[@]}" -- "$SCALE_ROOT/runs/$run/eval.json"; then
      echo "[asd-scaling/$run] incomplete; continuing with other models." >&2
    fi
    "$DRIVER/bin/python" "$SCALE_REPORT" --config "$SCALE_CONFIG" --status-only
  done
  if ! "${SCALE_SNAKE[@]}" --keep-going; then
    "$DRIVER/bin/python" "$SCALE_REPORT" --config "$SCALE_CONFIG" --status-only
    exit 1
  fi
  exit 0
fi

if [[ "$MODE" == "antigen-diagnostic-training" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  mapfile -t DIAGNOSTIC_SETTINGS < <("$DRIVER/bin/python" -c \
    'import yaml; c=yaml.safe_load(open("config/antigen_diagnostic_training.yaml")); print(c["experiment_root"]); print(*c["arms"], sep="\n")')
  DIAGNOSTIC_ROOT="${DIAGNOSTIC_SETTINGS[0]}"
  DIAGNOSTIC_SNAKE=("$DRIVER/bin/snakemake" -s workflow/antigen_diagnostic_training.smk \
    --profile workflow/profiles/gpu --rerun-incomplete)
  for arm in "${DIAGNOSTIC_SETTINGS[@]:1}"; do
    if ! "${DIAGNOSTIC_SNAKE[@]}" -- "$DIAGNOSTIC_ROOT/$arm/native/complete.json"; then
      echo "[diagnostic-training/$arm] incomplete; continuing with other arms." >&2
    fi
  done
  "${DIAGNOSTIC_SNAKE[@]}" --keep-going
  exit 0
fi

if [[ "$MODE" == "antigen-diagnostics" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  mapfile -t DIAGNOSTIC_SETTINGS < <("$DRIVER/bin/python" -c \
    'import yaml; c=yaml.safe_load(open("config/antigen_diagnostics.yaml")); print(c["experiment_root"]); print(*c["probes"], sep="\n")')
  DIAGNOSTIC_ROOT="${DIAGNOSTIC_SETTINGS[0]}"
  DIAGNOSTIC_SNAKE=("$DRIVER/bin/snakemake" -s workflow/antigen_diagnostics.smk \
    --profile workflow/profiles/gpu --rerun-incomplete)
  for model in "${DIAGNOSTIC_SETTINGS[@]:1}"; do
    if ! "${DIAGNOSTIC_SNAKE[@]}" -- "$DIAGNOSTIC_ROOT/probes/$model/complete.json"; then
      echo "[diagnostics/$model] incomplete; continuing with other probes." >&2
    fi
  done
  "${DIAGNOSTIC_SNAKE[@]}" --keep-going
  exit 0
fi

if [[ "$MODE" == "asd-prototype-iglm" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  mapfile -t IGLM_SETTINGS < <("$DRIVER/bin/python" -c \
    'import yaml; c=yaml.safe_load(open("config/asd_prototype_iglm.yaml")); print(c["experiment_root"]); print(c["mango_root"]); print(c["cohort_dir"])')
  IGLM_ROOT="${IGLM_SETTINGS[0]}"
  IGLM_MANGO="${IGLM_SETTINGS[1]}"
  IGLM_COHORT="${IGLM_SETTINGS[2]}"
  IGLM_SNAKE=("$DRIVER/bin/snakemake" -s workflow/asd_prototype_iglm.smk \
    --profile workflow/profiles/gpu --rerun-incomplete)
  IGLM_REPORT=("$DRIVER/bin/python" workflow/scripts/report_prototype_iglm.py \
    --root "$IGLM_ROOT" --mango-root "$IGLM_MANGO" --cohort "$IGLM_COHORT")
  for phase in sabdab2_scratch sabdab2_finetuned asd; do
    if ! "${IGLM_SNAKE[@]}" -- "$IGLM_ROOT/$phase/eval.json"; then
      echo "[IgLM/$phase] incomplete; continuing with other stages." >&2
    fi
    "${IGLM_REPORT[@]}"
  done
  if ! "${IGLM_SNAKE[@]}" --keep-going; then
    echo "Some IgLM outputs remain incomplete; writing the partial report." >&2
  fi
  "${IGLM_REPORT[@]}" --require-complete
  exit 0
fi

if [[ "$MODE" == "asd-loss-comparison" || "$MODE" == "asd-prototype" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  LOSS_CONFIG="config/${MODE//-/_}.yaml"
  mapfile -t LOSS_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(c["cohort_dir"]); print(*[f"{loss}/{arch}" for loss in c["objectives"] for arch in c["architectures"]], sep="\n")' "$LOSS_CONFIG")
  LOSS_ROOT="${LOSS_SETTINGS[0]}"
  LOSS_COHORT="${LOSS_SETTINGS[1]}"
  LOSS_ARMS=("${LOSS_SETTINGS[@]:2}")
  mkdir -p "$LOSS_ROOT"
  if [[ "$MODE" == "asd-prototype" ]]; then
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_prototype.py --config "$LOSS_CONFIG"
  else
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_transfer.py --output "$LOSS_COHORT"
  fi
  LOSS_SNAKE=("$DRIVER/bin/snakemake" -s workflow/asd_loss_comparison.smk \
    --profile workflow/profiles/gpu --configfile "$LOSS_CONFIG" --rerun-incomplete)
  for arm in "${LOSS_ARMS[@]}"; do
    if ! "${LOSS_SNAKE[@]}" -- "$LOSS_ROOT/sabdab2_scratch/$arm/eval.json"; then
      echo "[sabdab2_scratch/$arm] incomplete; continuing with other models." >&2
    fi
    "$DRIVER/bin/python" workflow/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT"
  done
  for arm in "${LOSS_ARMS[@]}"; do
    for phase in sabdab2_finetuned asd; do
      if ! "${LOSS_SNAKE[@]}" -- "$LOSS_ROOT/$phase/$arm/eval.json"; then
        echo "[$phase/$arm] incomplete; continuing with other models." >&2
      fi
      "$DRIVER/bin/python" workflow/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT"
    done
  done
  if ! "${LOSS_SNAKE[@]}" --keep-going; then
    echo "Some loss-comparison outputs remain incomplete; writing the partial report." >&2
  fi
  "$DRIVER/bin/python" workflow/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT" --require-complete
  exit 0
fi

if [[ "$MODE" == "abrank-transfer" || "$MODE" == "asd-transfer" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  TRANSFER_DATASET="${MODE%-transfer}"
  mapfile -t TRANSFER_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(c["cohort_dir"]); print(*c["controls"], sep="\n")' \
    "config/${TRANSFER_DATASET}_transfer.yaml")
  TRANSFER_ROOT="${TRANSFER_SETTINGS[0]}"
  TRANSFER_COHORT="${TRANSFER_SETTINGS[1]}"
  TRANSFER_TAGS=("${TRANSFER_SETTINGS[@]:2}")
  if [[ "$TRANSFER_DATASET" == "asd" ]]; then
    "$DRIVER/bin/python" workflow/scripts/prepare_asd_transfer.py --output "$TRANSFER_COHORT"
  fi
  TRANSFER_SNAKE=("$DRIVER/bin/snakemake" -s "workflow/${TRANSFER_DATASET}_transfer.smk" \
    --profile workflow/profiles/gpu --rerun-incomplete)
  for phase in "$TRANSFER_DATASET" sabdab2; do
    for tag in "${TRANSFER_TAGS[@]}"; do
      if ! "${TRANSFER_SNAKE[@]}" -- "$TRANSFER_ROOT/$phase/runs/$tag/eval.json"; then
        echo "[$phase/$tag] incomplete; continuing with other controls." >&2
      fi
    done
  done
  if ! "${TRANSFER_SNAKE[@]}" --keep-going; then
    echo "Some requested transfer outputs remain incomplete; writing the partial report." >&2
  fi
  "$DRIVER/bin/python" workflow/scripts/report_abrank_transfer.py --root "$TRANSFER_ROOT" --require-complete
  exit 0
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
elif [[ "$MODE" == "antigen-length-ablation" || "$MODE" == "iglm-baseline" ]]; then
  CONFIG_ARGS+=(config/antigen_controls.yaml config/antigen_length_ablation.yaml config/iglm_baseline.yaml)
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
    --keep-going "$@" -- "$target"
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

if [[ "$MODE" == "iglm-baseline" ]]; then
  if [[ ! -f artifacts/analysis/figures/fig9_antigen_context_length_ablation_data.csv ]]; then
    echo "Run ./run_gpu.sh antigen-length-ablation first to create the MANGO comparison." >&2
    exit 1
  fi
  "${SNAKEMAKE[@]}" --rerun-triggers mtime --keep-going iglm_baseline
  "${SNAKEMAKE[@]}" --rerun-triggers mtime --keep-going antigen_length_ablation
  echo "IgLM comparison: artifacts/analysis/figures/fig9_antigen_context_length_ablation.png"
  exit 0
fi

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

if [[ "$MODE" == "antigen-controls" || "$MODE" == "antigen-length-ablation" ]]; then
  SHUFFLED_SOURCE_EVAL="artifacts/runs/esm2__c49369e9/eval.json"
  SHUFFLED_SOURCE_DIR="artifacts/data/sabdab2_v0.1.0/embeddings/antigen/esm2"
  if [[ ! -f "$SHUFFLED_SOURCE_EVAL" || ! -d "$SHUFFLED_SOURCE_DIR" ]]; then
    echo "The cached ESM2 650M evaluation and embeddings are required for the antigen controls." >&2
    echo "Missing source: $SHUFFLED_SOURCE_EVAL or $SHUFFLED_SOURCE_DIR" >&2
    exit 1
  fi

  CONTROL_TAGS=(control_constant control_random control_shuffled control_pooled_esm2 control_cross_attention_esm2)
  CONTROL_TARGET=antigen_controls
  CONTROL_FIGURE=fig8_antigen_controls
  CONTROL_RERUN_ARGS=(--rerun-triggers=mtime)
  if [[ "$MODE" == "antigen-length-ablation" ]]; then
    CONTROL_TAGS=(control_constant_l1 control_constant_l25 control_constant_l50 control_constant_l75 control_constant
                  control_random_l1 control_random_l25 control_random_l50 control_random_l75 control_random
                  control_pooled_esm2 control_merged_esm2_l25 control_merged_esm2_l50 control_merged_esm2_l75)
    CONTROL_TARGET=antigen_length_ablation
    CONTROL_FIGURE=fig9_antigen_context_length_ablation
    # Re-evaluate older checkpoints when evaluation code gains context-length
    # metadata; completed training and unchanged evaluations remain reusable.
    CONTROL_RERUN_ARGS=(--rerun-triggers mtime code)
  fi
  RUN_FAILED=0
  for embedder in "${CONTROL_TAGS[@]}"; do
    echo
    echo "===== MANGO antigen control: $embedder ====="
    if ! run_for_embedder "$embedder" evaluate "${CONTROL_RERUN_ARGS[@]}"; then
      RUN_FAILED=1
      echo "[$embedder] embedding/training/evaluation failed; continuing." >&2
    fi
  done

  if [[ "$MODE" == "antigen-length-ablation" ]]; then
    if ! "${SNAKEMAKE[@]}" --rerun-triggers mtime --keep-going iglm_baseline; then
      RUN_FAILED=1
      echo "IgLM baseline failed; completed MANGO controls remain cached." >&2
    fi
  fi

  if [[ "$RUN_FAILED" -eq 0 ]]; then
    if ! "${SNAKEMAKE[@]}" "${CONTROL_RERUN_ARGS[@]}" --keep-going "$CONTROL_TARGET"; then
      RUN_FAILED=1
      echo "Antigen-control plotting failed; completed models remain cached." >&2
    fi
  else
    echo "Skipping the antigen-control plot until all ${#CONTROL_TAGS[@]} controls evaluate." >&2
  fi

  echo
  echo "Control evaluations are under artifacts/runs/control_*/eval.json"
  echo "Control figure: artifacts/analysis/figures/$CONTROL_FIGURE.png"
  echo "Control data: artifacts/analysis/figures/${CONTROL_FIGURE}_data.csv"
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
