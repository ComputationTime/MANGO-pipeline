#!/usr/bin/env bash
set -euo pipefail

# UNSUPPORTED side-study launcher (see experiments/README.md).
# Reuses the same bootstrap as ../run_gpu.sh. Usage: ./experiments/run_experiment.sh <mode>
#
#   ./experiments/run_experiment.sh abrank-transfer (audited AbRank controls, then SAbDab2 fine-tuning)
#   ./experiments/run_experiment.sh asd-transfer (audited ASD 2.0 controls, then SAbDab2 fine-tuning)
#   ./experiments/run_experiment.sh asd-loss-comparison (ESM2 CE/focal, normal/cross-attention, with/without ASD)
#   ./experiments/run_experiment.sh asd-prototype (small HER2-free ASD, one epoch per training stage)
#   ./experiments/run_experiment.sh asd-prototype-iglm (matched IgLM controls added to the prototype)
#   ./experiments/run_experiment.sh antigen-diagnostics (small paired probes of pilot checkpoints)
#   ./experiments/run_experiment.sh antigen-diagnostic-training (one-epoch controlled follow-ups)
#   ./experiments/run_experiment.sh asd-scaling (nested ASD-only 2k/4k data-size comparison)
#   ./experiments/run_experiment.sh asd-scaling-8k (extend the frozen comparison to 8k records)
#   ./experiments/run_experiment.sh asd-scaling-16k (16k extension and regular MANGO at every size)
#   ./experiments/run_experiment.sh asd-scaling-fixed-positions (add fixed-position MANGO at every size)
#   ./experiments/run_experiment.sh openalphaseq-pairing (external pairing benchmark of frozen 16k models)
#   ./experiments/run_experiment.sh openalphaseq-affinity (full dataset affinity correlations for train-all)
#   ./experiments/run_experiment.sh binding-affinity (local HER2/PD1/TIGIT CSVs; eight MANGO models + IgLM)
#   ./experiments/run_experiment.sh native-mechanism-audit (protected native cohort feasibility)
#   ./experiments/run_experiment.sh legacy-adapter-affinity (frozen adapter affinity comparison; no fitting)
#   ./experiments/run_experiment.sh affinity-bottleneck-audit (typed grouping and affinity-source curation)
#   ./experiments/run_experiment.sh affinity-improvement-{smoke,study} (completion-driven controlled experiments)
#   ./experiments/run_experiment.sh controlled-antigen-{smoke,study} (decoder-matched antigen ablations)
#   ./experiments/run_experiment.sh short-affinity-{smoke,study} (bounded six-test affinity screen)
#   ./experiments/run_experiment.sh native-mechanism-{smoke,study} (fixed-budget matched mechanism pilot)

MODE="${1:-}"
case "$MODE" in
  abrank-transfer|affinity-bottleneck-audit|affinity-improvement-smoke|affinity-improvement-study|affinity-ranking-pilot|affinity-ranking-pilot-smoke|antigen-diagnostic-training|antigen-diagnostics|asd-loss-comparison|asd-prototype|asd-prototype-iglm|asd-scaling|asd-scaling-16k|asd-scaling-8k|asd-scaling-fixed-positions|asd-transfer|binding-affinity|controlled-antigen-smoke|controlled-antigen-study|epitope-diagnostics|epitope-diagnostics-smoke|interface-supervision|interface-supervision-smoke|legacy-adapter-affinity|native-mechanism-audit|native-mechanism-smoke|native-mechanism-study|openalphaseq-affinity|openalphaseq-pairing|short-affinity-smoke|short-affinity-study) ;;
  *) echo "usage: $0 <mode>; see the header of this file for the list of modes" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

if [[ "$MODE" == "interface-supervision" || "$MODE" == "interface-supervision-smoke" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  if [[ "$MODE" == "interface-supervision-smoke" ]]; then
    "$DRIVER/bin/python" - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('experiments/config/interface_supervision.yaml').read_text())
root=Path('artifacts/experiments/interface_supervision_smoke_v1');root.mkdir(parents=True,exist_ok=True)
cfg.update(experiment_root=str(root),smoke=True,seeds=[300],pretraining_updates=2,adaptation_updates=2)
(root/'config.yaml').write_text(yaml.safe_dump(cfg))
PY
    export MANGO_INTERFACE_CONFIG=artifacts/experiments/interface_supervision_smoke_v1/config.yaml
  fi
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/interface_supervision.smk --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "epitope-diagnostics" || "$MODE" == "epitope-diagnostics-smoke" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  if [[ "$MODE" == "epitope-diagnostics-smoke" ]]; then
    "$DRIVER/bin/python" - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('experiments/config/epitope_diagnostics.yaml').read_text())
root=Path('artifacts/experiments/epitope_diagnostics_smoke_v2');root.mkdir(parents=True,exist_ok=True)
cfg.update(experiment_root=str(root),smoke=True,seeds=[200])
(root/'config.yaml').write_text(yaml.safe_dump(cfg))
PY
    export MANGO_EPITOPE_CONFIG=artifacts/experiments/epitope_diagnostics_smoke_v2/config.yaml
  fi
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/epitope_diagnostics.smk --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "affinity-ranking-pilot" || "$MODE" == "affinity-ranking-pilot-smoke" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  if [[ "$MODE" == "affinity-ranking-pilot-smoke" ]]; then
    "$DRIVER/bin/python" - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('experiments/config/affinity_ranking_pilot.yaml').read_text())
root=Path('artifacts/experiments/affinity_ranking_pilot_smoke_v1');root.mkdir(parents=True,exist_ok=True)
cfg.update(experiment_root=str(root),smoke=True,seeds=[200],pretraining_updates=2,adaptation_updates=2)
(root/'config.yaml').write_text(yaml.safe_dump(cfg))
PY
    export MANGO_RANKING_PILOT_CONFIG=artifacts/experiments/affinity_ranking_pilot_smoke_v1/config.yaml
  fi
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/affinity_ranking_pilot.smk \
    --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "affinity-bottleneck-audit" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/affinity_bottleneck.smk \
    --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "native-mechanism-audit" ]]; then
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/native_mechanism.smk \
    --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "legacy-adapter-affinity" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  exec "$DRIVER/bin/python" experiments/scripts/legacy_adapter_affinity.py run \
    --config "${MANGO_ADAPTER_AFFINITY_CONFIG:-config/legacy_adapter_affinity.yaml}"
fi

if [[ "$MODE" == "native-mechanism-smoke" || "$MODE" == "native-mechanism-study" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  NATIVE_ARGS=(--config "${MANGO_NATIVE_MECHANISM_CONFIG:-config/native_mechanism_pilot_v2.yaml}")
  if [[ "$MODE" == "native-mechanism-smoke" ]]; then NATIVE_ARGS+=(--smoke); fi
  exec "$DRIVER/bin/python" experiments/scripts/run_native_mechanism.py "${NATIVE_ARGS[@]}"
fi

if [[ "$MODE" == "short-affinity-smoke" || "$MODE" == "short-affinity-study" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  SHORT_ARGS=(--config "${MANGO_SHORT_CONFIG:-config/short_affinity_tests.yaml}")
  if [[ "$MODE" == "short-affinity-smoke" ]]; then
    SHORT_ARGS+=(--smoke)
  fi
  exec "$DRIVER/bin/python" experiments/scripts/run_short_affinity.py "${SHORT_ARGS[@]}"
fi

if [[ "$MODE" == "controlled-antigen-smoke" || "$MODE" == "controlled-antigen-study" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  CONTROLLED_ARGS=(--config "${MANGO_CONTROLLED_CONFIG:-config/controlled_antigen_study.yaml}")
  if [[ "$MODE" == "controlled-antigen-smoke" ]]; then
    CONTROLLED_ARGS+=(--smoke)
  fi
  exec "$DRIVER/bin/python" experiments/scripts/run_controlled_antigen.py "${CONTROLLED_ARGS[@]}"
fi

if [[ "$MODE" == "affinity-improvement-smoke" || "$MODE" == "affinity-improvement-study" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  IMPROVEMENT_ARGS=(--config "${MANGO_AFFINITY_STUDY_CONFIG:-config/affinity_improvement_study.yaml}")
  if [[ "$MODE" == "affinity-improvement-smoke" ]]; then
    IMPROVEMENT_ARGS+=(--smoke)
  fi
  exec "$DRIVER/bin/python" experiments/scripts/run_affinity_study.py "${IMPROVEMENT_ARGS[@]}"
fi

if [[ "$MODE" == "openalphaseq-pairing" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  exec "$DRIVER/bin/snakemake" -s experiments/workflow/openalphaseq_pairing.smk \
    --profile workflow/profiles/gpu --rerun-incomplete
fi

if [[ "$MODE" == "openalphaseq-affinity" || "$MODE" == "binding-affinity" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  AFFINITY_ROOT=artifacts/experiments/openalphaseq_affinity_full_v1
  AFFINITY_WORKFLOW=experiments/workflow/openalphaseq_affinity.smk
  if [[ "$MODE" == "binding-affinity" ]]; then
    AFFINITY_ROOT=artifacts/experiments/mango_binding_affinity
    AFFINITY_WORKFLOW=experiments/workflow/mango_binding_affinity.smk
  fi
  AFFINITY_SNAKE=("$DRIVER/bin/snakemake" -s "$AFFINITY_WORKFLOW"
    --profile workflow/profiles/gpu --rerun-incomplete --rerun-triggers mtime)
  "${AFFINITY_SNAKE[@]}" -- "$AFFINITY_ROOT/measurements_complete.json"
  RUN_FAILED=0
  # Separate invocations keep an optional embedder environment failure from
  # preventing every other model's correlation. GPU work remains sequential.
  for model in one_hot biopython esm2 esm3 iglm pyrosetta_pre esm3_structure esmif proteinmpnn; do
    if ! "${AFFINITY_SNAKE[@]}" -- "$AFFINITY_ROOT/scoring/$model.complete.json"; then
      RUN_FAILED=1
      echo "[$model] AlphaSeq scoring incomplete; continuing with other models." >&2
    fi
  done
  if ! "${AFFINITY_SNAKE[@]}" --keep-going; then
    RUN_FAILED=1
    if [[ "$MODE" == "binding-affinity" ]]; then
      "${AFFINITY_SNAKE[@]}" -- "$AFFINITY_ROOT/partial/local_summary.json" || true
    fi
  fi
  exit "$RUN_FAILED"
fi

if [[ "$MODE" == "asd-scaling" || "$MODE" == "asd-scaling-8k" || "$MODE" == "asd-scaling-16k" || "$MODE" == "asd-scaling-fixed-positions" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  SCALE_CONFIG=experiments/config/asd_scaling.yaml
  SCALE_REPORT=experiments/scripts/report_asd_scaling.py
  if [[ "$MODE" == "asd-scaling-fixed-positions" ]]; then
    SCALE_CONFIG=experiments/config/asd_scaling_fixed_positions.yaml
    SCALE_REPORT=experiments/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_scaling_16k.py --config "$SCALE_CONFIG"
  elif [[ "$MODE" == "asd-scaling-16k" ]]; then
    SCALE_CONFIG=experiments/config/asd_scaling_16k.yaml
    SCALE_REPORT=experiments/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_scaling_16k.py
  elif [[ "$MODE" == "asd-scaling-8k" ]]; then
    SCALE_CONFIG=experiments/config/asd_scaling_8k.yaml
    SCALE_REPORT=experiments/scripts/report_asd_scaling_8k.py
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_scaling_8k.py
  else
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_scaling.py
  fi
  export MANGO_ASD_SCALING_CONFIG="$SCALE_CONFIG"
  mapfile -t SCALE_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(*c["runs"], sep="\n")' "$SCALE_CONFIG")
  SCALE_ROOT="${SCALE_SETTINGS[0]}"
  SCALE_SNAKE=("$DRIVER/bin/snakemake" -s experiments/workflow/asd_scaling.smk \
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
    'import yaml; c=yaml.safe_load(open("experiments/config/antigen_diagnostic_training.yaml")); print(c["experiment_root"]); print(*c["arms"], sep="\n")')
  DIAGNOSTIC_ROOT="${DIAGNOSTIC_SETTINGS[0]}"
  DIAGNOSTIC_SNAKE=("$DRIVER/bin/snakemake" -s experiments/workflow/antigen_diagnostic_training.smk \
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
    'import yaml; c=yaml.safe_load(open("experiments/config/antigen_diagnostics.yaml")); print(c["experiment_root"]); print(*c["probes"], sep="\n")')
  DIAGNOSTIC_ROOT="${DIAGNOSTIC_SETTINGS[0]}"
  DIAGNOSTIC_SNAKE=("$DRIVER/bin/snakemake" -s experiments/workflow/antigen_diagnostics.smk \
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
    'import yaml; c=yaml.safe_load(open("experiments/config/asd_prototype_iglm.yaml")); print(c["experiment_root"]); print(c["mango_root"]); print(c["cohort_dir"])')
  IGLM_ROOT="${IGLM_SETTINGS[0]}"
  IGLM_MANGO="${IGLM_SETTINGS[1]}"
  IGLM_COHORT="${IGLM_SETTINGS[2]}"
  IGLM_SNAKE=("$DRIVER/bin/snakemake" -s experiments/workflow/asd_prototype_iglm.smk \
    --profile workflow/profiles/gpu --rerun-incomplete)
  IGLM_REPORT=("$DRIVER/bin/python" experiments/scripts/report_prototype_iglm.py \
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
  LOSS_CONFIG="experiments/config/${MODE//-/_}.yaml"
  mapfile -t LOSS_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(c["cohort_dir"]); print(*[f"{loss}/{arch}" for loss in c["objectives"] for arch in c["architectures"]], sep="\n")' "$LOSS_CONFIG")
  LOSS_ROOT="${LOSS_SETTINGS[0]}"
  LOSS_COHORT="${LOSS_SETTINGS[1]}"
  LOSS_ARMS=("${LOSS_SETTINGS[@]:2}")
  mkdir -p "$LOSS_ROOT"
  if [[ "$MODE" == "asd-prototype" ]]; then
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_prototype.py --config "$LOSS_CONFIG"
  else
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_transfer.py --output "$LOSS_COHORT"
  fi
  LOSS_SNAKE=("$DRIVER/bin/snakemake" -s experiments/workflow/asd_loss_comparison.smk \
    --profile workflow/profiles/gpu --configfile "$LOSS_CONFIG" --rerun-incomplete)
  for arm in "${LOSS_ARMS[@]}"; do
    if ! "${LOSS_SNAKE[@]}" -- "$LOSS_ROOT/sabdab2_scratch/$arm/eval.json"; then
      echo "[sabdab2_scratch/$arm] incomplete; continuing with other models." >&2
    fi
    "$DRIVER/bin/python" experiments/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT"
  done
  for arm in "${LOSS_ARMS[@]}"; do
    for phase in sabdab2_finetuned asd; do
      if ! "${LOSS_SNAKE[@]}" -- "$LOSS_ROOT/$phase/$arm/eval.json"; then
        echo "[$phase/$arm] incomplete; continuing with other models." >&2
      fi
      "$DRIVER/bin/python" experiments/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT"
    done
  done
  if ! "${LOSS_SNAKE[@]}" --keep-going; then
    echo "Some loss-comparison outputs remain incomplete; writing the partial report." >&2
  fi
  "$DRIVER/bin/python" experiments/scripts/report_asd_loss_comparison.py --root "$LOSS_ROOT" --require-complete
  exit 0
fi

if [[ "$MODE" == "abrank-transfer" || "$MODE" == "asd-transfer" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/artifacts/cache/matplotlib"
  TRANSFER_DATASET="${MODE%-transfer}"
  mapfile -t TRANSFER_SETTINGS < <("$DRIVER/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["experiment_root"]); print(c["cohort_dir"]); print(*c["controls"], sep="\n")' \
    "experiments/config/${TRANSFER_DATASET}_transfer.yaml")
  TRANSFER_ROOT="${TRANSFER_SETTINGS[0]}"
  TRANSFER_COHORT="${TRANSFER_SETTINGS[1]}"
  TRANSFER_TAGS=("${TRANSFER_SETTINGS[@]:2}")
  if [[ "$TRANSFER_DATASET" == "asd" ]]; then
    "$DRIVER/bin/python" experiments/scripts/prepare_asd_transfer.py --output "$TRANSFER_COHORT"
  fi
  TRANSFER_SNAKE=("$DRIVER/bin/snakemake" -s "experiments/workflow/${TRANSFER_DATASET}_transfer.smk" \
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
  "$DRIVER/bin/python" experiments/scripts/report_abrank_transfer.py --root "$TRANSFER_ROOT" --require-complete
  exit 0
fi
