#!/usr/bin/env bash
# New launcher: leave run_gpu.sh and every previous experiment unchanged.
set -euo pipefail
ATTENTION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ATTENTION_REPO"
MODE="${1:-smoke}"
case "$MODE" in smoke|study) ;; *) echo 'usage: ./experiments/run_interface_attention.sh {smoke|study}' >&2; exit 2 ;; esac
nvidia-smi >/dev/null
ATTENTION_CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$ATTENTION_CONDA" && -x artifacts/miniforge3/bin/conda ]]; then
  ATTENTION_CONDA="$ATTENTION_REPO/artifacts/miniforge3/bin/conda"
fi
if [[ -z "$ATTENTION_CONDA" ]]; then echo 'Conda is required.' >&2; exit 1; fi
export PATH="$(dirname "$ATTENTION_CONDA"):$PATH"
export MANGO_REQUIRE_CUDA=1 PYTHONNOUSERSITE=1 CONDA_CHANNEL_PRIORITY=strict
export XDG_CACHE_HOME="$ATTENTION_REPO/artifacts/cache"
export MPLCONFIGDIR="$ATTENTION_REPO/artifacts/cache/matplotlib"
export CONDA_PKGS_DIRS="$ATTENTION_REPO/.snakemake/conda-pkgs"
ATTENTION_DRIVER="$ATTENTION_REPO/.snakemake/gpu-driver/bin"
if [[ ! -x "$ATTENTION_DRIVER/snakemake" ]]; then
  echo 'Run the supported ./run_gpu.sh smoke setup first to create the isolated workflow driver.' >&2; exit 1
fi
if [[ "$MODE" == smoke ]]; then
  "$ATTENTION_DRIVER/python" - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('experiments/config/interface_attention.yaml').read_text())
root=Path('artifacts/experiments/interface_attention_smoke_v1')
root.mkdir(parents=True,exist_ok=True)
cfg.update(experiment_root=str(root),smoke=True,seeds=[400],pretraining_updates=2,adaptation_updates=2,bootstrap_replicates=100)
text=yaml.safe_dump(cfg)
path=root/'config.yaml'
if not path.exists() or path.read_text()!=text:path.write_text(text)
PY
  export MANGO_ATTENTION_CONFIG=artifacts/experiments/interface_attention_smoke_v1/config.yaml
fi
exec "$ATTENTION_DRIVER/snakemake" -s experiments/workflow/interface_attention.smk --profile workflow/profiles/gpu --rerun-incomplete
