#!/usr/bin/env bash
# ./run_gpu.sh doctor: read-only check of this machine against what MANGO needs.
# Prints PASS / WARN / FAIL per item; exits 1 if any FAIL. Never prints secrets.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
fails=0; warns=0
pass() { printf '  PASS  %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; warns=$((warns+1)); }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails+1)); }

echo "MANGO doctor ($ROOT_DIR)"

# Required GPU memory: config/gpu.yaml, overridden by MANGO_EXECUTION_CONFIG.
need_gb=20
for f in config/config.yaml config/gpu.yaml "${MANGO_EXECUTION_CONFIG:-}"; do
  [[ -n "$f" && -f "$f" ]] || continue
  v=$(grep -E '^\s*min_gpu_memory_gb:' "$f" | head -1 | sed -E 's/.*:\s*([0-9.]+).*/\1/')
  [[ -n "$v" ]] && need_gb="$v"
done

echo "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found: install an NVIDIA driver (>= 525.60.13)"
elif ! info=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1); then
  fail "nvidia-smi cannot talk to the driver: $info"
else
  while IFS=, read -r name mem drv; do
    name=$(echo "$name" | xargs); mem=$(echo "$mem" | xargs); drv=$(echo "$drv" | xargs)
    gb=$(( mem / 1024 ))
    pass "$name, ${gb} GiB, driver $drv"
    if (( gb < need_gb )); then
      fail "GPU has ${gb} GiB but config requires min_gpu_memory_gb=${need_gb} (12 GB cards: MANGO_EXECUTION_CONFIG=config/local_12gb.yaml)"
    fi
    if [[ "$(printf '%s\n525.60.13\n' "$drv" | sort -V | head -1)" != "525.60.13" ]]; then
      fail "driver $drv is older than 525.60.13 (CUDA 12.1 environments)"
    fi
  done <<< "$info"
fi

echo "Host"
free_gb=$(df -Pk . | awk 'NR==2{printf "%d", $4/1048576}')
if (( free_gb >= 150 )); then pass "free disk ${free_gb} GiB"
elif (( free_gb >= 60 )); then warn "free disk ${free_gb} GiB (150 GiB recommended for the full study)"
else fail "free disk ${free_gb} GiB (smoke needs several tens of GiB, study ~150 GiB)"; fi
ram_gb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
if (( ram_gb >= 64 )); then pass "system RAM ${ram_gb} GiB"
else warn "system RAM ${ram_gb} GiB (64 GiB recommended)"; fi

echo "Tools"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -z "$CONDA_BIN" && -x /workspace/miniforge3/bin/conda ]] && CONDA_BIN=/workspace/miniforge3/bin/conda
[[ -z "$CONDA_BIN" && -x "$ROOT_DIR/artifacts/miniforge3/bin/conda" ]] && CONDA_BIN="$ROOT_DIR/artifacts/miniforge3/bin/conda"
if [[ -n "$CONDA_BIN" ]]; then pass "conda: $CONDA_BIN"; else fail "conda/Miniforge not found"; fi
command -v git >/dev/null 2>&1 && pass "git $(git --version | awk '{print $3}')" || fail "git not found"
if [[ -x .snakemake/gpu-driver/bin/snakemake ]]; then pass "Snakemake driver env present"
else warn "Snakemake driver env not created yet (run_gpu.sh creates it on first use)"; fi

echo "Network (HTTPS, 8 s timeout)"
if command -v curl >/dev/null 2>&1; then
  for url in https://conda.anaconda.org https://huggingface.co https://pypi.org; do
    if curl -fsS -o /dev/null -I --max-time 8 "$url" 2>/dev/null; then pass "$url"
    else fail "cannot reach $url"; fi
  done
else
  warn "curl not found; skipping network checks"
fi

echo "Gated models"
if [[ -n "${HF_TOKEN:-}" ]]; then pass "HF_TOKEN is set (needed for uncached ESM3 weights)"
else warn "HF_TOKEN is not set: ESM3 will fail, other embedders still run (accept the ESM3 terms first)"; fi
warn "PyRosetta requires an eligible licence; not checked here"

echo
if (( fails )); then echo "doctor: $fails FAIL, $warns WARN. Fix the FAIL items before running."; exit 1; fi
echo "doctor: OK ($warns warning(s))."
