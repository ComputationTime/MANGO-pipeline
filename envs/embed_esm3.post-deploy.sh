#!/usr/bin/env bash
set -euo pipefail

# ESM 3's published PyTorch 2.3/CUDA 12.1 environment has no Blackwell
# (sm_120) kernels. Preserve the isolated ESM dependency solve, then replace
# PyTorch with the first stable CUDA 12.8 build that supports this GPU family.
python -m pip install --no-cache-dir --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
