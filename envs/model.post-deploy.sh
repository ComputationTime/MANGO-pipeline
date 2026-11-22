#!/usr/bin/env bash
set -euo pipefail

# The Conda CUDA 12.1 build is retained for dependency solving, then replaced
# with the first stable wheel carrying Blackwell (sm_120) CUDA 12.8 kernels.
python -m pip install --no-cache-dir --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
