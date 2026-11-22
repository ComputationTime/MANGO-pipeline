#!/usr/bin/env bash
set -euo pipefail

# chai_lab 0.6.1 pins a CUDA 12.4 PyTorch build that has no Blackwell
# (sm_120) kernels. Keep Chai's published dependency solve intact, then replace
# only torch with the first stable CUDA 12.8 build. This combination is covered
# by the structure-confidence GPU plumbing test.
python -m pip install --no-cache-dir --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
