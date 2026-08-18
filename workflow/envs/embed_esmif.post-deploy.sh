#!/usr/bin/env bash
set -euo pipefail

# The published ESM-IF stack pins CUDA 12.1 binaries without Blackwell
# (sm_120) kernels. Replace PyTorch and its one compiled PyG dependency with
# matching CUDA 12.8 wheels while keeping fair-esm's legacy Python API intact.
python -m pip install --no-cache-dir --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
  torch-scatter==2.1.2
