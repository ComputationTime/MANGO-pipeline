#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
