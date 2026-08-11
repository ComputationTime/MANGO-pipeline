#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint. The supported installer/runner now creates pinned
# per-rule environments and fetches every active model through Snakemake.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "MANGO setup is managed by run_gpu.sh; starting ${1:-smoke}."
exec "$ROOT_DIR/run_gpu.sh" "${1:-smoke}"
