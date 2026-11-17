#!/usr/bin/env bash
# Standard-library downloader; does not modify model or analysis environments.
set -euo pipefail
OPENALPHASEQ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${OPENALPHASEQ_DOWNLOAD_PYTHON:-python3}" \
    "$OPENALPHASEQ_REPO_ROOT/experiments/scripts/download_openalphaseq.py" "$@"
