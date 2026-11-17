#!/usr/bin/env bash
# Bootstrap the small downloader environment without changing model environments.
set -euo pipefail

ASD_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASD_ENV_DIR="$ASD_REPO_ROOT/artifacts/envs/asd_download"
ASD_PYTHON="$ASD_ENV_DIR/bin/python"

if [[ ! -x "$ASD_PYTHON" ]]; then
    "${ASD_DOWNLOAD_PYTHON:-python3}" -m venv "$ASD_ENV_DIR"
fi
if ! "$ASD_PYTHON" -c 'import importlib.metadata; assert importlib.metadata.version("gdown") == "6.4.2"' 2>/dev/null; then
    PIP_CACHE_DIR="$ASD_REPO_ROOT/artifacts/cache/pip" \
        "$ASD_PYTHON" -m pip install --disable-pip-version-check 'gdown==6.4.2'
fi

exec "$ASD_PYTHON" "$ASD_REPO_ROOT/workflow/scripts/download_asd.py" "$@"
