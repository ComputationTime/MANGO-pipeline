#!/usr/bin/env bash
# Fast, CPU-only repository checks. Run from anywhere.
#   scripts/check.sh            launcher syntax, snakemake dry-run, unit tests
# Env overrides:
#   SNAKEMAKE=<path>   snakemake binary (default: .snakemake/gpu-driver/bin/snakemake, else PATH)
#   TEST_PYTHON=<path> python with torch/pandas/numpy/yaml for the tests (default: python3)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0
step() { echo "== $*"; }

step "launcher syntax"
bash -n run_gpu.sh && bash -n experiments/run_experiment.sh || fail=1

SM="${SNAKEMAKE:-}"
[[ -z "$SM" && -x .snakemake/gpu-driver/bin/snakemake ]] && SM=.snakemake/gpu-driver/bin/snakemake
[[ -z "$SM" ]] && SM="$(command -v snakemake || true)"
if [[ -n "$SM" ]]; then
  step "snakemake dry-run (core DAG)"
  "$SM" --sdm conda --cores 1 --dry-run >/dev/null 2>&1 || { echo "dry-run failed"; fail=1; }
else
  echo "snakemake not found; skipping dry-run (run ./run_gpu.sh smoke once to create the driver)"
fi

step "unit tests (tests/)"
PY="${TEST_PYTHON:-python3}"
"$PY" -m unittest discover -s tests || { echo "tests failed (missing torch/pandas? set TEST_PYTHON to a workflow env python)"; fail=1; }
exit $fail
