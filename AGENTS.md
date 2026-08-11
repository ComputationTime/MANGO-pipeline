# MANGO agent handoff

This repository is the active MANGO project. Treat this file as persistent
context for coding agents working on a single NVIDIA GPU cloud instance.

## Immediate objective

Get the supported workflow running end to end on a very small SAbDab2 subset,
then run the full configured study. The model task is always:

> Given the antigen and light chain, generate the heavy chain.

The heavy chain must never enter the conditioning input. Training, evaluation,
held-out reconstruction, and generation all use the same antigen + masked-heavy
AbLang2 light-chain context contract.

Start from the repository root with:

```bash
./run_gpu.sh smoke
```

Only after smoke succeeds (or its partial report has been understood and fixed)
run:

```bash
./run_gpu.sh study
```

Read `docs/cloud_gpu_setup.md` before changing installation or execution logic.
The launcher is the supported cloud interface; `mango/MANGORunner.py` is legacy.

## Current supported scope

The default GPU modes run these antigen representations, sequentially on one
GPU: `one_hot`, `biopython`, `esm2`, `esmif`, and `proteinmpnn`.

- ESM3 is opt-in with `smoke-esm3` / `study-esm3` and requires `HF_TOKEN` after
  accepting the Hugging Face model terms.
- PyRosetta PRE is opt-in with `smoke-pyrosetta` / `study-pyrosetta`; its real
  mmCIF chain mapping still needs validation before scientific use.
- `*-all` enables both optional methods.
- AF-M, AF3, Boltz, Chai, Figure 2, TAP, therapeutic benchmark antibodies, and
  the eight therapeutic complex chains are deferred. Do not add them to the
  completion target while debugging the current milestone.
- Figures 1, 3, 4, and 5 are in scope. Figure 1 is train/test NLL. Figure 4 uses
  ANARCI germline assignment and reports distance to the nearest heavy-chain
  germline. TAP remains skipped.

## Important invariants

- Validation splitting is group-aware on `ab_ag_cluster`. Never replace it with
  row-wise random splitting, and do not allow a cluster to cross splits.
- Preserve the nested MANGO Git repository; do not flatten or remove `.git`.
- Keep generated data, environments, caches, weights, and results under
  `.snakemake/` or `artifacts/`; do not commit them.
- Do not put Hugging Face tokens, credentials, or licence material in the repo,
  YAML files, logs, or commands that will be committed.
- Public model weights should download automatically. Avoid introducing manual
  download steps unless an upstream licence or access gate makes that impossible.
- Keep per-rule Conda environments isolated. The embedders have incompatible
  dependency sets; installing everything into one global environment is not a
  valid fix.
- GPU-heavy rules share `gpu=1` and must remain serialized on a single card.
- Smoke metrics are plumbing checks and are not scientifically interpretable.

## Failure and resume behavior

`run_gpu.sh` deliberately launches every embedder in a separate Snakemake
invocation. A failed Conda environment, authentication check, weight download,
embedder, training run, or analysis branch must not prevent later embedders from
being attempted. Do not collapse these back into one combined pre-created DAG.

The launcher returns nonzero if anything remains incomplete, but first writes:

```text
artifacts/gpu/results_report.csv
artifacts/gpu/results_report.json
```

The CSV retains available split NLL, perplexity, example/token counts, and
prediction counts. The JSON distinguishes fully complete runs,
core-complete/analysis-incomplete runs, and failed runs, with the first missing
stage and expected log paths. Cross-embedder figures and
`artifacts/gpu/results_complete.json` are strict outputs and are only trustworthy
when all requested methods finish.

Every trained run also produces `training_curve.csv` and
`training_curve.png`. The CSV is flushed at every training iteration; the PNG
is atomically refreshed at the configured interval and after epoch-end
validation, so it is safe to view while training continues.

Rerun the identical launcher command after fixing a problem. Snakemake and the
batch embedding scripts reuse valid downloads and completed per-record outputs.

## Cloud-agent operating procedure

1. Verify `nvidia-smi`, free disk, system RAM, Conda, Git, and outbound HTTPS.
2. Run `./run_gpu.sh smoke` without manually preinstalling CUDA or model stacks.
3. If it fails, allow the launcher to finish attempting every embedder. Inspect
   `artifacts/gpu/results_report.json`, then the referenced files under
   `artifacts/logs/` and `.snakemake/log/`.
4. Diagnose the smallest failing layer: host/GPU, Conda solve, gated access,
   weight verification, embedding contract, training, prediction, or analysis.
5. Make a scoped, reproducible fix in workflow code or an environment YAML.
   Do not bypass validation, replace a pretrained model with random weights, or
   silently fall back to CPU when `MANGO_REQUIRE_CUDA=1`.
6. Rerun `./run_gpu.sh smoke`; report which embedders succeeded and their NLL and
   perplexity from the generated report.
7. Do not launch `study` until the smoke outcome and any remaining optional
   failures have been clearly reported to the user.

When reporting a failure, include the exact embedder, failed stage, relevant log
path, root cause, change made, and rerun result. Continue independently runnable
work instead of stopping at the first optional-method failure.

## Useful validation commands

```bash
bash -n run_gpu.sh
python -m unittest discover -s tests -v
snakemake --lint
snakemake --sdm conda --cores 1 --dry-run
```

Tests may need to run inside their Snakemake Conda environment because the host
Python intentionally does not contain pandas, PyTorch, or analysis packages.
Do not treat missing host-Python packages as evidence that the isolated workflow
environment is broken.
