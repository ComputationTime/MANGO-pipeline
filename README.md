# MANGO

MANGO is a Snakemake workflow that trains an antigen-conditioned antibody
language model. Its single task:

> Given an antigen and a requested chain type, generate that antibody chain.

No antibody sequence or antibody-derived embedding is ever a conditioning input.
Training, evaluation, held-out reconstruction and de novo generation all use the
same antigen-only input plus a learned heavy/light output selector.

The study trains one model per antigen representation (one-hot, BioPython,
PyRosetta PRE, ESM2, ESM3, ESM3+structure, ESM-IF, ProteinMPNN), identical in
every other respect, then compares them on held-out antigens.

```text
fetch → standardize → process → embed → train → evaluate
                                      └──────→ predict
                                      └──────→ generate → cohort → metrics → figures
```

## Quick start (one NVIDIA GPU, Linux)

Requirements: `nvidia-smi` with driver ≥ 525.60, Conda/Miniforge, Git, Bash,
≥ 64 GiB RAM, ≥ 150 GiB free disk, outbound HTTPS on the first run. No system
CUDA toolkit is needed; the per-rule Conda environments carry it. On a 12 GB
card add `MANGO_EXECUTION_CONFIG=config/local_12gb.yaml`.

```bash
./run_gpu.sh smoke        # 5-record plumbing check, five non-gated embedders
./run_gpu.sh smoke-all    # adds gated ESM3 and PyRosetta PRE
./run_gpu.sh study        # full eight-embedder study
./run_gpu.sh --help       # every mode
```

Smoke metrics only prove the plumbing; they are not scientifically meaningful.
Run `smoke` and `smoke-all` and look at their report before launching `study`.

Gated/licensed pieces: ESM3 needs accepted Hugging Face terms and `HF_TOKEN`
(export it; never write it into the repo). PyRosetta needs an eligible academic
or commercial licence. Check the licences of Boltz-2, Chai-1 and the other
upstream models before redistributing results.

## What you get

Every invocation writes `artifacts/gpu/results_report.{json,csv}` (NLL,
perplexity, counts, and the first failed stage per embedder). Each embedder runs
in its own Snakemake invocation, so one failure never blocks the others; the
launcher exits nonzero if anything is incomplete. Re-run the same command to
resume. Figures land in `artifacts/analysis/figures/`; trained runs in
`artifacts/runs/<embedder>__<hash>/` (with `training_curve.csv/png`).

See [docs/reproducing_the_paper.md](docs/reproducing_the_paper.md) for the
figure-by-figure map.

## Repository layout

```text
run_gpu.sh              supported entry point (smoke / small / study / ablations)
config/                 config.yaml + mode overlays (smoke, small, gpu*, local_12gb)
workflow/Snakefile      workflow entry; rules/ stages, scripts/ implementations, envs/ per-rule Conda
mango/                  model components used by the workflow
tests/                  fast contract tests
docs/                   setup, config reference, reproduction map, troubleshooting
experiments/            UNSUPPORTED side studies (ASD, AbRank, OpenAlphaSeq, affinity, ...)
_to_review/             quarantined legacy files awaiting deletion (see its README)
artifacts/              all generated data/models/results (untracked)
```

## Documentation

- [docs/cloud_gpu_setup.md](docs/cloud_gpu_setup.md): fresh-instance walkthrough and weight sources
- [docs/reproducing_the_paper.md](docs/reproducing_the_paper.md): figures, commands, outputs
- [docs/config_reference.md](docs/config_reference.md): every section of `config/config.yaml`
- [docs/troubleshooting.md](docs/troubleshooting.md): failures and fixes
- [docs/pipeline_reference.md](docs/pipeline_reference.md): detailed module-level reference
- [AGENTS.md](AGENTS.md): invariants and procedure for coding agents

## Development checks

```bash
scripts/check.sh   # bash -n, snakemake lint + dry-run, unit tests (see script for env notes)
```

Deferred: AF-Multimer, AF3, TAP and the therapeutic benchmark complexes.
