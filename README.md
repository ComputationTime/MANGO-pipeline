# MANGO

MANGO is a Snakemake workflow for training an antigen-conditioned antibody
language model. Its single task is:

> Given an antigen and a light chain, generate the heavy chain.

The heavy chain is never part of the conditioning input. The antibody context
embedding is produced by AbLang2 from `*|L`, where the heavy-chain slot is
masked. Training, evaluation, held-out reconstruction, and de novo generation
all use that same input contract.

The Snakemake workflow is the canonical interface. The older
`mango/MANGORunner.py` API is legacy code and is not the supported execution
path for the current model.

## Current milestone

The repository default remains the one-hot baseline. The single-GPU study
profile expands that same DAG to every currently runnable, non-gated antigen
representation: one-hot, BioPython, ESM2, ESM-IF, and ProteinMPNN.

```text
fetch → standardize → process → embed → train → evaluate
                                      └──────→ predict
                                      └──────→ generate (explicit target)
                                                  └→ cohort → metrics → figures
```

- `process` creates a validation set by holding out complete
  `ab_ag_cluster` groups. Row-wise random validation is deliberately rejected.
- `embed` builds the selected antigen embeddings and masked-heavy/light-chain
  AbLang2 context embeddings. AbLang2 weights are downloaded once to
  `artifacts/weights/ABLANG-ablang2-paired`, never by individual embed jobs.
- `train` optimizes only heavy-chain tokens.
- `evaluate` reports heavy-chain NLL and perplexity on train, validation, and
  test splits.
- `predict` generates one heavy-chain reconstruction for every configured
  held-out record.
- `generate` produces the configured number of de novo heavy chains for a
  bounded set of held-out records.
- `analysis` is explicit and produces handbook Figures 1 through 5. Figure 2
  runs Boltz-2 and Chai-1 on a bounded, cluster-diverse generated subset. TAP
  remains deferred.

ESM3 is an opt-in sixth representation because its weights require accepting
the Hugging Face terms and supplying a token. PyRosetta PRE is a seventh,
academic/non-commercial opt-in and still needs a real mmCIF smoke validation.
AF-M is a stub. Therapeutic targets, AF3, and TAP remain deferred.
Boltz-2/Chai-1 structure confidence has passed a two-sequence GPU plumbing
check and now runs by default in smoke/small/study with mode-specific bounds.

For a beginner-oriented, linked explanation of the hypothesis, architecture,
terminology, and every Snakemake module, open
[`docs/site/index.html`](docs/site/index.html) in a web browser.

## Repository layout

```text
config/config.yaml             experiment configuration
workflow/Snakefile             canonical workflow entrypoint
workflow/rules/                stage definitions
workflow/scripts/              executable stage implementations
workflow/envs/                 isolated rule environments
mango/utils/                   model components used by the workflow
tests/                         fast contract tests
docs/site/index.html           linked beginner-oriented project guide
artifacts/                     generated data and models; ignored by Git
```

All generated files live under `artifacts/`. A trained run is identified by an
embedder tag and a hash of the dataset, processing, embedding, and model
configuration:

```text
artifacts/runs/one_hot__<hash>/
├── config.json
├── model_config.json
├── metrics.jsonl
├── training_curve.csv
├── training_curve.png
├── checkpoints/best.pt
├── checkpoints/latest.pt
├── eval.json
└── predictions_test.csv
```

## Environment

Create the workflow driver environment:

```bash
conda env create -f environment.yaml
conda activate mango
```

Per-rule environments under `workflow/envs/` are the recommended execution
mode because model dependencies stay isolated:

```bash
snakemake --sdm conda --cores 8 --dry-run
```

## One NVIDIA GPU cloud run

For a new-instance walkthrough, including every automatic weight source and
troubleshooting step, read [docs/cloud_gpu_setup.md](docs/cloud_gpu_setup.md).

An RTX A6000 is a good fit: the workflow requires at least 20 GiB and the card
has 48 GiB. Provision a Linux instance with:

- one NVIDIA GPU and a working `nvidia-smi`;
- NVIDIA driver 525.60.13 or newer (the rule environments use CUDA 12.1);
- Conda or Miniforge, Git, and Bash;
- at least 64 GiB system RAM and preferably 150 GiB free disk;
- outbound HTTPS access on the first run for Conda packages and model weights.

The host does not need a separately installed CUDA toolkit: the isolated Conda
environments carry the CUDA runtime. From the repository root, first run the
five-record plumbing check:

```bash
./run_gpu.sh smoke
```

Then launch the full filtered SAbDab2 study with one command:

```bash
./run_gpu.sh study
```

The launcher creates its own pinned Snakemake driver, checks CUDA with a real
matrix multiplication, creates isolated CUDA environments, downloads and
verifies all active embedder weights, downloads/reuses SAbDab2, embeds, trains one model per representation,
evaluates, reconstructs held-out heavy chains, generates the configured design
cohort, and writes Figures 1, 3, 4, and 5. GPU jobs request a single shared
`gpu=1` resource, so the representations run sequentially. In GPU mode each
embedder is one restartable batch and loads its pretrained model once rather
than once per structure.

Each run also writes `training_curve.csv` at every training iteration and
atomically refreshes `training_curve.png` every 250 iterations and after each
validation pass. The x-axis is the global training iteration. Training loss is
shown per iteration (plus a rolling mean), while validation loss is evaluated
once per epoch and plotted at that epoch's final iteration.

Every invocation writes a report and a flat numbers table:

```text
artifacts/gpu/results_report.json
artifacts/gpu/results_report.csv
```

The launcher runs each embedder in a separate Snakemake invocation, including
its Conda environment creation. A failed package install, weight download, or
scientific job therefore cannot prevent later embedders from being attempted.
The report lists the failed method and stage while retaining split NLL,
perplexity, example/token counts, and prediction counts for every core run that
completed. It separately flags a core-complete run whose analysis metrics are
incomplete. The launcher returns a nonzero exit status for automation, but
writes the partial report first. Cross-embedder figures may remain unavailable
when one required branch fails.

Strict successful completion is additionally recorded at:

```text
artifacts/gpu/results_complete.json
```

The completion manifest lists every run, evaluation, prediction table, figure,
and the GPU/driver preflight report. Snakemake keeps successful outputs, so rerunning
the same command resumes missing work. Batch embedding also reuses valid
per-record outputs after an interrupted batch.

Structure confidence from Boltz-2 and Chai-1 runs automatically in
`run_gpu.sh smoke`, `small`, and `study`. The default study folds three designs
from each of ten independently held-out antigen clusters per model and
predictor; smoke folds one design from one target. For a dedicated rerun with
five diffusion samples per design, run:

```bash
.snakemake/gpu-driver/bin/snakemake -s workflow/Snakefile \
  --profile workflow/profiles/gpu \
  --configfile config/gpu.yaml config/structure.yaml structure_confidence
```

This folds the configured deterministic design subset as heavy + cognate light +
antigen complexes, writes normalized confidence tables, and renders Figure 2.
AF3 is deferred to cluster-side execution and later ingestion through the same
table contract. Boltz and Chai use single-sequence mode unless
`structure_prediction.use_msa_server` is explicitly enabled.

To include ESM3, first accept access to `esm3-sm-open-v1`, then export the
standard Hugging Face token and choose the corresponding mode:

```bash
export HF_TOKEN=...
./run_gpu.sh smoke-esm3
./run_gpu.sh study-esm3
```

Do not start with `study-esm3`: validate the token, environment, and model
download with `smoke-esm3` first.

PyRosetta and combined modes follow the same pattern:

```bash
./run_gpu.sh smoke-pyrosetta   # then study-pyrosetta
./run_gpu.sh smoke-all         # ESM3 + PyRosetta; then study-all
```

To check only environments, connectivity, authorization, and weights:

```bash
./run_gpu.sh weights           # or weights-esm3 / weights-pyrosetta / weights-all
```

The `fetch` dependency downloads and extracts SAbDab2 automatically from the
configured Zenodo record. Zenodo distributes one archive (about 876 MB
compressed), so even a small processed subset requires the complete archive
once; subsequent runs reuse `artifacts/data/sabdab2_v0.1.0/`.

## Running the workflow

Run from the repository root. Snakemake automatically discovers
`workflow/Snakefile`.

```bash
# Individual checkpoints
snakemake --sdm conda --cores 8 fetch
snakemake --sdm conda --cores 8 weights
snakemake --sdm conda --cores 8 process
snakemake --sdm conda --cores 8 embed
snakemake --sdm conda --cores 8 train

# Train if needed, then evaluate and reconstruct held-out heavy chains
snakemake --sdm conda --cores 8 inference

# Equivalent default invocation
snakemake --sdm conda --cores 8

# Potentially large; intentionally not part of the default target
snakemake --sdm conda --cores 8 generate

# Generate if needed, score a deterministic cohort, and render figures 1/3/4/5
snakemake --sdm conda --cores 8 analysis

# Five-row, cluster-separated, one-epoch end-to-end smoke run
snakemake --configfile config/smoke.yaml --sdm conda --cores 4 analysis
```

The main settings are in `config/config.yaml`. In particular:

- `active_embedders` is restricted to `one_hot` in the repository default;
  `config/gpu.yaml` selects the five supported cloud representations.
- `processing.val.cluster_column` identifies the leakage boundary;
  `processing.val.fraction` is the fraction of training clusters held out.
- `model.*` controls training and early stopping.
- `model.predict_splits` controls held-out reconstruction.
- `generation.target_selection` chooses one representative from every held-out
  test cluster by default; `generation.max_targets` optionally caps that broad
  cohort and `generation.n_per_target` bounds sampling per target.
- `analysis.cohort.n_per_target` bounds the common sequence-scoring cohort.
- `analysis.ab_likeness`, `analysis.germline`, and
  `analysis.developability` configure the independent metric modules.

Analysis outputs are separated by contract:

```text
artifacts/analysis/<run>/cohort.csv
artifacts/analysis/<run>/metrics/{iglm,antiberty,ablang2,germline,biophysical}.csv
artifacts/analysis/figures/fig{1,3,4,5}_*.{png,csv}
```

Figure 4 uses the pinned Bioconda ANARCI database to assign the nearest heavy
V and J germlines. It records the gene calls, reconstructed V/J reference,
raw Levenshtein distance, and length-normalized distance for every design.

## Tests

Run the lightweight contract tests inside an environment containing pandas:

```bash
python -m unittest discover -s tests -v
```

Before a full run, also validate the workflow itself:

```bash
snakemake --lint
snakemake --sdm conda --cores 1 --dry-run
```

## Deferred work

AlphaFold-Multimer, AF3, therapeutic targets, and TAP are not included by the
GPU completion target. Boltz-2 and Chai-1 confidence and Figure 2 are included
on the bounded default subset. PyRosetta PRE and ESM3 are
available only through explicit opt-in modes. ESM3 and ESM-IF
were adapted from the contributed `mango-embedders` module. ESM3 remains
sequence-only and gated; ESM-IF encodes antigen chains independently rather
than as a joint multichain complex.

The implemented Boltz/Chai contract and intended AF3 ingestion are documented in
`docs/structure_prediction_plan.md`.
