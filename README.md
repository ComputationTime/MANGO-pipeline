# MANGO

MANGO is a Snakemake workflow for training an antigen-conditioned antibody
language model. Its single task is:

> Given an antigen and a requested chain type, generate that antibody chain.

No antibody sequence or antibody-derived embedding is supplied as conditioning.
Training, evaluation, held-out reconstruction, and de novo generation all use
the same antigen-only biological input plus a learned heavy/light output
selector. Heavy-chain generation remains the downstream-analysis default.

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
- `embed` builds only the selected antigen representations. AbLang2 remains an
  independent generated-sequence scorer, not a MANGO input embedder.
- `train` treats each configured heavy or light target as a separate example.
- `evaluate` reports token-weighted NLL and perplexity overall and by chain type.
- `predict` reconstructs the chain selected by `generation.chain` for every
  configured held-out record.
- `generate` produces the configured number of de novo selected chains for a
  bounded set of held-out records; downstream design analysis requires heavy.
- `analysis` is explicit and produces handbook Figures 1 through 5. Figure 2
  runs Boltz-2 and Chai-1 on a bounded, cluster-diverse generated subset. TAP
  remains deferred.

The full `study` now includes ESM3 as the sixth representation and PyRosetta
PRE as the seventh. ESM3 requires accepted Hugging Face terms and `HF_TOKEN`;
PyRosetta requires academic/non-commercial eligibility, and its PRE results
must retain the mmCIF chain-mapping validation warning.
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

Then accept the ESM3 terms, set `HF_TOKEN`, and validate the two additional
study embeddings before launching the full filtered SAbDab2 study:

```bash
./run_gpu.sh smoke-all
./run_gpu.sh study
```

The launcher creates its own pinned Snakemake driver, checks CUDA with a real
matrix multiplication, creates isolated CUDA environments, downloads and
verifies all active embedder weights, downloads/reuses SAbDab2, embeds, trains one model per representation,
evaluates, reconstructs the configured held-out chain, generates the design
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

This folds the configured deterministic heavy-design subset with the recorded
reference light chain and antigen, writes normalized confidence tables, and
renders Figure 2. The reference partner is used only by this downstream quality-
control analysis; it is never a MANGO conditioning input.
AF3 is deferred to cluster-side execution and later ingestion through the same
table contract. Boltz and Chai use single-sequence mode unless
`structure_prediction.use_msa_server` is explicitly enabled.

Smoke and study now also apply compute-minimal Rosetta quality-control scoring
to those structures. To request only that analysis and its figure, run:

```bash
snakemake --sdm conda --cores 1 rosetta_interface
```

This CPU target applies fixed-backbone `InterfaceAnalyzer` scoring to
the highest-confidence Boltz and Chai structure for each generated design. It
does not relax, dock, or redesign the complexes. Its energies measure the raw
predicted coordinates and must not be interpreted as binding affinities.

To include ESM3, first accept access to `esm3-sm-open-v1`, then export the
standard Hugging Face token and choose the corresponding mode:

```bash
export HF_TOKEN=...
./run_gpu.sh smoke-esm3
./run_gpu.sh study-esm3
```

Do not start with `study-esm3`: validate the token, environment, and model
download with `smoke-esm3` first.

Because plain `study` now uses all seven embedders, validate both special
dependencies together before launching it:

```bash
./run_gpu.sh smoke-pyrosetta   # then study-pyrosetta
./run_gpu.sh smoke-all         # ESM3 + PyRosetta; then plain study
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
  the launcher combines `config/gpu.yaml` and `config/gpu_all.yaml` for the
  seven-embedder full study while ordinary smoke retains five embedders.
- `processing.val.cluster_column` identifies the leakage boundary;
  `processing.val.fraction` is the fraction of training clusters held out.
- `model.*` controls training and early stopping.
- `model.target_chains` controls which chain targets are learned; neither target
  sequence is used as conditioning.
- `model.predict_splits` controls held-out reconstruction.
- `generation.chain` selects heavy or light output; heavy is the analysis default.
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
on the bounded default subset. PyRosetta PRE and ESM3 participate in the full
seven-embedder study and also remain available through focused suffixed modes.
ESM3 and ESM-IF were adapted from the contributed `mango-embedders` module.
ESM3 remains sequence-only and gated; ESM-IF encodes antigen chains
independently rather than as a joint multichain complex.

The implemented Boltz/Chai contract and intended AF3 ingestion are documented in
`docs/structure_prediction_plan.md`.

## Antigen-only model contract

The default architecture is `antigen_chain_gpt2_v3`. Its prefix contains the
projected antigen representation, one learned heavy/light output selector, and
the beginning-of-sequence token. A causal GPT-2 decoder then emits the requested
chain. Only target residues and EOS receive loss; no antibody sequence,
antibody embedding or species label is accepted as conditioning.

The `antigen-controls` mode also trains `control_cross_attention_esm2`, using
architecture `antigen_chain_cross_attention_gpt2_v1`. It reuses the same ESM2
antigen tensors as separate cross-attention memory at every decoder layer and
keeps only the H/L selector and BOS in the prefix. It has extra attention
parameters and a distinct checkpoint identity. See
[the comparison controls](docs/comparison_controls.md) for the exact comparison.

Use `./run_gpu.sh antigen-length-ablation` to sweep constant, random and
token-merged ESM2 inputs over 1 token, ¼L, ½L, ¾L and L per record.
All use width 1280; fractional lengths
are rounded up with a one-token minimum. ESM2 merges similar frozen antigen
tokens with mass-weighted averages; its endpoints reuse mean-pooled ESM2 and
the original full-length ESM2 evaluation. It writes separate train/test NLL
panels to `artifacts/analysis/figures/fig9_antigen_context_length_ablation.png`,
with exact values and actual context-length statistics in the matching
`fig9_antigen_context_length_ablation_data.csv`. See
[the cloud setup guide](docs/cloud_gpu_setup.md) for the run and resume details.

Use `./run_gpu.sh iglm-baseline` to train the full IgLM architecture from random
initialization on the same SAbDab2 cohort and add a horizontal reference to
Figure 9. It uses the same optimizer, seed, target sequences, training order and
validation stopping rule as MANGO. IgLM receives native H/L tokens and a fixed
species start token, with residues plus EOS scored by the common token mean.
The baseline has no antigen context and therefore one reference value per split.
The full `antigen-length-ablation` mode also includes this baseline.

For the separately audited AbRank experiment, use `./run_gpu.sh abrank-transfer`
after preparing the CATH-protected cohorts. This trains eight controls on
AbRank, then fine-tunes matching checkpoints on SAbDab2. New plots and results
stay under `artifacts/experiments/abrank_cath_transfer_v1/`. See
[the cohort, contamination audit, and transfer protocol](docs/abrank_transfer.md).

The current pretraining test uses **ASD 2.0**: run `./run_gpu.sh asd-transfer`.
It includes all BUZZ affinity bins with positive binding evidence, protects the
same SAbDab2 validation/test cohorts, and writes separate results under
`artifacts/experiments/asd_cath_transfer_v1/`. See [the ASD transfer protocol](docs/asd_transfer.md).

Use `./run_gpu.sh asd-loss-comparison` for the eight ESM2 models comparing
standard/focal loss, normal/cross-attention MANGO, and SAbDab2-only/ASD-pretrained
training. All are evaluated with ordinary NLL on the same native cohort.
See [the loss comparison protocol](docs/asd_loss_comparison.md).

For rapid iteration, use `./run_gpu.sh asd-prototype`: a HER2-free ASD subset
with 2,000 training records and one epoch per stage for all eight final ESM2
models. See [the prototype protocol](docs/asd_prototype.md).

After the prototype finishes, `./run_gpu.sh asd-prototype-iglm` adds two matched
IgLM controls: SAbDab2-only and HER2-free ASD → SAbDab2. Both start from random
weights and use CE with one epoch per stage. It reuses the completed MANGO
evaluations and writes `plots/sabdab2_loss_comparison_with_iglm.png` / `.csv`
in the prototype results directory.

Use `./run_gpu.sh antigen-diagnostics` to probe antigen dependence, region NLL
and cluster uncertainty in the prototype. `./run_gpu.sh antigen-diagnostic-training`
runs small controlled follow-ups for target positions, pooling and IgLM warm
starts. See [the diagnostic protocol](docs/antigen_diagnostics.md).

Use `./run_gpu.sh asd-scaling` for the ASD-only 2,000/4,000-record comparison of
IgLM and paired/shuffled ESM2 cross-attention, with fixed held-out targets and
two-epoch optimization controls. See [the scaling protocol](docs/asd_scaling.md).
Use `./run_gpu.sh asd-scaling-8k` to append three 8,000-record runs and compare
them with those frozen baselines. The antigen cap is relaxed for 8k; two antigens
supply 33% of its records. Held-out targets and the entire 4k training prefix
remain fixed.
Use `./run_gpu.sh asd-scaling-16k` to extend the comparison to 16,000 records
and train regular MANGO with an antigen prefix at 2k/4k/8k/16k. The figure shows
overall and CDR3 NLL. The dominant antigen requires a partially shuffled 16k
control; the retained correct-antigen fraction is recorded and shown on the plot.
Use `./run_gpu.sh asd-scaling-fixed-positions` to add the improved MANGO prefix
variant with antigen-independent antibody positions at all four training sizes.

Changing from an older conditioning contract requires fresh training. The
architecture participates in run identity, and checkpoint loading rejects older
models. For standalone inference, `generate_designs.py` requires a compatible
checkpoint, one antigen embedding, and a chain choice:

```bash
python workflow/scripts/generate_designs.py \
  --ckpt artifacts/runs/NEW_RUN/checkpoints/best.pt \
  --antigen-emb artifacts/path/to/antigen.pt \
  --chain heavy --out artifacts/heavy_designs.csv \
  --gen-cfg '{"n_per_target": 10, "batch_size": 1, "max_new_tokens": 180}'
```
