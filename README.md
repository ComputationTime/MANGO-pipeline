# MANGO

MANGO trains an antigen-conditioned antibody language model. Its single task is:

> Given an antigen and a requested chain type, generate that antibody chain.

No antibody sequence or antibody-derived embedding is supplied as conditioning.
Training, evaluation, held-out reconstruction, and de novo generation all use
the same antigen-only biological input plus a learned heavy/light output
selector. Heavy-chain generation remains the downstream-analysis default.

## Pipeline at a glance

Four master scripts at the repository root run the whole study, in order. Every
setting lives in [`config/config.yaml`](config/config.yaml).

```text
build_dataset.py  ->  run_embedding.py  ->  run_training.py  ->  run_analysis.py
 fetch SAbDab2         antigen embeddings    train / evaluate /    cohort, scorers,
 standardize           (one per embedder)    predict / generate    structure QC,
 filter + split                                                    every figure
```

| Step | Master script | Implementation folder |
| --- | --- | --- |
| 1. Build the dataset and the train/val/test split | `build_dataset.py` | `scripts/data/` |
| 2. Embed antigens | `run_embedding.py` | `scripts/embedding/` |
| 3. Train, evaluate, predict, generate | `run_training.py` | `scripts/training/` |
| 4. Score designs and plot | `run_analysis.py` | `scripts/analysis/` |

`scripts/lib/` holds helpers shared by all stages (`pipeline.py` resolves the
config into paths and run ids; `device_common.py`, `mmcif.py`). The earlier
Snakemake workflow, GPU launcher and one-off studies are kept in `archive/`.

### 1. Dataset and splits

```bash
python build_dataset.py                    # fetch -> standardize -> process
python build_dataset.py --stages process   # redo only the filtering/splitting
```

The held-out **test** split comes from the dataset's antigen-aware split table;
the **validation** split is carved out of train by holding out complete
`ab_ag_cluster` groups (row-wise validation is rejected as leaky). The result is
`artifacts/data/sabdab2_v<version>/processed/records.csv`.

### 2. Embeddings: one script, choose the embedder

```bash
python run_embedding.py --all                      # every embedder
python run_embedding.py --embedders one_hot esm2   # a chosen subset
python run_embedding.py --esm2 --proteinmpnn       # per-embedder flags
python run_embedding.py --list                     # registered embedders
```

Embedders: `one_hot`, `biopython`, `pyrosetta_pre`, `esm2`, `esm3`, `esm3_structure`, `esmif`,
`proteinmpnn`. Each runs in its own subprocess, so a failure is
reported without stopping the others, and conflicting dependency stacks can use
separate environments via `--python esm3=/path/to/envs/embed_esm3/bin/python`
(see `envs/*.yaml`). ESM3 needs accepted Hugging Face terms and `HF_TOKEN`;
PyRosetta requires its own licensed install.

### 3. Training regimen

```bash
python run_training.py --all
python run_training.py --embedders one_hot esm2 --stages train evaluate
```

For each embedder: `train` (early stopping on val) -> `evaluate` (NLL and
perplexity by split, `eval.json`) -> `predict` (held-out reconstruction) ->
`generate` (de novo designs against one held-out target per test cluster). A run
lives in `artifacts/runs/<tag>__<hash>/` where the hash covers the dataset,
filters, embedder and model config. Existing checkpoints and design tables are
reused unless `--retrain` / `--regenerate` is passed.

### 4. Analysis and plots

```bash
python run_analysis.py --all                       # score + render everything
python run_analysis.py --stages plots              # re-plot from saved scores
python run_analysis.py --stages plots --only fig4  # a single figure
```

| Figure | Content |
| --- | --- |
| `fig1_nll` | NLL / perplexity by split and embedder |
| `fig2_structure_confidence` | Boltz-2 / Chai-1 confidence |
| `fig3_ablikeness` | IgLM, AntiBERTy, AbLang2 perplexity |
| `fig4_ld_germline` | Levenshtein distance from germline: **one violin plot, all embedders** |
| `fig5_developability` | six biophysical properties as violins |
| `fig6_gene_families` | heavy V/J families and species |
| `rosetta_interface_plumbing` | fixed-backbone interface QC |

Figures are written to `artifacts/analysis/figures/` with a `*_data.csv` beside
each. ANARCI usually needs its own environment:
`--python germline=/opt/anaconda3/bin/python`. Never relabel AbLang2 confidence
as pseudo-perplexity; use `--allow-missing-ablang2` to draw an explicit empty
panel when it was not scored.

## Repository layout

```text
build_dataset.py  run_embedding.py  run_training.py  run_analysis.py
config/config.yaml        experiment configuration (single source of truth)
scripts/data/             fetch, standardize, filter, train/val/test split
scripts/embedding/        one module per embedder + batch runner
scripts/training/         train, evaluate, predict, generate
scripts/analysis/         scorers, plot_*.py modules, generate_figures.py
scripts/lib/              shared helpers
mango/                    model components
envs/                     per-tool conda environments
tests/                    fast contract tests
archive/                  previous Snakemake workflow and one-off studies
artifacts/                generated data, models, figures (ignored by Git)
```

## Environment

```bash
conda env create -f environment.yaml
conda activate mango
```

Heavy, conflicting tools (ESM2/ESM3/ESM-IF, PyRosetta, ANARCI, AbLang2, Boltz,
Chai) have minimal environments in `envs/`; point the master scripts at them with
`--python`. Set `MPLCONFIGDIR` to a writable directory to avoid Matplotlib cache
warnings.

## Tests

```bash
python -m unittest discover -s tests -v
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
`archive/docs/structure_prediction_plan.md`.

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

## Antigen-only model contract

The default architecture is `antigen_chain_gpt2_v3`. Its prefix contains the
projected antigen representation, one learned heavy/light output selector, and
the beginning-of-sequence token. A causal GPT-2 decoder then emits the requested
chain. Only target residues and EOS receive loss; no antibody sequence,
antibody embedding or species label is accepted as conditioning.
