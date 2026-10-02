# MANGO workflow — long-term module status and contracts

> Repository default: one-hot antigen embedding. The local-GPU overlay runs
> one-hot, BioPython, ESM2, ESM-IF, and ProteinMPNN sequentially, with ESM3 as
> an authenticated opt-in and PyRosetta PRE as an academic/non-commercial
> validation opt-in. Therapeutic targets, AF-M, AF3, and TAP are deferred.
> Boltz-2/Chai-1 Figure 2 confidence runs by default on a bounded,
> cluster-diverse subset and has passed a two-sequence GPU plumbing check.

What each module does, what it emits, and exactly what is left to implement.
The global state file (`config/config.yaml`) is the single source of truth for
every hyperparameter and experiment option; nothing in `workflow/` hardcodes a
path, a model size, or a split name.

**The task, everywhere: given the antigen and a requested chain type, generate
that antibody chain.** No antibody sequence or antibody-derived embedding enters
the model prefix. See `module_signatures.md` for the exact sequence layout.

## Active milestone pipeline

```
fetch → standardize → process → embed antigen → train
                                             ├→ evaluate
                                             ├→ predict
                                             └→ generate (explicit)
                                                   └→ analysis (explicit)
```

| Stage | Output | Status |
|---|---|---|
| `fetch` | raw dataset + ready marker | works |
| `standardize` | `standardized.csv` (dataset-agnostic contract) | works |
| `process` | `records.csv` (filtered, resolved, split) | works |
| `embed` | `<tag>/{train,val,test}/<id>.pt` + `embedder_config.json` | five non-gated methods in the GPU profile; persistent-model batches avoid per-record reloads |
| `targets` | external therapeutic complexes | deferred and outside the active DAG |
| `train` | `runs/<tag>__<hash>/` | works |
| `evaluate` | `eval.json` → figure 1 | works |
| `predict` | `predictions_test.csv` | works |
| `generate` | `designs/<run>/<id>/designs.csv` | works |
| `analysis` | cohort + modular metrics + figures | Figures 1–5 work; Figure 2 uses bounded Boltz-2/Chai-1 confidence; AF3/TAP deferred |

## Module contracts

### 1. Dataset fetcher
`dataset.name` (global state) → raw dataset + `.<name>_ready` marker.
Downstream depends on the marker, not the thousands of unpacked files.

### 2. Standardization
Raw → `standardized.csv`. **This is the dataset seam.** Contract columns:

```
id, pdb_path, antigen_chains,
expected_heavy_seq, expected_light_seq, expected_ag_seq, split
```

All but `id`/`pdb_path` may be empty. Adapters may carry extra columns through;
`process` uses them when present. Validated by `standardize_common.validate`.

Add a dataset: write `scripts/standardize_<name>.py`, register it in
`rules/standardize.smk`, set `dataset.name`. Nothing else changes.

### 3. Processing
`standardized.csv` → `records.csv`, adding:

```
chains, resolved_H_seq, resolved_L_seq, resolved_ag_seq, split ∈ {train,val,test}
```

Resolved sequences come from the standardizer when it has them (SAbDab2 does —
this avoids parsing ~15k structures) and are otherwise parsed from the structure
via `lib/mmcif.py`. A Snakemake **checkpoint**: the kept id set is unknown until
it runs, and embedding fans out over exactly those ids.

### 4. Embedders ×8

`embeddings/antigen/<tag>/{train,val,test,target}/<id>.pt`, plus one
`embedder_config.json` per tag. Each `.pt` holds
`{embedding: [L,H] float32, shape, embedder, model_name, length, dim, chains,
chain_separator_token, id, seq_source}`.

All eight rules write the same output pattern and constrain `{embedder}` to
their own active tags (`method_constraint`), so the DAG stays unambiguous and
two tags may share a method (e.g. two ESM2 sizes) for free.

| tag | class | H | status |
|---|---|---|---|
| `one_hot` | naive | 21 | **works** |
| `biopython` | biophysical | 11 | **works** — chain descriptor repeated per residue |
| `esm2` | learned | 320–5120 by size | **works** |
| `proteinmpnn` | learned | 128 | **works** |
| `pyrosetta_pre` | biophysical | 1 | contributed implementation adapted, **untested locally** |
| `esm3` | learned | 1536 | contributed implementation adapted; sequence-only, needs weights/auth |
| `esmif` | learned | 512 | contributed implementation adapted; needs fair-esm GVP environment |
| `afm` | learned | representation-dependent | **stub** (input prep done) |

Remaining implementation/runtime constraints:

- **esm3** — the contributed implementation is sequence-only and uses per-chain
  embedding plus retained BOS separator rows. Do not describe it as a
  sequence+structure representation. Weights may require Hugging Face auth.
- **esmif** — the contributed implementation encodes each antigen chain's
  N/CA/C backbone independently and inserts a zero separator. It is not a joint
  multichain-complex encoding; the fair-esm GVP/PyG dependencies are pinned to
  PyTorch 2.5.1 and CUDA 12.1 in an isolated environment.
- **afm** — its prepared fold input contains antigen chains only, matching the
  study-wide conditioning contract. It still needs weights, MSAs, and GPU
  infrastructure; strongly prefer external execution plus result ingestion.
  `representation` stays configurable (`single` | `pair` | `structure`).
- **pyrosetta_pre** — explicit mmCIF loading and author-chain checks are now in
  place. Confirm the PyRosetta environment and Rosetta chain mapping on its
  first real cluster run.

### 5. Train
Teacher-forced causal LM over each configured target chain. The prefix contains
projected antigen tokens, a learned heavy/light output selector, and BOS. Loss is
masked to target residues plus EOS; no antibody sequence is conditioning input.

One run per embedder at `runs/<tag>__<experiment_hash>/`:

```
config.json          full global state this run was built from
model_config.json    architecture + d_Ag_rep (inference reads this)
metrics.jsonl        one JSON object per epoch, including global iteration
training_curve.csv   iteration-level train loss + epoch-end validation loss
training_curve.png   live-refreshed iteration-axis loss plot
checkpoints/best.pt  lowest val loss
checkpoints/latest.pt
```

`experiment_hash` covers dataset + filters + embedding choices + the embedder's
own params + model hyperparameters, so changing any of them yields a new run
directory rather than silently overwriting a result.

### 6. Inference
- `evaluate` → `eval.json`: token-weighted NLL + perplexity overall and by chain
  type per split (figure 1).
- `predict` → `predictions_test.csv`: test-split selected-chain reconstruction.
- `generate` → `designs/<run>/<id>/designs.csv`: Aim 2 de novo selected chains,
  one job per (run, structure). Duplicates are kept — the repeat rate is itself
  signal about how sharply a representation constrains generation.

**What generation designs against** is `generation.source`, a split name:

- `test` (current) — one deterministic representative from every held-out
  `ab_ag_cluster`. Selection fails if any chosen cluster occurs in train or
  validation; `generation.max_targets` can optionally cap the diverse cohort.
- `target` — the eight therapeutic complexes. Requires
  `generation.targets[].antigen_chains`; `standardize_targets` refuses to guess
  them and reports the chain ids, lengths, and sequences it found. An optional
  recorded partner chain is downstream structure-analysis metadata only.

Either source must supply an antigen. No antibody sequence is a model input.

### 7. Analysis
Every rule fans **in** over `analysis.embedders`, so figures compare
representations rather than describing one.

| figure | source | status |
|---|---|---|
| `fig1_nll` | `eval.json` | **works** — train and cluster-held-out test NLL |
| `fig2_structure_confidence` | Boltz-2/Chai-1 normalized confidence tables | **bounded default** — ranking confidence, pTM, ipTM; AF3 deferred |
| `fig3_ablikeness` | independent LM score tables | **works** — IgLM, AntiBERTy, AbLang2 |
| `fig4_ld_germline` | ANARCI germline table | **works** — raw and normalized LD |
| `fig5_developability` | BioPython table | **works** — GRAVY and charge@7.4; TAP deferred |

Figures 3–5 share `cohort.csv`, a deterministic per-target selection of
successful designs. Metric tools are isolated into separate environments and
tables; plotting jobs perform no model inference. Figure 4 stores ANARCI's
nearest heavy V/J calls and the reconstructed germline reference used for LD.

Boltz-2 and Chai-1 structure confidence execution is available through the
default bounded `structure_confidence` branch and separate environments; GPU smoke
validation remains required. AF3 ingestion and complex metrics remain deferred.
Suggested next order: AF3 ingestion, then pDockQ2, interface hydrophobicity, CDR
SAP, and interface ddG.

## Conventions for new modules

- Read the row from `records.csv` via `embed_common.load_row`.
- Save with `embed_common.save_embedding(path, tensor_LH, build_meta(...))`.
- Per-rule conda env in `workflow/envs/`. Heavy or conflicting deps stay
  isolated — ESM2 (`fair-esm`) and ESM3 (`esm`) both import as `esm` and can
  never share an environment.
- Stage or prefetch pretrained assets through explicit Snakemake rules. Direct
  files live under `artifacts/weights`; package-managed ESM/Hugging Face files
  live under project caches in `artifacts/cache` with verified readiness markers
  under `artifacts/weights/pretrained`. Compute jobs retain the upstream
  automatic-download fallback but normally see a warm cache.
- Entrypoint pattern: a pure function plus a `main()` dispatching on the
  injected `snakemake` object or argparse, so every script also runs standalone.
- No `from __future__ import annotations` in a `script:`-invoked file.

## Figures

Charts follow one grammar: colour encodes the **embedder** (assigned in config
order from a CVD-validated 8-slot palette, never cycled), one measure per axis,
never a second y-scale. Every figure also writes `<figure>_data.csv` — the exact
numbers behind the panels, which doubles as the accessible table view.

Add a figure as a plot-only script under `scripts/analysis/`, declare its exact
metric-table inputs in `rules/analysis/plots.smk`, and add it under
`analysis.plots` in the config.
