# MANGO pipeline — module signatures

> This document includes planned interfaces beyond the first completion
> milestone. The active Snakefile defaults to training/inference with the
> one-hot antigen embedder and exposes sequence-only analysis explicitly.
> Therapeutic targets, AF3, and TAP are deferred. Boltz-2 and Chai-1
> structure confidence run by default on a bounded, cluster-diverse subset and
> remain directly addressable through the `structure_confidence` target.

The input/output contract of every module. This is the reference for *what
crosses each boundary*; `planned_modules.md` covers implementation status and
open decisions.

## The task

**Given the antigen and the light chain, predict the heavy chain.**

Every module honours that one sentence, and it is what makes the study's
comparison meaningful — the antigen representation is the only thing that
varies, so it is the only thing that can explain a difference between models.

| | |
|---|---|
| conditioning | antigen embedding (`<tag>`) + light-chain context embedding |
| target | heavy chain |
| never an input | the heavy chain, at any stage |

The no-leak guarantee is **structural, not procedural**: the antibody context
embedding is built by running AbLang2 on `'*|L'` — the light chain with the
heavy slot masked — so the heavy chain is absent from every artifact the model
reads. No stage has to remember to hide it. The context directory is named
`ablang2_light_only` for the same reason: embeddings built under a different
task cannot be silently reused.

Loss is computed on heavy tokens only (the context block is labelled `-100`),
so every reported NLL is a heavy-chain NLL, undiluted by antigen or light-chain
length.

Conventions used below:

- `→` separates input from output.
- `NULLABLE` marks a column that may be empty.
- Multi-chain fields are **comma-separated**, one segment per antigen chain, in
  the same order as `antigen_chains`.
- `<tag>` is an embedder tag from `config.embedders` (the study's central axis).
- `<run>` is `<tag>__<experiment_hash>`.

---

## 0. Global state file

`config/config.yaml` — every hyperparameter and experiment option. Nothing in
`workflow/` hardcodes a path, a model size, or a split name.

| Section | Controls |
|---|---|
| `experiment` | study name, global seed |
| `dataset` | which dataset + version + URL, artifact root, split table/column |
| `processing` | filters, validation-split strategy |
| `embedding` | `seq_source` (resolved\|expected), splits to embed, antibody method + `context` |
| `embedders` | **the 8 antigen representations** — registry of tags |
| `active_embedders` | which tags this invocation runs |
| `model` | architecture, training hyperparameters, retrain gating |
| `generation` | `source` split, target cap, designs per target, sampling |
| `structure_prediction` | local predictors, deterministic design cap, targets (null ⇒ follow generation), sampling/recycling and optional MSA service |
| `analysis` | compared embedders, metrics, figures, format |
| `weights` | external weight roots and source URLs |

### Embedder registry entry

```yaml
embedders:
  <tag>:
    method: one_hot | biopython | pyrosetta_pre | esm2 | esm3 | esmif |
            proteinmpnn | afm     # selects the rule + script
    class:  naive | biophysical | learned   # figure grouping
    label:  "One-hot"                       # figure label
    # ...remaining keys are that method's parameters, e.g.
    size: t33_650M                          # esm2
    model: vanilla_models                   # proteinmpnn
    noise: 2                                # proteinmpnn
    representation: single                  # afm
    include_light_chain: true               # afm
```

`method` chooses the rule; everything else is passed to the script as
`params.spec`. Several tags may share a method — each rule constrains its
`{embedder}` wildcard to only its own active tags.

### Derived identity

```python
experiment_hash(tag) = sha1({dataset, processing, embedding,
                             embedders[tag], model, experiment.seed})[:8]
run_id(tag)          = f"{tag}__{experiment_hash(tag)}"
```

Changing any input yields a new run directory rather than overwriting a result.

---

## 1. Dataset fetcher

```
dataset.name, dataset.version, dataset.zenodo_url
  → artifacts/data/<name>_v<version>/splits_final/…
  → artifacts/data/<name>_v<version>/.<name>_ready        (marker)
```

| | |
|---|---|
| rule | `fetch_sabdab2` · alias `fetch` |
| script | `fetch_sabdab2.py` |
| env | `fetch.yaml` |

Downstream depends on the **marker**, not the thousands of unpacked files. The
marker is written only after the archive is extracted and the configured split
table verified.

---

## 2. Data standardization

```
raw dataset (marker)
  → artifacts/data/<name>_v<version>/standardized/standardized.csv
```

| | |
|---|---|
| rule | `standardize` |
| script | `standardize_<dataset.name>.py` (registered in `standardize.smk`) |
| contract | `standardize_common.REQUIRED_COLUMNS` / `.validate()` |
| env | `process.yaml` |

**This is the dataset seam.** Everything downstream is dataset-agnostic.

### Output schema

| Column | Notes |
|---|---|
| `id` | unique; also the stem of every embedding filename |
| `pdb_path` | structure file for this row |
| `antigen_chains` | author chain ids of the antigen chains |
| `expected_heavy_seq` | NULLABLE |
| `expected_light_seq` | NULLABLE |
| `expected_ag_seq` | NULLABLE, one segment per antigen chain |
| `split` | NULLABLE — source split if the dataset has one |

Validated on write: required columns present, `id` unique, `id`/`pdb_path`
non-empty, and `antigen_chains` / `expected_ag_seq` agree in arity.

**Optional passthrough.** An adapter may emit extra columns; `process` uses them
when present. SAbDab2 passes through `pdb_id, heavy_chain, light_chain,
antigen_types, resolved_H_seq, resolved_L_seq, resolved_ag_seq, ab_type,
resolution, ab_ag_cluster` — the resolved sequences let `process` skip parsing
~15k structures.

**Adding a dataset:** write `scripts/standardize_<name>.py` with the same
entrypoint shape, register it in `rules/standardize.smk`, set `dataset.name`.

---

## 3. Data processing

```
standardized.csv + config.processing
  → artifacts/data/<name>_v<version>/processed/records.csv
```

| | |
|---|---|
| rule | `process_records` — a Snakemake **checkpoint** · alias `process` |
| script | `process_records.py` |
| env | `process.yaml` |

A checkpoint because the kept `id` set is unknown until it runs; embedding fans
out over exactly those ids.

### Output schema

Standardized columns, plus:

| Column | Notes |
|---|---|
| `chains` | every chain used: heavy, light, then antigen chains |
| `resolved_H_seq` | structurally resolved heavy sequence |
| `resolved_L_seq` | structurally resolved light sequence |
| `resolved_ag_seq` | one segment per antigen chain |
| `split` | `train` \| `val` \| `test` — always populated |

Resolved sequences come from the standardizer when available, else are parsed
from the structure via `lib/mmcif.py` (backbone-complete residues only, the same
residue set the structure embedders use — so `L` stays comparable across
representations).

### Filters (`config.processing`)

| Key | Effect |
|---|---|
| `require_paired` | keep only rows with both a heavy and a light chain |
| `antigen_types` | every antigen chain type must be in this set |
| `max_antigen_len` | cap on summed **resolved** antigen length; `null` disables |
| `require_structure_file` | drop rows whose structure file is missing |
| `val.strategy` | must be `cluster`; row-wise random splitting is rejected |
| `val.cluster_column` | grouping boundary, currently `ab_ag_cluster` |
| `val.fraction` | fraction of training clusters assigned to validation |
| `val.seed` | deterministic cluster sampling seed |

---

## 4. Embedder modules ×8

```
records.csv (+ structure file for structure-based methods)
  → embeddings/antigen/<tag>/embedder_config.json
  → embeddings/antigen/<tag>/{train,val,test,target}/<id>.pt
```

| | |
|---|---|
| rules | `embed_antigen_<method>` · aliases `embed_antigen`, `embed_antibody`, `embed` |
| scripts | `embed_antigen_<method>.py` |
| wildcards | `{embedder}` (constrained per method), `{split}`, `{instance}` |
| env | one per method — heavy/conflicting deps stay isolated |

### Antibody context embedder

The antibody side is **held constant** across the study — that is what makes any
difference between models attributable to the antigen representation:

```
records.csv → embeddings/antibody/ablang2_light_only/{train,val,test,target}/<id>.pt
```

What is embedded is the **light chain with the heavy slot masked**: AbLang2 is
run on `'*|L'`, never `'H|L'`. This is the study's no-leak guarantee, and it
lives here rather than in the model so that no downstream stage can undo it. The
context is part of the directory name, so a change of task cannot silently reuse
embeddings built under the old one. Its meta carries `chains: ["L"]`,
`context: "light_only"`, `masked_chains: ["H"]`.

`m` (the context length) = `len(light) + 2` — the mask token, the `|`
separator, and one row per light-chain residue.

### `.pt` payload

```python
{
  "embedding": Tensor[L, H] float32 cpu,
  "shape": [L, H],
  "embedder": "<tag>",
  "model_name": "esm2_t33_650M_UR50D",
  "length": L,
  "dim": H,
  "chains": ["A", "B"],
  "chain_separator_token": True,
  "id": "<id>",
  "seq_source": "resolved" | "expected" | "structure",
  # method-specific extras, e.g. biopython: axis="residue", charge_ph=7.2
}
```

Built by `embed_common.build_meta`, written by `embed_common.save_embedding`
(which also enforces 2-D `[L, H]`).

### `embedder_config.json`

```json
{ "tag": …, "method": …, "class": …, "label": …,
  "params": { …method-specific… }, "seq_source": … }
```

One per embedder, written by its own rule so it exists before any structure is
embedded and can be regenerated without the embedder's heavy environment.

### The eight

| tag | method | input | H |
|---|---|---|---|
| `one_hot` | sequence | resolved/expected seq | 21 |
| `biopython` | sequence | resolved/expected seq | 11 (chain descriptor repeated per residue) |
| `pyrosetta_pre` | structure | `.cif` | 1 |
| `esm2` | sequence | resolved/expected seq | 320–5120 by size |
| `esm3` | sequence | resolved/expected seq | 1536 |
| `esmif` | structure | `.cif` | 512 |
| `proteinmpnn` | structure | `.cif` + weights | 128 |
| `afm` | sequence → predicted | antigen seq **+ light chain** | representation-dependent |

`L` convention: sequence methods emit one row per residue plus one
chain-separator row between chains, so `L = Σ chain_len + (n_chains − 1)`.
The adapted ESM-IF and configured PyRosetta PRE methods also insert a separator
row between chains; ProteinMPNN emits backbone-complete residues without one.

One deliberate exception:

- `afm` — folded on the antigen chains **plus the light chain**, so
  `L = Σ ag_chain_len + len(light) + n_chains − 1`. AF-M's value over a
  sequence model is that it models the interface; folding the antigen alone
  would make it a strictly worse ESM2. The light chain is inside the
  conditioning set, and the heavy chain is never given to AF-M, so this cannot
  leak the target. Cross-attention imposes no constraint on `L` (the antigen is
  keys/values), but do not read this `L` as comparable to the other seven.
  `embed_antigen_afm.build_fold_input` assembles and validates the chain dict —
  that part is implemented and testable without any AF infrastructure.

---

## 5. Target module (Aim 2 held-out therapeutic complexes) — optional

**Currently inactive.** `generation.source` is `test`, so Aim 2 designs against
held-out complexes from the dataset itself and this module is not in the DAG.
It stays wired so switching to the therapeutic panel is a one-line config
change (`generation.source: target`) plus filling in `antigen_chains`.

```
config.generation.targets
  → artifacts/targets/structures/<PDB>.cif
  → artifacts/targets/targets_records.csv       (records-shaped, split="target")
```

| | |
|---|---|
| rules | `fetch_target_structure`, `standardize_targets` · aliases `targets`, `embed_targets` |
| scripts | `fetch_target_structure.py`, `standardize_targets.py` |

The targets table uses the **same schema as `records.csv`** with
`split="target"`, so all eight antigen embedder rules serve these structures
unchanged — their embeddings land at
`embeddings/antigen/<tag>/target/<PDB>.pt` beside the train/val/test folders.

Each entry needs **both** `antigen_chains` and `light_chain` — the model
conditions on the antigen and the light chain, so both must be named. No heavy
chain is ever recorded: `resolved_H_seq` is deliberately empty, because that is
what the model predicts.

`standardize_targets` **refuses to guess** either field: a null value, a chain
absent from the structure, or a `light_chain` that is also listed as an antigen
chain is a hard error listing every chain found, with its length and sequence.
Conditioning on the wrong chain — the Fab instead of the antigen, or the heavy
chain we are meant to predict — would silently invalidate every design.

---

## 6. MANGO train module

```
records.csv + embeddings/antigen/<tag>/… + embeddings/antibody/ablang2_light_only/…
  → artifacts/runs/<run>/
```

| | |
|---|---|
| rule | `train_model` · alias `train` |
| script | `train_mango.py` |
| wildcard | `{run}` → tag via `tag_for_run()` |
| env | `model.yaml` |

### Sequence layout (`model_common.MangoModel`)

Teacher-forced causal LM over the heavy chain:

```
positions: [ ctx_0 … ctx_{m-1} ]  [ < ]  [ h_1 … h_n ]  [ > ]
            antigen-conditioned   start   heavy chain    end
            light-chain context          (teacher-forced)

labels:     -100 × (m+1)                  h_1 … h_n       >
```

- `ctx` = the light-chain context embedding cross-attended against the antigen
  embedding (`fuse`), dim 480 = MANGO's hidden size.
- Heavy tokens enter through GPT2's own `wte`, so each heavy position sees only
  earlier heavy positions — the causal mask does the work.
- Labels are `-100` across the whole context block, so loss covers exactly
  `len(H) + 1` tokens (residues + end token) and nothing else.
- Generation reuses this identical prefix (`ctx` + `<`), so what is sampled is
  exactly what was trained.

### Run directory

| File | Contents |
|---|---|
| `config.json` | `run_id`, `experiment_hash`, `embedder`, and the full global state |
| `model_config.json` | architecture + `d_ag`; what inference reads |
| `metrics.jsonl` | one object per epoch: `epoch, iteration, train_loss, val_loss, embedder, run_id, improved` |
| `training_curve.csv` | iteration-level training loss plus epoch-end validation loss |
| `training_curve.png` | live-refreshed loss plot with training iteration on the x-axis |
| `checkpoints/best.pt` | lowest val loss |
| `checkpoints/latest.pt` | last epoch — resume point |

### `model_config.json`

```json
{ "run_id": …, "experiment_hash": …, "embedder": …,
  "antibody_embedder": "ablang2_light_only",
  "task": "antigen+light->heavy",
  "d_ag": 21, "d_model": 480,
  "n_cross_attn_heads": 1, "n_cross_attn_layers": 1,
  "vocab_size": 26, "n_layer": 4, "n_head": 8, "n_positions": 2048 }
```

`d_ag` is read from a real embedding tensor at train time, so a mis-declared
hidden dim is impossible.

### Checkpoint payload

```python
{ "cross_attn": state_dict, "lm": state_dict, "optimizer": state_dict,
  "d_ag": int, "n_heads": int, "n_layers": int,
  "embedder": str, "antibody_embedder": str,
  "run_id": str, "experiment_hash": str, "val_loss": float, "epoch": int }
```

Training optimises on `model.train_splits`, early-stops on `model.val_splits`,
and never reads test. `model.retrain: false` marks training inputs `ancient()`
so upstream changes do not retrigger it.

---

## 7. MANGO inference modules

### 7a. Evaluate — the data behind figure 1

```
records.csv + checkpoints/best.pt + model_config.json
  → runs/<run>/eval.json
```

| | |
|---|---|
| rule | `evaluate_model` · alias `evaluate` |
| script | `evaluate_mango.py` |

```json
{ "run_id": …, "experiment_hash": …, "embedder": …,
  "task": "antigen+light->heavy", "d_ag": …,
  "checkpoint_epoch": …, "checkpoint_val_loss": …,
  "splits": { "train": { "nll": …, "perplexity": …, "n_tokens": …,
                         "n_examples": …, "n_skipped_missing_embeddings": … },
              "val": {…}, "test": {…} } }
```

NLL is **token-weighted** (total cross-entropy ÷ predicted tokens), so it is
insensitive to a split's length distribution and bars are comparable. Predicted
tokens are the heavy chain plus its end token — `n_tokens = Σ (len(H) + 1)` —
so figure 1 compares heavy-chain NLLs, not a number diluted by how long the
antigens happened to be.

### 7b. Predict — test-split reconstruction

```
records.csv + best.pt + antigen embeddings (test) + context embeddings (test)
  → runs/<run>/predictions_test.csv
```

| | |
|---|---|
| rule | `predict_model` · alias `predict` |
| script | `predict_mango.py` |

```
embedder, run_id, status, split, id, pdb_path, antigen_chains,
light_seq, true_heavy_seq, predicted_heavy_seq, prediction_length, checkpoint
```

Conditioned on the antigen embedding and the light-chain context — the model's
full and only input. `true_heavy_seq` is carried for comparison; it is never
fed to the model, and the context embedding was built with the heavy slot
masked. A per-structure failure is recorded in `status`, not raised.

### 7c. Generate — Aim 2 de novo designs

```
best.pt + model_config.json + records for generation.source
  + embeddings/antigen/<tag>/<source>/<id>.pt
  + embeddings/antibody/ablang2_light_only/<source>/<id>.pt
  → artifacts/designs/<run>/<id>/designs.csv
```

| | |
|---|---|
| rule | `generate_designs` · alias `generate` |
| script | `generate_designs.py` |
| wildcards | `{run}`, `{instance}` |

```
embedder, run_id, target_id, split, design_index, sequence, length, status
```

`generation.n_per_target` heavy chains per structure. Only heavy chains are
ever generated — that is the task. One job per `(run, structure)` so the
pipeline's most expensive step parallelises and one bad target does not cost
the whole sweep.

**Which structures**: `generation.source` names a split — `test` (current) uses
held-out dataset complexes, `target` uses the therapeutic panel of §5. Either
source must supply both an antigen and a light chain, since that pair is the
entire conditioning signal. For dataset splits, `generation.target_selection`
selects one deterministic representative per held-out `ab_ag_cluster` and fails
if a selected cluster occurs in train or validation. `generation.max_targets`
can optionally cap that already-diversified cohort.

Duplicates are **kept** — the repeat rate is itself signal about how sharply a
representation constrains generation.

---

## 8. Analysis modules

Every rule fans **in** over `analysis.embedders` (null ⇒ `active_embedders`), so
figures compare representations rather than describing one.

### 8a. Shared cohort and metric tables

```
designs/<run>/*/designs.csv
  → artifacts/analysis/<run>/cohort.csv
  → artifacts/analysis/<run>/metrics/{iglm,antiberty,ablang2,germline,biophysical}.csv
```

| | |
|---|---|
| cohort rule | `analysis_cohort` |
| metric rules | `analysis_score_iglm`, `analysis_score_antiberty`, `analysis_score_ablang2`, `analysis_score_germline`, `analysis_biophysical` |
| scripts | independent scripts under `workflow/scripts/analysis/` |

The cohort is a deterministic per-target subset of successful designs, selected
from `(seed, target_id, design_index)` and joined to its light-chain context.
Figure 3 uses IgLM mean log likelihood, AntiBERTy mean pseudo-log-likelihood,
and AbLang2 paired H|L confidence. Figure 4 records ANARCI's nearest heavy V/J
calls, reconstructed germline reference, raw LD, and normalized LD. Figure 5
records GRAVY and charge at pH 7.4.

### 8b. Structure confidence (Aim 3) — bounded default

```
designs.csv + records.csv (generated H + cognate L + all antigen chains)
  → analysis/<run>/structures/<id>/<predictor>_raw/
  → analysis/<run>/structures/<id>/<predictor>_confidence.csv
```

| | |
|---|---|
| rules | `predict_boltz_confidence`, `predict_chai_confidence` |
| target | `structure_confidence` |
| predictors | `boltz2`, `chai`; AF3 ingestion deferred |

```
embedder, run_id, target_id, design_index, sequence, predictor, sample_index,
confidence_score, ptm, iptm, complex_plddt, mean_pae,
has_inter_chain_clashes, structure_path, status
```

Capped at `structure_prediction.n_designs` per target — folding every design is
not affordable; which designs were selected is recorded so the sampling is
auditable.

Which structures get folded: `structure_prediction.targets`, or — when that is
null, the default — the first `n_targets` ids of the generation set, so Aim 3
always folds designs Aim 2 actually produced.

### 8c. Complex metrics (Aim 3) — deferred

```
<sp_method>_scores.csv (all predictors)
  → analysis/<run>/structures/<id>/complex_metrics.csv
```

| | |
|---|---|
| rule | `analysis_complex_metrics` |
| script | `analysis_complex_metrics.py` |

```
embedder, run_id, target_id, design_index, sp_method,
pdockq2, interface_hydrophobicity, cdr_sap, interface_ddg, status
```

### 8d. Figures

```
eval.json ×N | metrics/*.csv ×N
  → artifacts/analysis/figures/<figure>.<format>
  → artifacts/analysis/figures/<figure>_data.csv
```

| | |
|---|---|
| rules | one pure plotting rule per figure; aliases `figures`, `analysis` |
| scripts | `workflow/scripts/analysis/plot_*.py` |

| Figure | Source | Panels |
|---|---|---|
| `fig1_nll` | `eval.json` | training and cluster-held-out test NLL |
| `fig3_ablikeness` | three LM tables | IgLM · AntiBERTy · AbLang2 |
| `fig4_ld_germline` | ANARCI table | raw LD distributions; normalized LD retained in CSV |
| `fig5_developability` | biophysical table | GRAVY · charge@pH 7.4 |

Every figure also writes `<figure>_data.csv` — the exact numbers behind the
panels, and the accessible table view.

Colour encodes the **embedder**, assigned in config order from a fixed palette,
so a representation keeps its colour in every panel. One measure per axis;
never a second y-scale. Plot scripts perform no model inference.

**Adding a figure:** add a plot-only script under `scripts/analysis/`, declare
its inputs in `rules/analysis/plots.smk`, and add it under `analysis.plots`.

---

## Aggregation targets

```bash
snakemake --sdm conda --cores 8 <target>
```

| Target | Builds |
|---|---|
| `fetch` | raw dataset + marker |
| `standardize` | `standardized.csv` |
| `process` | `records.csv` |
| `embed_antigen` / `embed_antibody` / `embed` | embeddings for active tags |
| `train` / `evaluate` / `predict` / `generate` | per active tag |
| `inference` | evaluation plus held-out predictions per active tag |
| `analysis_metrics` | deterministic cohort plus all sequence metric tables |
| `figures` / `analysis` | enabled figures plus plotted-data CSVs |
| `all` | default — `inference` |

Widen the antigen representation study only after validating an embedder, then
add its tag to `active_embedders`.

---

## Script entrypoint contract

Every script is both a Snakemake `script:` target and a standalone CLI:

```python
def <module>(...) -> None:
    """Pure function: explicit arguments, no global state."""

def main():
    smk = globals().get("snakemake")
    if smk is not None:
        <module>(...)          # from smk.input / smk.output / smk.params
        return
    # argparse fallback for standalone runs and debugging
```

Constraints: no `from __future__ import annotations` in a `script:`-invoked
file; `mango.*` submodules are imported without executing `mango/__init__.py`
(see `embed_common.import_mango_mpnn`, `model_common.import_mango`) so a minimal
environment never has to satisfy the full model stack.
