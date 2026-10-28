# Rapid HER2-free ASD prototype

Run `./run_gpu.sh asd-prototype` for the small, one-epoch version of the
[eight-model ESM2 comparison](asd_loss_comparison.md). This is the current rapid
iteration workflow. The full experiment and its interrupted checkpoints remain
separate; the prototype starts fresh model weights.

`config/asd_prototype.yaml` specifies:

- A maximum of 2,000 ASD training records, 200 validation and 200 test records.
- Antigens up to 512 residues, with no truncation.
- Seeded round-robin selection across antigens, at most ten records per antigen.
- Exactly **one epoch for every training stage**: four ASD pretraining stages,
  four SAbDab2-only runs, and four SAbDab2 fine-tuning runs.

The resulting ASD cohort contains:

| Split | Records | Scored chain examples | Distinct antigen sequences |
|---|---:|---:|---:|
| Training | 2,000 | 3,979 | 1,247 |
| Validation | 95 | 185 | 40 |
| Test | 200 | 393 | 93 |

The source is the existing, positive-binding, CATH-protected ASD audit. The
preparer identifies HER2/ERBB2 target names, orthologs and multi-target names in
the source annotation database and excludes all corresponding antigen sequence
IDs across sources. It additionally excludes every BUZZ antigen ID. The current
audit blocks ten HER2-associated antigen IDs, removing 525,392 source records;
the short-antigen filter removes another 69,915. The remaining pool is sampled
for breadth rather than allowing a single library to dominate. No new binding
thresholds or negative examples are introduced.

The original group-aware ASD train/validation/test assignments remain intact.
Subsampling occurs within those fixed splits. CATH families, exact antibody
chains and `ab_ag_cluster` IDs cannot cross splits. Individual chain target
masks are recomputed after selection so sampling does not discard a shared light
target merely because its earlier representative row was omitted.

HER2 removal applies to **ASD pretraining**. The native SAbDab2 comparison stays
byte-identical to the preceding audited cohort: 509 training, 56 validation,
and all 426 original test records, including the 362-record primary subset.
The same native data are used for every final model.

All representations are frozen ESM2 650M. Existing per-antigen ESM2 tensors are
reused through `esm2_cache_dir`; a separate small index binds them to prototype
records. No weights from the long ASD-trained MANGO checkpoint initialize the
prototype. Normal and cross-attention models are trained with standard CE or
focal loss (gamma 2); focal applies in both pretraining and fine-tuning for its
arms. Evaluation and checkpoint selection always use ordinary NLL.

All outputs stay under `artifacts/experiments/asd_her2_free_prototype_v1/`:

- `cohort/split_audit.json`: counts, HER2 exclusion IDs, pinned inputs, selection
  settings and overlap checks.
- `asd/`, `sabdab2_scratch/`, `sabdab2_finetuned/`: stage checkpoints, live curves,
  metrics and evaluations.
- `results_report.json` / `.csv`: available results as the queue progresses.
- `plots/sabdab2_loss_comparison.png` / `.csv`: the eight-model comparison.
- `verification.json`: independent checks of epochs, initialization, cohort
  membership, token counts and plotted values after completion.

Rerunning the same launcher command reuses the immutable cohort, frozen
embeddings and completed stages. These are pilot results from a deliberately
small, short-sequence subset and a one-epoch training budget.

Verify the completed prototype independently:

```bash
python3 workflow/scripts/verify_asd_loss_comparison.py \
  --root artifacts/experiments/asd_her2_free_prototype_v1
```

Add the matched IgLM baselines with `./run_gpu.sh asd-prototype-iglm`.
`config/asd_prototype_iglm.yaml` points to the completed prototype and inherits
its immutable training settings, including seed, Adam learning rate, gradient
accumulation, chain target masks and one epoch per stage. It trains:

- IgLM from random initialization on SAbDab2 only.
- IgLM from random initialization on HER2-free ASD, then fine-tunes its weights
  on SAbDab2 with a fresh Adam optimizer.

These are the full 12,889,600-parameter IgLM architecture and native vocabulary.
They use native H/L selectors and a fixed `[HUMAN]` species token; no per-record
species annotation, antigen or antibody prompt is supplied. Public IgLM weights
are never loaded. Both baselines train with ordinary cross-entropy and score
the same residues plus EOS, including the exact primary and full test subsets.
Focal loss remains a MANGO comparison arm.

IgLM stage outputs live under
`artifacts/experiments/asd_her2_free_prototype_iglm_v1/`. Its report independently
checks cohort fingerprints, exact masked training examples, one epoch, fresh
fine-tuning optimizer provenance, per-chain counts and pooled NLL. Run the
verification again with:

```bash
python3 workflow/scripts/report_prototype_iglm.py \
  --root artifacts/experiments/asd_her2_free_prototype_iglm_v1 \
  --mango-root artifacts/experiments/asd_her2_free_prototype_v1 \
  --cohort artifacts/experiments/asd_her2_free_prototype_v1/cohort \
  --require-complete
```

The combined ten-model figure and its 40-row table are
`artifacts/experiments/asd_her2_free_prototype_v1/plots/sabdab2_loss_comparison_with_iglm.png`
and `.csv`. The eight-model figure remains available. Rerunning this launcher
reuses completed IgLM stages and does not retrain MANGO or recompute embeddings.
