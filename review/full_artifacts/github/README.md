# MANGO Full-Run Review Artifacts

This directory is updated one model at a time as each full run completes. It
contains lightweight review outputs only: no checkpoints, pretrained weights,
embedding tensors, or raw 40,000-row design sets.

## Completed models

| Model | Run ID | Epochs | Train NLL (final) | Validation NLL (best/final) | Test NLL | Status |
|---|---|---:|---:|---:|---:|---|
| one_hot | `one_hot__79bed882` | 5 | 0.6289 | 1.0290 | 1.0792 | Complete |

For `one_hot`, validation NLL improved at every epoch: 1.5787, 1.2418,
1.1363, 1.0581, and 1.0290. Five epochs were stable and useful, but the curve
had not fully plateaued. A longer follow-up should retain early stopping and
the best-validation checkpoint because the train/validation gap is widening.

Start with:

- `embedders/one_hot/run/training_curve.png` for the per-iteration train NLL
  and epoch-end validation NLL.
- `embedders/one_hot/run/metrics.jsonl` for exact epoch summaries.
- `embedders/one_hot/run/eval.json` for train/validation/test NLL and
  perplexity from the best checkpoint.
- `embedders/one_hot/analysis/metrics/` for the five 800-row analysis tables.

The full `one_hot` run generated 40,000 designs (eight targets, 5,000 each),
all with status `ok`. The raw design CSVs are intentionally excluded from Git.
The balanced 800-design scoring cohort is included.
