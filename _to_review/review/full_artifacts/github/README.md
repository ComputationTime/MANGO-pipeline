# MANGO Full-Run Review Artifacts

This directory is updated one model at a time as each full run completes. It
contains lightweight review outputs only: no checkpoints, pretrained weights,
embedding tensors, or raw 40,000-row design sets.

## Completed models

| Model | Run ID | Epochs | Train NLL (final) | Validation NLL (best/final) | Test NLL | Status |
|---|---|---:|---:|---:|---:|---|
| one_hot | `one_hot__5a3e5f6e` | 10 | 0.4486 | 0.9098 / 0.9098 | 0.9960 | Complete |

For `one_hot`, the ten validation NLL values were 1.5822, 1.2272, 1.0860,
1.0353, 1.0088, 0.9672, 0.9507, 0.9118, 0.9255, and 0.9098. Validation
temporarily regressed at epoch 8, then recovered to a new best at epoch 9;
patience 2 therefore avoided stopping on a single noisy epoch. Compared with
the preserved five-epoch baseline (`one_hot__79bed882`, test NLL 1.0792), the
ten-epoch best checkpoint improves test NLL to 0.9960. Training remained
finite and stable, though the train/validation gap continues to widen.

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
