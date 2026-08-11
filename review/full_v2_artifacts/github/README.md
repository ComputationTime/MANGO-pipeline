# MANGO Full-v2 Review Artifacts

Full-v2 replaces the fixed-order full baseline with reproducible epoch-wise
training shuffling. Training uses `seed + epoch`; validation order remains
fixed. The lightweight files in this directory are intended for GitHub review.
Checkpoints, pretrained weights, embedding tensors, and raw 40,000-row design
sets remain in canonical local storage and are intentionally excluded here.

## Completed models

| Model | Run ID | Epochs run | Best epoch | Final train NLL | Best validation NLL | Test NLL | Status |
|---|---|---:|---:|---:|---:|---:|---|
| one_hot | `one_hot__a288de4f` | 7 | 4 | 0.3922 | 0.9096 | 1.0063 | Complete |

One-hot validation NLL was 1.2440, 1.0576, 0.9939, 0.9107, 0.9096,
0.9303, and 0.9214. Patience 2 stopped after epoch 6 and retained epoch 4 as
the best checkpoint. Every executed epoch contains all 2,636 training records
exactly once, and all seven epoch orders are distinct. The best checkpoint
evaluates to train/validation/test NLL 0.3965/0.9156/1.0063.

Start with `embedders/one_hot/run/training_curve.png`. It includes raw
per-structure train NLL, a rolling train mean, epoch-mean train NLL, and
epoch-end validation NLL. Exact epoch summaries and shuffle seeds are in
`metrics.jsonl`; best-checkpoint evaluation is in `eval.json`.

The run generated 40,000 successful designs across eight held-out targets. A
balanced 800-design cohort and five 800-row `ok` analysis tables are included
under `embedders/one_hot/analysis/`.
