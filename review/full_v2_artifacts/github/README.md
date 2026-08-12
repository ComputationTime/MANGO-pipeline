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
| biopython | `biopython__9cced44a` | 6 | 3 | 0.4408 | 0.9386 | 0.9958 | Complete |
| pyrosetta_pre | `pyrosetta_pre__f5601ee6` | 6 | 3 | 0.3824 | 0.9017 | 0.9816 | Complete |
| esm2 | `esm2__e37a6a17` | 6 | 3 | 0.4203 | 0.8992 | 0.9677 | Complete |

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

Biopython validation NLL was 1.2240, 1.0472, 0.9783, 0.9386, 0.9494, and
0.9628. Patience 2 stopped after epoch 5 and retained epoch 3. Each executed
epoch contains all 2,636 training records exactly once in a distinct order.
Its best checkpoint evaluates to train/validation/test NLL
0.4417/0.9430/0.9958. Biopython also generated 40,000 successful designs and
completed all five 800-row analysis tables; its review files mirror the
one-hot layout under `embedders/biopython/`.

PyRosetta-pre validation NLL was 1.1976, 1.0288, 0.9954, 0.9017, 0.9539,
and 0.9382. Patience 2 stopped after epoch 5 and retained epoch 3. Each of its
six epochs also contains all 2,636 training records exactly once in a distinct
order. Its best checkpoint evaluates to train/validation/test NLL
0.4031/0.9072/0.9816. The official licensed runtime successfully loaded and
mapped all 3,194 study mmCIF records. PyRosetta-pre generated 40,000 successful
designs and completed all five 800-row analysis tables under
`embedders/pyrosetta_pre/`.

ESM2 validation NLL was 1.2291, 1.0304, 0.9430, 0.8992, 0.9321, and
0.9141. Patience 2 stopped after epoch 5 and retained epoch 3. Each executed
epoch contains all 2,636 training records exactly once in a distinct order.
Its best checkpoint evaluates to train/validation/test NLL
0.4305/0.9042/0.9677. ESM2 generated 40,000 successful designs and completed
all five 800-row analysis tables under `embedders/esm2/`.
