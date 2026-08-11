# Small-run review artifacts

This directory is a lightweight, self-contained export of the successful
seven-embedder small run. It intentionally excludes pretrained assets,
embeddings, Conda environments, and model checkpoints.

- `summary/` contains the strict completion manifest, results report, and GPU
  preflight.
- `figures/` and `figure_data/` contain the four cross-model analysis plots and
  their source CSVs.
- `embedders/<name>/` contains evaluation results, test predictions,
  train/validation NLL curves, generated designs, and all five analysis metric
  tables.

The full local bundle one directory above links to the saved checkpoints and
canonical artifacts. Those large files are not included here or committed to
Git.
