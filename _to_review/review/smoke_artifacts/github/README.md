# Smoke-run review artifacts

This directory is a lightweight, self-contained export of the successful
seven-embedder smoke run. It intentionally excludes pretrained assets,
embeddings, Conda environments, and model checkpoints.

- `summary/` contains the strict completion manifest, incremental report, and
  GPU preflight.
- `figures/` and `figure_data/` contain the four cross-model analysis plots and
  their source CSVs.
- `embedders/<name>/` contains evaluation results, test predictions,
  train/validation NLL curves, generated designs, and all analysis metrics.

The full local bundle one directory above also links to saved checkpoints and
canonical artifacts, but those large files are not committed to Git.
