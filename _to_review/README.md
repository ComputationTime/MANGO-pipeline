# _to_review

Items quarantined by the cleanup (see `../CLEANUP_PLAN.md`, Phase 1). Nothing was
deleted. Review, then `git rm -r _to_review` when you are happy.

| Path | What it is |
|---|---|
| `review/` | Tracked snapshot of old generated review bundles (smoke/small/full/full_v2). `run_gpu.sh` regenerates `review/` locally; it is now gitignored. |
| `MANGO_BINDING_DATA/` | 84 MB of AlphaSeq CSVs + early scoring/plot scripts (untracked until now). Referenced by `config/mango_binding_affinity.yaml` (`source_dir`); update that path if the study is re-run. |
| `docs/site/`, `docs/mango_diagnostic_workflow.html` | Old beginner HTML guide. |
| `legacy/MANGORunner.py` | Retired pre-workflow runner (raised an error on use). |
| `legacy/embed_antigen_afm.py`, `legacy/embed_afm.yaml` | AF-M stub embedder; its rule and config entry were removed. |
| `legacy/install_mango.sh` | Compatibility wrapper that only called `run_gpu.sh`. (The side-study helper scripts went to `experiments/`.) |

`artifacts/` was **untracked** (`git rm --cached`) but left on disk.

## Added in Phase 3 (dead code)

| Path | Why |
|---|---|
| `legacy/rules/analysis.smk` | Self-declared legacy prototype, never included by the Snakefile. |
| `legacy/scripts/analysis_{plots,complex_metrics,predict_structures,seq_metrics}.py` | Only used by the dead `analysis.smk`. |
| `legacy/scripts/{generate_all_plots,watch_training_completion}.py` | No references anywhere. |
| `legacy/envs/structure_prediction.yaml` | Only used by the dead `analysis.smk`. |
