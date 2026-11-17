# MANGO cleanup plan

Status: **draft, agreed scope, no code moved yet.**
Written so it can be resumed by Claude, Codex or a human. Work through the
phases in order and tick the boxes. Each phase should be its own commit.

## Decisions already made (from the planning Q&A)

| Topic | Decision |
|---|---|
| Side-study workflows (ASD, AbRank, OpenAlphaSeq pairing/affinity, affinity studies, native mechanism, controlled antigen, interface attention/supervision, epitope/antigen diagnostics, short affinity, IgLM baseline, ...) | **Move to `experiments/`**. Not deleted. Removed from the default workflow, with their own README. |
| Supported core scope | All 8 embedders; Boltz-2/Chai-1 structure confidence + Rosetta interface QC; Figures 1-5 analysis stack; Fig 6/8 plots, FLAb and AlphaSeq plots. |
| Large tracked outputs and legacy code (`artifacts/` tracked files, `review/`, `MANGO_BINDING_DATA/`, `docs/site`, `mango/MANGORunner.py`, AF-M stub, old shell scripts) | **Move all into one folder, `_to_review/`**, so the owner can inspect and delete it by hand. Nothing is deleted by the agent. |
| Audience | Researchers **reproducing the paper**: one command, short README, documented config. Extensibility is out of scope for now. |

## Invariants to preserve (from AGENTS.md)

- Antigen-only conditioning plus a learned heavy/light selector.
- Group-aware validation split on `ab_ag_cluster`.
- Per-rule isolated Conda envs; GPU rules serialized (`gpu=1`).
- One Snakemake invocation per embedder in `run_gpu.sh`, so one failure does not block the rest.
- Keep nested `.git`. Never commit tokens. `.snakemake/` and `artifacts/` stay untracked.
- `./run_gpu.sh smoke`, `smoke-all` and `study` must keep working after every phase.

## Phase 0: safety net (do first)

- [ ] Commit or stash the current uncommitted changes (about 60 modified files, including `AGENTS.md`, configs, `run_gpu.sh`, rules). Do not mix them with cleanup commits.
- [ ] `git tag pre-cleanup` and create branch `archive/experiments` at the current HEAD. Everything moved or removed stays recoverable.
- [ ] Record the baseline: run `bash -n run_gpu.sh`, `python -m unittest discover -s tests`, `snakemake --lint` and `snakemake --sdm conda --cores 1 --dry-run`. Save the output in the scratchpad. Later phases must not regress it.
- [ ] Run `./run_gpu.sh smoke` once (needs GPU access outside the sandbox, with `MANGO_EXECUTION_CONFIG=config/local_12gb.yaml`) so there is a known-good reference.

## Phase 1: quarantine non-code clutter into `_to_review/`

Use `git mv` for tracked files so history follows. Keep `_to_review/` tracked
for one commit, so the owner can see the move. The owner later deletes it, or
we add it to `.gitignore` and `git rm -r --cached` it.

- [ ] `review/` (about 250 tracked files: `full_artifacts`, `full_v2_artifacts`, `smoke_artifacts`, `small_artifacts`, `structure_confidence_sample`) → `_to_review/review/`
- [ ] Tracked files under `artifacts/` (706 files; the dir is in `.gitignore` but files were committed) → `git rm --cached -r artifacts/`. Files stay on disk, since they are 317 GB of real outputs. Do **not** move them. This is the one exception to "move, don't delete", because it only changes tracking.
- [ ] `MANGO_BINDING_DATA/` (84 MB, `0_grab_alphaseq.py`, `1_score_w_iglm.py`, PNGs, `.out` logs) → `_to_review/MANGO_BINDING_DATA/`. Check first that `plot_mango_binding_flab.py` and `prepare_alphaseq.py` do not read from it (see open questions).
- [ ] `docs/site/` (184 K beginner HTML guide) and `docs/mango_diagnostic_workflow.html` → `_to_review/docs/`
- [ ] Legacy code: `mango/MANGORunner.py` and any other file in `mango/` not imported by `workflow/` → `_to_review/legacy/`. Run `grep -rn "from mango\|import mango" workflow tests` first to find what is still used. `mango/utils` and `mango/model` are the "model components used by the workflow", so they stay.
- [ ] `embed_antigen_afm.py` + `workflow/envs/embed_afm.yaml` (AF-M stub) → `_to_review/legacy/`
- [ ] Old shell scripts: `download_asd.sh`, `download_openalphaseq.sh`, `run_interface_attention.sh`, `install_mango.sh` (check that `install_mango.sh` is not what `setup.py` or the docs point to) → `_to_review/legacy/`
- [ ] Junk: `workflow/.DS_Store`, all `__pycache__/`, `tests/__pycache__`. Make sure `.gitignore` covers `.DS_Store` and `__pycache__`.
- [ ] Add `_to_review/README.md` listing what is here and why, with one line per item.

## Phase 2: move side studies into `experiments/`

Target layout:

```text
experiments/
  README.md                  # what these are, "not supported, not run by CI"
  <study_name>/
    <study>.smk              # was workflow/<study>.smk
    config.yaml              # was config/<study>.yaml
    README.md                # was docs/<study>.md
    scripts/                 # was workflow/scripts/*
    tests/                   # was tests/test_*
```

Studies to move. Each has a `workflow/*.smk`, `config/*.yaml`, `docs/*.md`,
scripts and tests with a matching prefix:

| Study | Matching files (prefix) |
|---|---|
| asd (scaling, 8k, 16k, fixed_positions, transfer, prototype, prototype_iglm, loss_comparison) | `asd_*`, `prepare_asd_*`, `report_asd_*`, `download_asd.py`, `plot_asd_*` |
| abrank_transfer | `abrank_*`, `prepare_abrank_*`, `number_abrank_*`, `report_abrank_*`, `plot_abrank_*` |
| openalphaseq (pairing, affinity) | `*openalphaseq*`, `fold_openalphaseq_antigens.py` |
| affinity studies (bottleneck audit, improvement, ranking pilot, lineage review, short affinity, accelerated) | `affinity_*`, `*_affinity_*`, `short_affinity_*`, `accelerated_affinity_search.py`, `audit_affinity_bottleneck.py`, `run_short_affinity*.py`, `config/affinity_primary_ledger_v1.json` |
| native_mechanism (+ pilot, smoke, schedules, CATH inventory) | `native_mechanism_*`, `prepare_native_*`, `inventory_native_cath*.py`, `prepare_cath_sequences.py`, `workflow/envs/cath_split.yaml` |
| controlled_antigen | `controlled_antigen_*`, `run_controlled_antigen.py`, `report_controlled_antigen.py` |
| interface_attention / interface_supervision | `interface_*`, `plot_interface_geometry_usage.py` |
| epitope / antigen diagnostics | `epitope_diagnostics*`, `antigen_diagnostic*`, `audit_epitope_benchmark.py`, `audit_antigen_containment.py`, `summarize_antigen_diagnostics.py`, `train_antigen_diagnostic.py` |
| homology / witness | `homology_backend.smk`, `*witness*`, `benchmark_homology_backend.py` |
| iglm_baseline, legacy_adapter_affinity, esm2_ablation, antigen_length_ablation | `iglm_baseline.py`, `legacy_adapter_affinity.py`, `plot_esm2_ablation.py`, `plot_antigen_length_ablation.py` |
| misc | `calibrate_seed_intervals.py`, `review_on_study_exit.py`, `collect_review_artifacts.py`, `embedding_benchmark.py`, `time_command.py` (verify each; they may be core) |

Steps:

- [ ] Build an **import and rule dependency map first**. For every script under `workflow/scripts/`, find who references it (`grep -rn <name> workflow config run_gpu.sh tests`). A script referenced only by one side-study goes with that study. A script shared with the core (e.g. `embed_common.py`, `model_common.py`, `device_common.py`, `embedding_cache.py`) stays in `workflow/scripts/`.
- [ ] Move study by study. After each study, run `snakemake --lint` and the dry run for the core. Make sure nothing in `workflow/Snakefile`, `rules/*.smk` or `run_gpu.sh` still refers to a moved file.
- [ ] Each moved study keeps working **if** run from `experiments/<study>` with a documented command. Do not promise more than that. Put a banner at the top of `experiments/README.md`: *unsupported, may be stale, history is on branch `archive/experiments`*.
- [ ] Remove now-empty sections from `config/` and `workflow/envs/` (post-deploy scripts and YAMLs used only by moved studies move along with them).

## Phase 3: simplify the core

After Phases 1 and 2 the core should be roughly: `fetch, standardize, process, embed, model, iglm, gpu, analysis/*, targets`.

- [ ] **Configs**: reduce `config/` to `config.yaml` plus a small set of mode overlays. Current modes: `smoke`, `small`, `gpu` (study), `gpu_all`, `gpu_esm3`, `gpu_pyrosetta`, `local_12gb`, `structure`. Decide whether `gpu_esm3`/`gpu_pyrosetta`/`gpu_all` can become flags of `run_gpu.sh` rather than separate files. Document every key in `config.yaml` with a comment.
- [ ] **`run_gpu.sh` (746 lines)**: the supported entry point. Split into `run_gpu.sh` (thin: argument parsing, mode table) plus `workflow/scripts/launcher/` helpers if needed. Add `./run_gpu.sh --help` listing every mode (`smoke`, `smoke-all`, `small`, `study`, `*-esm3`, `*-pyrosetta`, `*-all`). Preserve the per-embedder isolated Snakemake invocations.
- [ ] **`rules/embed.smk` (642 lines) and `rules/common.smk` (465 lines)**: look for repeated per-embedder rule boilerplate that can be generated from one table (embedder name → env, script, gpu flag, gated flag). Keep one rule per embedder if generating them obscures the DAG; only dedupe when it is a clear win.
- [ ] **Duplicate rule files**: `rules/analysis.smk` and `rules/analysis/*.smk` both exist. Check whether `analysis.smk` is dead (the Snakefile only includes `rules/analysis/*`).
- [ ] **Envs**: `workflow/envs/analysis.yaml`, `analysis_core.yaml`, `analysis_plot.yaml` and `structure_prediction.yaml` vs `structure_boltz.yaml`/`structure_chai.yaml` look overlapping. Remove unused ones after confirming with `grep -rn "envs/" workflow`.
- [ ] **Scripts**: group `workflow/scripts/` into subfolders (`embed/`, `model/`, `analysis/`, `launcher/`), keeping `analysis/` as is. Update `script:`/`shell:` paths in rules.
- [ ] **Tests**: keep only tests for the core (`test_chain_task`, `test_cluster_validation`, `test_embedder_adapters`, `test_embedding_cache`, `test_gpu_batch`, `test_gpu_report`, `test_model_*`, `test_training_*`, `test_standardize_sabdab2`, `test_structure_confidence`, `test_analysis_modules`, `test_flab`, `test_proteinmpnn_chain_aliases`, `test_esm3_structure`, `test_run_identity`, `test_training_curve`, and similar). Study tests move with the study. Verify the suite passes inside its Snakemake env.
- [ ] **Dead code**: run `vulture` or a grep-based sweep on `workflow/scripts` and `mango/utils` after the moves. Report candidates. Do not auto-delete.

## Phase 4: make it usable by others (reproduction path)

- [ ] **README rewrite** (target under 150 lines): 1) what MANGO is (one paragraph, one diagram); 2) requirements (NVIDIA GPU, disk, RAM, Conda, HF token for ESM3, PyRosetta licence note); 3) three commands: `./run_gpu.sh smoke`, `smoke-all`, `study`; 4) where outputs land and how to read `results_report.csv/json`; 5) how to resume after failure; 6) pointers to deeper docs.
- [ ] `docs/` trimmed to: `cloud_gpu_setup.md`, `module_signatures.md` (if still accurate), one `architecture.md`, one `config_reference.md`, `reproducing_the_paper.md` (map each figure to the command and output file), `troubleshooting.md` (gated models, Conda solve failures, GPU OOM and `local_12gb.yaml`).
- [ ] **`AGENTS.md`**: update to match the new layout (point at `experiments/` and `_to_review/`), keep the invariants and operating procedure.
- [ ] Add `LICENSE` check, `CITATION.cff` (if a paper exists), and a note on third-party licences (ESM3, PyRosetta, Boltz, Chai).
- [ ] `setup.py`: decide whether it is still needed. The workflow does not need a pip install, so remove it or reduce it to the model package.
- [ ] Add a CI-friendly check script (`scripts/check.sh`) running `bash -n`, unit tests, `snakemake --lint` and the dry run. Optional: a GitHub Actions workflow running it on CPU only.
- [ ] `preflight`: confirm `workflow/scripts/gpu_preflight.py` gives clear errors (missing `nvidia-smi`, no `HF_TOKEN`, low disk) and links to the troubleshooting doc.

## Phase 5: verify and hand over

- [ ] `./run_gpu.sh smoke` and `./run_gpu.sh smoke-all` pass, with outputs equivalent to the Phase 0 baseline. (`study` only after both smoke results are reported to the owner, as AGENTS.md requires.)
- [ ] `git status` clean, `_to_review/` and `experiments/` reviewed by the owner.
- [ ] Final report: files moved, files untracked, lines removed, tests before/after.

## Open questions to resolve during execution

1. **Fig 8 (antigen controls) vs controlled_antigen study**: `plot_antigen_controls.py`, `embed_antigen_control.py`, `shuffle_antigen_embeddings.py`, `pool_antigen_embeddings.py` and `config/antigen_controls.yaml` look like they serve Fig 8 (core) but share names with the `controlled_antigen` study (moved). Decide per file after the dependency map. Default: keep what Fig 8 needs in the core.
2. **FLAb / AlphaSeq plots**: `prepare_alphaseq.py`, `score_flab.py`, `plot_flab.py` and `plot_mango_binding_flab.py` are in scope, but `mango_binding_affinity` (workflow, config, docs, `prepare_`/`report_`/`reuse_mango_binding_cache.py`) and `MANGO_BINDING_DATA/` may feed them. Check whether the data-prep stays core or moves with the study.
3. **`iglm.smk` and `score_iglm.py`** are core (IgLM scoring), while `iglm_baseline.py` and `asd_prototype_iglm` are studies. Confirm the split.
4. **ANARCI/AbLang2/AntiBERTy post-deploy scripts**: keep, since they are part of the Fig 1-5 stack.
5. **Current uncommitted changes**: do they belong to the core, or to a study that is about to be moved? Resolve in Phase 0.

## Suggested prompt to resume

> Read `CLEANUP_PLAN.md` and `AGENTS.md`. We are at Phase N. Do only that phase,
> commit it separately, run the validation commands, and report what you moved
> and anything from "Open questions" that you had to decide.
