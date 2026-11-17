# Reproducing the study

All commands run from the repository root on a single NVIDIA GPU host. Outputs
go to `artifacts/` (figures in `artifacts/analysis/figures/`, each with a
`*_data.csv` holding the plotted numbers).

## Order of operations

1. `./run_gpu.sh smoke`: plumbing check. Fix anything it reports.
2. `./run_gpu.sh smoke-all`: also validates gated ESM3 and PyRosetta PRE
   (needs `HF_TOKEN`). PyRosetta PRE mmCIF chain mapping still needs validating
   here before its scientific results are used.
3. `./run_gpu.sh study`: full eight-embedder study, then figures 1-6.
4. Optional ablations below. Re-running a command resumes missing work.

Every run writes `artifacts/gpu/results_report.{json,csv}`; read it first.
`artifacts/gpu/results_complete.json` exists only when every requested method
finished.

## Figure map

| Figure | File stem in `artifacts/analysis/figures/` | Produced by |
|---|---|---|
| 1. Train/test NLL per antigen representation | `fig1_nll` | `smoke`, `small`, `study` |
| 2. Boltz-2 / Chai-1 structure confidence | `fig2_structure_confidence` | `smoke`, `small`, `study` (bounded cluster-diverse subset) |
| Rosetta interface QC (fixed-backbone `InterfaceAnalyzer`) | `rosetta_interface_plumbing` | same; quality-control only, not the PRE embedder |
| 3. Antibody-likeness (IgLM / AntiBERTy / AbLang2) | `fig3_ablikeness` | `smoke`, `small`, `study` |
| 4. Distance to nearest heavy-chain germline (ANARCI) | `fig4_ld_germline` | `smoke`, `small`, `study` |
| 5. Developability (sequence biophysics; TAP is skipped) | `fig5_developability` | `smoke`, `small`, `study` |
| 6. Heavy-chain V/J gene families | `fig6_gene_families` | `smoke`, `small`, `study` |
| 7. ESM2 size ablation | `fig7_esm2_size_ablation` | `./run_gpu.sh esm2-ablation` |
| 8. Antigen controls (constant, random, pooled, shuffled, cross-attention) | `fig8_antigen_controls` | `./run_gpu.sh antigen-controls` |
| 9. Antigen context-length ablation + IgLM | `fig9_antigen_context_length_ablation` | `./run_gpu.sh antigen-length-ablation`, `iglm-baseline` |

`train-all` trains and evaluates all eight models plus a matched from-scratch
IgLM. `study-esm3`, `study-pyrosetta`, `study-all` (and the `smoke-`/`small-`
forms) select which gated embedders join; `weights[-esm3|-pyrosetta|-all]`
only downloads weights.

FLAb / AlphaSeq benchmark plots are standalone scripts in
`workflow/scripts/analysis/` (`prepare_alphaseq.py`, `score_flab.py`,
`plot_flab.py`, `plot_mango_binding_flab.py`); they have no launcher mode yet.

## What is fixed by design

- Validation splitting is group-aware on `ab_ag_cluster`; a cluster never
  crosses splits.
- The generation cohort is one deterministic representative per held-out
  cluster, `n_per_target` designs each (`config/config.yaml`, `generation`).
- Structure confidence uses the recorded light chain/antigen as reference
  partners for QC only, never as MANGO input.
