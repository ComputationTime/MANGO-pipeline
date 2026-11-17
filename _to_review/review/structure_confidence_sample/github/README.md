# Boltz-2 and Chai-1 structure-confidence plumbing sample

This directory contains the tracked review copy of the first successful
sequence-to-structure-confidence-to-figure test. It verifies that the MANGO
workflow can take generated heavy-chain sequences, combine each with the
cognate light chain and all antigen chains, run both local structure
predictors, normalize their confidence outputs, and render Figure 2.

## Contents

- `figures/fig2_structure_confidence_sample.png` — comparison plot rendered
  from the normalized Boltz-2 and Chai-1 outputs.
- `figure_data/fig2_structure_confidence_sample_data.csv` — exact rows consumed
  by the plotting script.
- `predictions/boltz2_confidence.csv` — normalized Boltz-2 predictions.
- `predictions/chai_confidence.csv` — normalized Chai-1 predictions.
- `figures/rosetta_interface_plumbing.png` — minimal paired fixed-backbone
  Rosetta interface-energy plot for the same structures.
- `figure_data/rosetta_interface_plumbing_data.csv` — exact rows consumed by
  the Rosetta plot.
- `predictions/boltz2_rosetta.csv` and `predictions/chai_rosetta.csv` —
  normalized Rosetta `InterfaceAnalyzer` outputs.

## Scope and interpretation

This is a plumbing check, not a scientific result. It contains two generated
heavy chains from the `one_hot` smoke model and one held-out target
(`pdb_00005otj_B_A`). Each predictor used one diffusion sample per generated
chain. The sample is too small for statistical inference or comparison of the
predictors.

The Rosetta values score raw predictor coordinates without relaxation. Their
large positive energies indicate clashes and must not be interpreted as
binding-affinity estimates; these files demonstrate plumbing and coordinate
quality control only.

The CSV files retain the generated sequences, target and design identifiers,
predictor confidence metrics, status, and original artifact paths for audit
provenance. The large predicted mmCIF structures and model caches remain under
the ignored `artifacts/` tree and are intentionally not committed.

The default study configuration now uses a broader paired pilot of three
generated chains from each of ten independently held-out antigen clusters per
trained model and predictor. Fixed-backbone Rosetta interface scoring and its
paired quality-control plot now run after both predictors in smoke and study.
