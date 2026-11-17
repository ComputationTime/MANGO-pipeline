# Exploratory four-arm affinity-ranking experiment

Run the plumbing check and then the pilot:

```bash
./run_gpu.sh affinity-ranking-pilot-smoke
./run_gpu.sh affinity-ranking-pilot
```

The pilot writes to `artifacts/experiments/affinity_ranking_pilot_v1/`.
Configuration is in `config/affinity_ranking_pilot.yaml`. Original protected
cohorts, confirmatory protocols, and prior results remain unchanged.

## Question and controlled intervention

Does a measured-affinity ranking objective improve MANGO's affinity ranking
more than it improves an unconditional scratch IgLM control?

The four arms are IgLM CE, MANGO CE, IgLM CE plus ranking, and MANGO CE plus
ranking. Five paired optimization seeds are fixed before fitting. Each model
family first receives 1,000 native CE updates from scratch; the resulting
checkpoint is shared by that family's CE and ranking adaptation arms. Both
adaptation arms reset Adam identically and receive 300 updates with exactly
the same native examples, affinity-pair examples, dropout seed, CE weights,
and target-token exposure. Final checkpoints are always used; there is no
early stopping or model selection based on any evaluation metric.

The native CE minibatch contains four examples. Every adaptation batch also
contains the same strong and weak RNase-A VHH variants. CE is 0.75 times the
mean native-example CE plus 0.25 times the mean pair-example CE. Ranking adds
0.1 times `softplus(NLL_strong - NLL_weak)`. The CE control computes the same
forward passes with a zero ranking coefficient. This avoids attributing
extra sequence exposure to the ranking objective. Group-balanced training
schedules are frozen before outcomes are computed.

MANGO uses frozen pretrained ESM2 650M antigen embeddings, fixed 32-token mean
pooling, and cross-attention. The IgLM control has a wider feed-forward layer
to match total parameters within 0.1%. The native decoder initialization is
shared before the extra branch/width is added; extra unconditional projection
weights start at zero so they do not change the initial decoder function.
These models differ in architecture and effective capacity, so the factorial
contrast describes an objective-by-model-family interaction. It does not by
itself prove that antigen information causes the difference.

## Data and explicit exploratory scope

Use only the 227 previously protected native records, keeping every existing
reserved component held out. Deduplication of antigen/chain/target sequences
leaves 99 training records (33 groups) and 36 test records (16 groups).
No original source group or full protection component crosses splits.
No prior holdout enters training. Antibody sequences are autoregressive
targets; conditioning contains only antigen information and the H/L selector.

The affinity intervention uses one same-assay, same-parent VHH pair from the
[2007 RNase-A study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2148503/):
wild type at 23 nM and clone a at 0.18 nM in yeast display. Both variable-domain
sequences already occur in the protected training cohort. Common assay tags
and termini are not generated. The publication's Table 1 value of 0.15 nM
for clone a is retained as a discrepancy; either value gives the same binary
ordering and exceeds the 100-fold separation used here. Published +/- values
have unresolved error definitions and replicate counts; they are not imputed
as SD, SEM, or confidence intervals.

This is a new exploratory protocol. It does **not** satisfy the old strict
affinity-label gate or its requirement for 100 heavy test examples. The old
confirmatory study remains infeasible. The smaller existing holdout is used
only to describe native-sequence modeling trade-offs. One affinity-training
parent cannot establish a general affinity-training rule.

The external endpoint uses the identical frozen 1,000-row selections for each
of HER2, PD1 and TIGIT used previously. These panels were already inspected,
lack independent parent IDs, and remain excluded from training. No new test
set, independent confirmation, or biological confidence interval is claimed.

Two initially considered FLAb DMS tables (G6.31 and D44.1) were not used for
training: their primary papers describe enrichment-based landscapes. A
downstream column labeled KD is insufficient evidence of a measured KD for
every variant. Downloaded source files remain under `artifacts/data/`.

## Endpoints and checks

The predeclared primary endpoint is the equal-target macro Spearman interaction:

```text
(MANGO_rank - MANGO_CE) - (IgLM_rank - IgLM_CE)
```

All coefficients correlate negative NLL with negative supplied log10 KD.
Also report each target and the MANGO-minus-IgLM gap under both objectives.
The in-sample RNase-A preference gap shows whether the intervention learned
its training pair; it is not held-out affinity evidence. Held-out native token
NLL detects damage to general sequence modeling.

Every error bar is sample SD across paired optimization seeds. No row
bootstrap, biological CI, significance test, or automatic advancement is
performed. Smoke metrics are plumbing checks only.

The workflow verifies source/code hashes, group isolation, pair membership,
CE/ranking checkpoint and exposure matching, total parameter matching,
scalar-versus-batched likelihood parity for both H/L selectors, and antigen
invariance of the unconditional control. GPU jobs share `gpu=1` and run
sequentially in isolated environments. Each fit flushes a training-curve CSV
every update and refreshes its plot every 100 updates and at completion.

Independent post-run validation recomputes all 120 target-level correlations
using SciPy, checks exact schedule/token exposure and the logged ranking
coefficient, and draws the paired interaction plot:

```bash
.snakemake/conda-gpu/39ded49e05920e30bca84326296484a5_/bin/python \
  workflow/scripts/validate_affinity_ranking_pilot.py
```

The interpreter path above is this host's existing isolated analysis
environment. Another host should run the validator inside the environment
defined by `workflow/envs/openalphaseq_data.yaml`.
