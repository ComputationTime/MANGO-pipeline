# Native-cohort mechanism experiments

This is a new exploratory study. Previous ASD protocols and outcomes remain
unchanged. Eligibility is established before any new model fitting. Run the
complete native-cohort audit with `./run_gpu.sh native-mechanism-audit`.

The source is the existing native CATH inventory, with original source groups,
CATH-family links, exact antigen/antibody links and all accepted near-antibody
links retained. All prior native holdout components remain reserved. Complete
components containing external benchmark antigen containment, external antibody
homology or reserved ASD antibody homology are quarantined. Both directional
antigen coverage searches run chain by chain. The validated full GPU antibody
search retains its original thresholds and candidate ceiling.

Each evaluation panel must contain at least 20 components and 100 heavy examples.
The proposed small native pilot requires 150 training records and 30 antigens;
this is a different data source and does not change the 1,000-record minimum in
the frozen ASD studies. No old holdout is trained. An infeasible audit blocks
all native fits and produces an explicit report.

If eligible, five matched seeds (100–104) each train native IgLM from scratch
for 1,000 updates, then start all adaptation cells from that seed's fixed final
checkpoint. Six cells cross decoder freezing versus joint updating with correct,
shuffled or constant 32-token antigen memory. All six cells have identical total
architecture, initially zero cross-attention outputs and identical initial shared
tensors. Pairing comparisons within a freezing condition have identical trainable
parameter counts, target order, tokens, ordinary CE objective, learning rate and
dropout. The freezing intervention deliberately changes which parameters update.
An unconditional continuation is reported separately. All fits use a fixed
500-update adaptation budget; no held-out outcome chooses a checkpoint.

The primary mechanistic contrasts are shuffled-minus-correct heavy NLL in each
freezing condition and their interaction. Constant-minus-correct separates
antigen content from the added branch. A separate ordinary-CE versus CE-plus-
pairing comparison includes the corresponding shuffled-pair control. It measures
the interaction between objective and real pairing, rather than combining
unconfirmed improvements. A parameter-matched unconditional scratch control is
required before claiming superiority over IgLM. Native IgLM remains a reference.

Plots must show all seed effects and seed SD separately from biological
uncertainty. Biological component bootstrap intervals condition on the five
realized seeds; they do not estimate population uncertainty over initialization.
No small-seed percentile interval supplies a confirmation p-value. This native
source has already been inspected and cannot provide pristine confirmation.

HER2/PD1/TIGIT scoring is exploratory, on a sequence-hash-selected panel frozen
without reading affinities. Report per-target Spearman correlation against
negative log10 Kd using negative NLL, plus perplexity-versus-log10 Kd Pearson.
Only optimization-seed SD is available without independent parental labels;
never label it a biological confidence interval. A verified independent affinity
table is required for affinity-supervised ranking; absence blocks that experiment.

## Resumed fixed-budget pilot v2

Use `./run_gpu.sh native-mechanism-smoke`, then
`./run_gpu.sh native-mechanism-study`. Both use
`config/native_mechanism_pilot_v2.yaml`; smoke has its own root and only tests
software with real cached examples and four updates per stage. Smoke results
have no scientific interpretation.

The saved v1 graph and directional antigen searches were checked against their
frozen inputs, hashes and completed control flow. Before antibody exclusions,
only 298 records in 52 components remain. This is already below the two-panel
design's minimum of 350 records; its interrupted external alignment is never
treated as completed evidence. The new audit searches only these remaining
components against the complete external/reserved databases, retaining all
original cutoffs. No accepted graph edges are dropped.

V2 was specified before new model fitting as an exploratory single-panel pilot.
It uses fixed final checkpoints, no validation-based hyperparameter/checkpoint
selection, no automatic extensions and no confirmation. It retains the minimum
of 20 evaluation components, 100 evaluation heavy examples, 150 training records
and 30 training antigens. Complete components are assigned; old holdouts remain
reserved. If either the cohort or a complete deranged schedule is infeasible,
the pilot reports the obstruction without fitting scientific models.

The completed v2 audit found 227 protected records in 49 components, below even
the one-panel minimum of 250. Scientific training is therefore blocked. The
runner and reporting remain useful for a future separately frozen eligible
cohort; smoke runs do not override this outcome.

There are 14 declared fits per seed: native and parameter-matched wider scratch
sources, their continuations, six freezing-by-memory cells, two pairing-objective
cells, and paired/shuffled one-hot antigen cells. One-hot pairing is isolated
within its own matched architecture; an across-representation difference also
changes input width and parameter count. The wider unconditional comparator
matches the joint ESM2 branch's trained parameter count within 0.1%; frozen ESM2
pretraining and extraction compute are additional and are not parameter matched.
Frozen branches are compared causally against their identical frozen constant
and shuffled controls. The frozen-branch versus native-continuation contrast is
composite and cannot isolate freezing alone.

`paired_effects.png` separates seed SD from biological component bootstrap
intervals conditional on the five realized seeds. `affinity_correlations.png`
shows seed SD only; absent parent-lineage labels prevent biological affinity
intervals. `uncertainty.json` stores the underlying paired estimates. No p-values,
formal confirmation, or inferred generated-antibody binding gains are reported.
`results_report.json` and `findings.md` distinguish data infeasibility from
execution failure. `feasibility.png` displays the actual count obstruction.
