# Short MANGO affinity tests v3 — proposed protocol

This is an experimental design, not an execution record. No jobs are launched
by writing this document. Existing v1 and v2 protocols and outputs remain
frozen. Implementation would use a new `artifacts/experiments/affinity_short_v3/`
root, frozen inputs, predeclared arms, and a completion-triggered review.

The question is whether a specific intervention improves antigen-dependent
modeling or measured affinity ranking over an equally supervised unconditional
model. Improved performance on the training set is not evidence of that gain.

## Short-test budgets and common controls

- One GPU, sequential jobs. Training fits use 500 optimizer updates, microbatch
  one chain, seeds 0/1/2, and a 90-second GPU wall-time limit per fit. A timeout
  is an incomplete arm; do not compare it against completed 500-update arms.
  Use the fixed final checkpoint rather than selecting the best checkpoint on
  the evaluation panel. A 250-update snapshot can describe learning progress;
  it does not decide selection or extend the budget.
- The complete initial screen has a 90 GPU-minute ceiling: up to 15 minutes
  for bounded feature preparation, five for existing-checkpoint diagnostics,
  up to 67.5 for 45 training fits, with the remainder reserved for overhead.
  These are ceilings,
  not measured runtime predictions. CPU audit has a separate 15-minute ceiling.
  Exhausted preparation produces an incomplete result, not a smaller silently
  substituted cohort. Missing independent affinity labels skip the ranking arm.
- All conditional models and the additional U-native control use the same native
  IgLM decoder/vocabulary and exact shared starting tensors within each seed.
  The primary U controls widen only the IgLM feed-forward layers to match their
  conditional counterpart's parameter count within 0.1%. Shared tensor slices
  retain native initial values; new feed-forward output weights start at zero.
  U-L separately matches the learned resampler's additional parameters. These
  parameter controls implement the user's subsequent fair-capacity requirement
  and add six fits. All controls are trained from scratch. Do not
  warm-start from v1: newly reserved groups may have entered those checkpoints.
  Conditional branches have matched initial tensors across their arms, and
  start with zero-output cross attention. Use the existing v2 architecture
  implementation in a separately versioned runner; do not modify frozen files.
- Every new fit starts from its seed's initial state. Objective and pooling
  interventions are compared separately against the common baseline, never
  added cumulatively during this screen. Joint decoder updating and dropout
  policy are fixed across all fits; freezing is deferred to a later test.
- Fixed learning rate 1e-4, optimizer, chain order, target examples, selectors,
  ordinary CE reduction and EOS definition. Antigen memory has 32 tokens.
  H/L examples follow the audited masks. Sampling order is selected once from
  training data, limits dominant-group exposure, and is shared by every arm.
  Freeze that order before fitting; record membership and target-token hashes.
- Only antigen features and the requested chain selector condition generation.
  Antibody sequences are autoregressive targets. Region annotations and affinity
  values supply training losses, never conditioning features. No oracle epitope,
  native antibody framework, or target-derived antibody embedding is supplied.
- Record optimizer updates, target tokens, forward calls, GPU time, memory and
  trainable parameters. Pairing/ranking objectives require additional forwards;
  matching target exposure does not make compute identical. Report that cost
  explicitly; a promising result subsequently needs a compute-matched check.

## Test 1: Can a larger protected cohort support the experiments?

Run an audit only, without model training. Compare the inherited 512-residue
eligibility restriction with 1,024 and 2,048. This is a data feasibility check,
not an accuracy ablation. The largest eligible pool defines the homology graph
before any length-based subset is inspected, so graph labels do not change
opportunistically across candidate limits. Select the smallest candidate that
meets the frozen feasibility criteria below, using no model outcomes or affinity
values. If more than one works, record the alternatives but test only that one.

A read-only count during design found the following before homology screening:

| Antigen-length ceiling | Training rows | Original training source groups |
| --- | ---: | ---: |
| 512 | 3,694 | 69 |
| 1,024 | 73,395 | 83 |
| 2,048 | 74,161 | 87 |

These counts already exclude the existing BUZZ/AlphaSeq/FLAb source tags. They
do not include the subsequent external/reserved homology exclusions and are
not independent-component counts. The 4,096 ceiling gave no further eligible
records in this source and is not included as another candidate.

Retain all external antigen/antibody protection, original holdout restrictions,
`ab_ag_cluster`, CATH-family grouping and antibody-lineage links. Never move an
old validation/test record into training. Preserve whole components. Training
requires at least 1,000 eligible records and 100 antigens; reserve distinct
screening and confirmation panels with at least 20 components and 100 heavy
examples each. Use bounded component-aware selections and audit their masks
after deduplication. Thresholds do not change in response to the result.

For long sequences, prepare frozen features using bounded per-chain windows,
with chain breaks and residue-to-window mapping recorded. Never simply take the
first 512 residues. Use the same feature construction in every training arm.
Feature preparation is bounded to the selected training/evaluation cohort, with
cache reuse. Wider encoding may restore records without restoring independent
components: if the audit remains infeasible, stop Tests 3–6 and still report
Test 2. Do not repeat the previous training launcher on the unchanged pool.

## Test 2: Does the current model contain useful antigen-dependent information?

No weight updates. Use the frozen v1 MANGO checkpoints for seeds 0/1/2 and their
scratch-IgLM references. Select at most 1,000 reliable pairs per HER2/PD1/TIGIT
by a fixed sequence-hash rule without using affinity values. Save that selection
before scoring. All candidates receive scores under all three antigens.

Compare ordinary full-chain likelihood, CDR3-only likelihood, and a conditional
residual: mean NLL under the two alternative antigens minus NLL under the
measured target. Region scores use target-token counts, and the same candidate
and position mask are used on both sides of each intervention. Alternative
antigens are controls, not asserted experimentally confirmed non-binders.

IgLM must be invariant to antigen identity within numerical tolerance. Its
residual is identically zero, so its residual correlation is undefined; compare
candidate ranking against its ordinary score. Apply evaluation and inference
mode and the existing scalar/cached likelihood checks. The v1 architecture
difference prevents attributing model-family differences solely to conditioning.

For affinity evaluation use larger-is-better scores: negative NLL, or the
conditional residual, versus negative supplied log10(Kd). Primary diagnostic
metric is per-target Spearman correlation; retain the original perplexity versus
log10(Kd) Pearson coefficient as a secondary continuity measure. Report paired
seed differences and target-specific results. Do not pool targets to obtain a
larger apparent sample.

A mean Spearman increase of at least 0.03 over ordinary MANGO and over ordinary
IgLM, with positive paired differences in at least two of three seeds, flags a
scoring idea for independent follow-up. This is an exploratory flag only: these
external assays have already been inspected and lack independent lineage labels.
No row bootstrap, biological CI, formal claim, or new training selection is
based on this result. These assays remain excluded from training in Tests 3–6.

## Test 3: Does a pairing objective add useful information?

After Test 1 succeeds, fit these arms on the same frozen 500-update schedule:

| Arm | Training memory | Objective |
| --- | --- | --- |
| U-native | None; native scratch IgLM | Ordinary chain CE |
| U | None; parameter-matched scratch IgLM | Ordinary chain CE |
| C | Correct antigen | Ordinary chain CE |
| S | Deranged training antigen | Ordinary chain CE |
| P | Correct antigen | CE plus pairing contrast |
| PS | Deranged training antigen | CE plus the identical pairing contrast |

Six arms times three seeds gives 18 fits. In P and PS, use one alternative
training antigen for each target example and compare length-normalized sequence
log-likelihoods with a softplus contrast. The pairing term's fixed coefficient
is 0.1; do not tune it on the evaluation panel. The PS control uses the same
loss and computation, with the biological pairing destroyed. Shuffling preserves
antigen marginals separately for each chain selector. Exclude known positive
pairings from decoy eligibility. Unknown pairings remain putative negatives;
report that limitation, and retain a CE term to limit damage from false negatives.

Evaluate every conditional arm with the correct antigen, two fixed alternative
antigens, and constant memory. Primary training-screen endpoint is correct-
antigen heavy CDR3 NLL. Also report full-heavy NLL and the paired alternative-
minus-correct NLL gap. Increasing only the alternative-antigen loss is not enough
to advance an intervention: correct-antigen predictive performance must improve.

Comparisons: C versus U and S establish the baseline; P versus C tests the
objective; P versus PS tests whether the benefit needs the real pairing. An
antigen-blind model's pairing loss has no informative pairing gradient, so U is
not described as having received equivalent usable antigen supervision.

## Test 4: Does region balancing expose a binding-relevant learning signal?

Add two fresh arms, C-CDR and U-CDR, each starting from its matching initial
tensors. They use ordinary CE with equal contributions from mean CDR-residue
loss and mean non-CDR-residue loss; EOS remains in the non-CDR term. No pairing
loss is added. Six fits in total.

Compare C-CDR versus C and U-CDR versus U. The key conditioning contrast is
C-CDR versus U-CDR, with the change in that contrast relative to C versus U
reported as an interaction. This prevents crediting antigen conditioning for a
generic improvement caused by reweighting the training objective. CDR3 NLL is
primary, while full-chain NLL and wrong-antigen interventions detect trade-offs.
This revisits an inconclusive earlier idea rather than claiming it is new.

## Test 5: Is fixed average pooling discarding antigen information?

Add two fresh ordinary-CE arms using a learned 32-token antigen resampler:
correct-pair L and shuffled-pair LS. Same frozen input features, decoder, output
positions and memory-token count as C and S. Add U-L, an unconditional IgLM
feed-forward-width control matched to the learned model. Nine fits in total.

Compare L versus C and LS versus S. Report the paired-pairing effect and its
change with the resampler. Parameter counts and extra compute are explicit:
an isolated improvement in both correct and shuffled arms may reflect capacity
rather than better biological conditioning. This screen does not claim that
different pooling mechanisms have identical effective capacity.

## Test 6: Does measured-affinity supervision improve the generator's ranking?

This arm is independent of the unsupervised pairing experiment and requires a
new, curator-verified affinity training table. It must contain target identity,
parent/lineage identity, assay provenance, measurement type, censoring information
and experimental uncertainty or replicate information. Keep current external
benchmark assays and their relatives excluded. Do not use structure confidence,
expression, or source labels as substitutes for measured affinity.

Use only confidently ordered pairs from the same target, parental context,
construct format and assay conditions. Uncensored measurements must differ by
at least ten-fold and by more than the documented experimental uncertainty.
Do not equate Kd and IC50 or use pairs whose ordering is unknown because of
censoring. If eligible pairs or independent evaluation groups are absent, mark
this test infeasible and continue the other tests.

Add C-Rank and U-Rank, trained on the same strong/weak pairs with ordinary CE on
both targets plus a softplus preference on length-normalized sequence
log-likelihood; fixed ranking coefficient 0.1. Add fresh C-AffCE and U-AffCE
references that see the exact same assay-pair schedule and CE on both targets
without a preference term. Twelve fits in total. The primary comparison is
C-Rank versus U-Rank; C-Rank versus C-AffCE and U-Rank versus U-AffCE isolate
the effect of preference training. Original C/U fits used a different sampling
population and cannot serve as causal controls for affinity supervision.
A gain over unranked IgLM alone does not establish a conditioning advantage.
Match examples, target tokens and optimizer updates across these four arms,
and report their extra forwards relative to ordinary chain CE. Because this is
an objective ablation, the
candidate scorer is the same autoregressive generator; no separate antibody
embedding becomes conditioning input.

Primary metric: mean within-assay Spearman correlation, with equal assay weight.
Report per-assay Pearson correlations and fixed top-decile enrichment as
secondary endpoints. Retain all eligible measured variants, not just pairs used
in the training loss. Independent parent/target component labels determine
biological resampling. Report unavailable intervals rather than row-wise CIs
when those labels cannot be defended.

## Screening decisions, uncertainty and confirmation

These budgets deliberately provide triage, not a final generalization study.
The 0.1 loss coefficients and following effect thresholds are prospective design
choices, not literature-validated optima. A negative short test is inconclusive
about longer training; no automatic budget extensions follow a negative result.

For Tests 3–5, an intervention is promising if mean correct-antigen CDR3 NLL
improves by at least 0.01 nats/token over its CE reference, the paired effect is
positive in at least two of three seeds, and full-heavy NLL worsens by no more
than 0.02 nats/token. A conditioning claim additionally needs an improved
correct-versus-alternative-antigen gap of at least 0.01 nats/token, without the
gain being explained entirely by its shuffled/unconditional control. Report all
contrasts; these rules are triage flags rather than multiplicity-adjusted claims.
For Test 6, the corresponding practical threshold is a mean Spearman improvement
of at least 0.03 over the equally supervised unconditional model and over its
matched conditional assay-CE reference, again positive in at least two seeds.

Show every seed, a paired-effect plot, and 95% crossed bootstrap intervals with
10,000 draws over seeds and independent biological components. For NLL, sum
paired target losses and token counts within sampled components before division.
For affinity, resample the appropriate independent biological units and
recompute within-assay correlations. Seeds are never additional biological units.
Three-seed bootstrap tails remain exploratory. Test 2 can show only seed SD and
per-target effects, given its missing independent lineage annotations.

At most one change per objective family advances to a separately preregistered
confirmation study using fresh seeds, the untouched confirmation panel, a longer
fixed budget, compute-matched controls, and multiplicity correction across its
declared primary contrasts. Do not compose the apparent pilot winners before
their individual effects are confirmed. NLL, pairing specificity and affinity
ranking are distinct outcomes; a gain in one does not establish the others.
Improved binding of generated candidates ultimately requires experimental
validation, beyond these short computational screens.

## Unattended operation

The proposed controller freezes selection and schedules before any fitting,
serializes GPU work, writes a completion/error marker for every stage, and uses
blocking subprocess completion waits. It makes only the decisions above.
Failures, missing labels, infeasible cohorts and time limits are explicit;
independent diagnostics still finish. The assistant reviews results once the
entire short-test controller exits, using the existing exit-triggered mechanism.
No intermediate metric polling, repeated confirmation requests, or automatic
training restarts are part of this design.
