# Evidence required for the ongoing MANGO advantage goal

The goal remains to obtain several MANGO-style models that outperform a fairly
trained, similarly sized IgLM comparator. A short pilot, a successful software
test, or a favorable single seed does not establish that outcome. Binding
affinity prediction and antibody-chain likelihood are separate claims.

This document fixes the evidence requirements before reviewing the v3 short
screen. It does not launch another experiment or modify any existing protocol.

## Fair comparisons

- Use at least two distinct, individually specified conditioned variants, with
  architecture and objective changes isolated before combining them. Report
  all attempted variants, failed fits, data exclusions and negative findings.
- Pair each candidate with an unconditional IgLM architecture variant matched
  in total trainable parameters within 0.1%. Also retain native scratch IgLM
  as a reference. A wider feed-forward IgLM must be labeled as such.
- Account separately for frozen feature extractors, their pretraining, and
  whole-system inference parameters and compute. The v3 match concerns trained
  decoder/conditioning parameters; frozen ESM2 features bring additional
  pretrained information and extraction cost. Do not describe that comparison
  as matching all inference parameters. A strong follow-up includes an antigen
  encoder learned from scratch with whole-system parameter matching, or another
  explicitly justified control for this information/computation difference.
- Start both sides from scratch on the audited training data. Match datasets,
  supervised labels, example order, target tokens, optimizer updates, optimizer,
  learning-rate schedule, dropout policy and checkpoint-selection rule.
  Preserve shared initialization wherever tensor shapes permit it.
- Affinity-ranked MANGO is compared with affinity-ranked IgLM, with matched
  CE-only controls on exactly the same assay-pair schedule. Antigen-specific
  losses and their additional computation must be explicit. Include an
  unconditional compute-matched sensitivity check before making an efficiency
  claim. A gain over an undertrained or differently supervised baseline does
  not qualify.
- Keep the generation task antigen plus requested chain type to full antibody
  chain. No supplied antibody sequence, antibody embedding, native framework,
  or oracle antibody-derived epitope crop may enter conditioning.

## Independent evidence

Maintain original `ab_ag_cluster` and biological protection requirements.
Never train on previously held-out rows. Every development/confirmation panel
needs a documented lineage/antigen grouping and overlap audit; no row-random
substitution is permitted when a grouped split is infeasible.

Development results can select a small, declared candidate family and a common
fixed training budget. Confirmation uses at least five fresh matched seeds and
an untouched biological panel. Choose that budget before confirmation, and
never add updates or seeds in response to a near-significant result. A timeout
or incomplete arm cannot be compared as if it completed the matched regimen.

For a binding claim, use experimental affinity labels with assay provenance,
construct format, censoring status, parent/lineage labels and quality criteria.
Report results within assays and targets. Current HER2/PD1/TIGIT library scores
have already been inspected and lack defensible independent parent labels;
they remain exploratory and cannot provide the final biological confirmation.
Enough independent assay/lineage units must exist to support the declared
interval procedure; a large variant count alone cannot establish that.

Where structural or sequence source data are enlarged, audit the complete
candidate cohort before choosing any bounded subset. More records need not
mean more independent biological groups. Available native SAbDab2 rows are a
candidate source for a future, separately specified audit: the current native
cohort has 991 rows (509 training, 56 validation, 426 test) and 76/4/61 original
source groups before additional lineage merging. None may be assumed independent
or moved into training on the strength of these counts.

## Outcomes and statistical decisions

For binding-affinity ranking, the primary statistic is the candidate-minus-
matched-IgLM difference in mean within-assay Spearman correlation with a
larger-is-stronger affinity variable. Weight assays equally and disclose each
assay's effect. Pearson correlation and top-decile enrichment are secondary.
Predeclare a practically meaningful mean improvement of 0.03.

For chain prediction, use held-out heavy-chain token-weighted NLL, including
EOS, with positive improvement defined as IgLM minus MANGO. Report CDR3 and
whole-chain results separately. A practically meaningful point improvement
is 0.01 nats/token. Correct-versus-alternative-antigen effects diagnose use of
conditioning but are not themselves measured binding affinity.

Use paired seed and biological-unit resampling, preserving the same draws for
both models. Aggregate target losses and token counts before NLL division;
recompute within-assay correlations for affinity resamples. Report individual
seeds, paired effects, and confidence intervals. Seed variation does not replace
biological uncertainty. Correlations that are undefined in a resample and
unreliable small-unit intervals must be reported explicitly.

Confirmation tests use a predeclared paired inferential method, a positive
interval lower bound and a multiplicity-corrected result, as well as the
practical point threshold. The inferential implementation must be checked for
null calibration before confirmation; exploratory percentile-bootstrap tails
from three seeds are insufficient evidence on their own.

The completed `artifacts/experiments/interval_calibration_v1/` stress test
confirms this restriction. In a zero-effect, seed-only normal scenario, nominal
95% percentile intervals excluded zero in 23.65% of simulations with three
seeds and 15.475% with five. Even this simple limiting case failed calibration.
The separate Student t reference also failed under skewed seed effects, and
does not address biological uncertainty. Neither method is approved for
confirmation. Five fresh seeds is a minimum independence requirement, not
evidence that an interval is calibrated; simulation counts and Monte Carlo
intervals are recorded with the frozen calibration protocol. Future inference
must also validate biological-axis uncertainty, nonlinear assay correlations
where relevant, and the declared corrected alpha levels before model outcomes
are reviewed.

Across successive confirmation rounds, allocate family-wise alpha as
`0.05 / 2**round`, beginning with round 1. Use Holm correction across every
declared primary candidate contrast within each round. This spending rule
does not authorize adaptive reuse of a test panel: a failed confirmation cannot
become the tuning set for another claim on the same biological evidence.
New hypotheses require fresh confirmation evidence or an independently valid
reusable-holdout method specified before testing.

Call a variant confirmed only for the endpoint actually tested. At least two
conditioned variants must independently satisfy the fair-comparison and
confirmation requirements before claiming the requested variety of MANGO
advantages. Given this conversation's binding objective, sequence-only success
does not close the affinity question. Improved computational ranking also does
not establish that generated candidates bind better without experimental testing.

## Execution discipline

Continue with the next scientifically motivated ablation or data audit after
each completed batch. Preserve old protocols and code, or retain immutable
source snapshots when a new version is necessary. Do not select the best seed,
discard inconvenient biological groups, replace failed arms with silent
fallbacks, or keep extending a fit until it crosses a significance threshold.
If independent data are unavailable, report that limitation and pursue feasible
data acquisition or other useful diagnostics; do not manufacture a positive
result by changing the evaluation definition.

All GPU work remains serialized. The assistant reads experiment logs after
the relevant launcher exits. A live process is a wait, not evidence of success
or a reason to restart it. Completion-triggered reviews and the active goal
coordinate subsequent work.
