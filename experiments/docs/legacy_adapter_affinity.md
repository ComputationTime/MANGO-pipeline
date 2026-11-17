# Frozen adapter affinity experiment

Run `MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./experiments/run_experiment.sh legacy-adapter-affinity`.

This inference-only experiment evaluates nine existing checkpoints: correctly
paired frozen antigen adapters, shuffled-antigen frozen adapters and native IgLM
continuation, each for seeds 0, 1 and 2. It loads no public pretrained IgLM weights
and changes no model or legacy training protocol. All checkpoint and input hashes,
exact model selections, training exposures and implementation hashes are frozen
before new affinity scores are computed.

The primary estimand is the correctly-paired minus shuffled-adapter difference
in equal-target mean Spearman correlation of negative heavy-chain NLL against
negative supplied log10 Kd. This isolates real versus shuffled antigen pairing
within the existing frozen-decoder architecture. Per-target differences and
perplexity-versus-log10 Kd Pearson coefficients are reported. Correctly-paired
versus IgLM-continuation differences are secondary composite comparisons because
freezing, architecture and dropout policy differ. They cannot isolate conditioning.

The 1,000 reliable pairs per target are the previously frozen, sequence-hash
selection from v3, not newly chosen examples based on affinity values. HER2,
PD1 and TIGIT are evaluated separately; their correlations have equal weight
in the primary statistic. They are not pooled into one larger apparent sample.

Every checkpoint is restored in evaluation mode and scored in inference mode.
Scalar, cached, repeated and batched likelihoods must agree within 3e-6 nats/token
at batch sizes 1, 8 and 64 for both selectors and EOS. Frozen adapter decoder
tensors are independently checked against their original scratch IgLM source.

Plots display each seed and ±1 sample SD over matched optimization seeds.
These are not biological confidence intervals, standard errors or significance
tests. Related variants lack independent parent-lineage identifiers, and the
benchmark and legacy training source have already been inspected. The historical
checkpoint selection used only two validation groups. The stricter new native
audit does not retrospectively certify the legacy cohort. These limitations are
retained regardless of whether a coefficient improves. No model is advanced,
no affinity calibration is fitted, and no new generalization or binding claim
is made from this screen.

`affinity_correlations.csv`, `paired_differences.csv`, `affinity_correlations.png`,
`paired_effects.png`, `results_report.json` and `findings.md` are written under
`artifacts/experiments/legacy_adapter_affinity_v1/`. Scoring jobs are serialized,
independent failures do not stop later checkpoints, and a final report is
attempted even if some scores are missing.
