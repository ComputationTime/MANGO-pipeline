# Controlled antigen study v2

This follow-up separates antigen information from added decoder capacity and
decoder freezing. IgLM here is trained from scratch; no public IgLM checkpoint
is loaded. The secondary adapter experiment starts from that scratch-trained
checkpoint and is reported separately from the primary scratch comparison.

Run from the repository root on the local 12 GB card:

```bash
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./experiments/run_experiment.sh controlled-antigen-smoke
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./experiments/run_experiment.sh controlled-antigen-study
```

The smoke run uses twelve real cached records, one epoch and one seed. It tests
the workflow and likelihood implementation; its metrics have no scientific
interpretation. The full configuration is
`experiments/config/controlled_antigen_study.yaml`. Output roots are separate from the
unchanged `affinity_improvement_v1` experiment. Repeating the identical command
reuses verified completed jobs. Frozen settings, code hashes and data manifests
reject incompatible resumes. A changed protocol requires a new output root.

## Causal comparisons

All models use native IgLM vocabulary, four layers, eight heads, width 512,
inner width 2048, positional capacity 512, native H/L selector and the same
fixed HUMAN token. Targets are the requested complete antibody chain and EOS.
Only antigen features may enter cross attention. No antibody embedding enters
conditioning; teacher-forced earlier antibody tokens are autoregressive outputs.
The unconditional comparator receives only the two control tokens.

Frozen production ESM2 650M residue features are deterministically pooled into
32 ordered tokens. Correct, shuffled and constant-memory models have identical
added architecture, parameter counts and starting tensors within each seed.
Cross-attention output projections initially equal zero, preserving the exact
unconditional likelihood. The constant control supplies zero features before
the trainable projection, retaining its bias and every other added parameter.
Training shuffles are complete antigen derangements separately within each
requested chain, preserving antigen frequencies across its training examples.
Inference uses the correct antigen for all arms except the constant control.

| Primary arm | Starting decoder | Memory | Updates |
| --- | --- | --- | --- |
| IgLM | Random scratch | None; no added branch | Decoder |
| Correct | Identical scratch decoder | Correct antigen | Decoder and branch |
| Constant | Identical scratch decoder | 32 zero feature tokens | Decoder and branch |
| Shuffled | Identical scratch decoder | Deranged training antigen | Decoder and branch |

The secondary experiment is a 2×3 factorial: frozen versus jointly updated
decoder crossed with correct, shuffled or constant memory. All six cells start
from the same seed's best primary scratch IgLM checkpoint. A seventh control
continues unconditional IgLM for the same additional training budget. Adam,
learning rate 1e-4, examples, order, objective and optimizer-step/token budgets
are matched within each comparison. Frozen and jointly updated models both
retain training dropout 0.1. Wall time, parameters, trainable parameters, forward
counts, exact initial decoder hashes and target-token budgets are recorded.

## Data gate and exclusions

The v1 screened pool is reused only after verifying original source hashes and
that the earlier candidate caps were nonbinding. The inherited audit excludes
the external affinity benchmark, related antigen sequences and antibody
lineages, native SAbDab2 protection overlaps, unsupported targets and invalid
chain examples; its exact exclusion counts remain linked in the new audit.
This study additionally searches all eligible antibody chains against one
another using MMseqs (90% identity, 80% bidirectional coverage).

Connected components join original `ab_ag_cluster`, antigen CATH family, exact
antibody chains and near antibody lineages. Components cannot cross splits.
Any component containing an old validation or test row is held out in full;
no old holdout row may become training data. Extra whole old-training components
can be reserved to form new development and confirmation panels. Each panel
requires at least 20 independent components and 100 heavy-chain examples after
deduplication; training requires at least 1,000 records and 100 antigens and is
capped at 2,000 records. These minima are feasibility guards, not a guarantee
of statistical power. An infeasible audit produces a report and stops training
without weakening thresholds or using row-wise random splitting.

The panels have distinct roles within this follow-up. Related source data have
already been examined in earlier studies, so confirmation is an internal
replication, not a pristine external validation. Fully independent evidence
will require additional protected biological groups.

## Fixed sequence and uncertainty

1. Run the separately versioned inference amendment on all six frozen v1
   baseline/IgLM selections. Restore in evaluation mode **and** inference mode;
   check scalar versus cached likelihood at batch sizes 1, 8 and 64, repeated
   calls, both selectors and EOS. Maximum NLL discrepancy is 3e-6.
2. Audit the data before any new training. Only a feasible audit enables ESM2,
   ANARCI and training jobs. Each stage runs in its own Conda environment and GPU
   stages share the single `gpu=1` resource.
3. Train all primary and secondary pilots with seeds 0, 1 and 2, two epochs per
   stage. Select each checkpoint using ordinary validation H/L token-weighted
   NLL. The selection panel is also used for pilot screening, so pilot effects
   are explicitly exploratory and potentially optimistic.
4. Advance a family only if its pairing contrast improves heavy NLL by at
   least 0.01 nats/token on average and at least two of three seed effects are
   positive. Secondary screening considers either frozen or joint pairing.
   Confirmation repeats the whole selected family with fresh seeds 10–14 and
   the same cohort size and epochs, scoring only the separate confirmation
   panel for final contrasts. Secondary sources use matching primary IgLM
   checkpoints even if the primary family does not advance.
5. Freeze model selection before reading new external affinity scores. Failures
   are recorded and independently runnable arms continue. Missing contrasts
   cannot advance or produce positive claims.

The primary endpoint is heavy-chain token-weighted NLL. Full-chain CDR and CDR3
summaries, light-chain checkpoint validation and wrong-antigen interventions
are secondary diagnostics. Paired contrasts are defined in
`controlled_antigen_common.py`; positive values favor the named effect:

- Primary: correct versus shuffled (pairing), correct versus constant (content),
  constant versus IgLM (branch capacity), correct versus IgLM (overall).
- Secondary: correct versus shuffled within frozen and joint cells, frozen
  versus joint under correct memory, and the difference between their pairing
  benefits (interaction).

Error bars are 95% crossed bootstrap intervals with 10,000 draws. Seed and
biological-component axes are independently resampled; all arms share the same
draws. Loss sums and token counts are aggregated before division, and each
seed receives equal weight. Seeds are optimization replicates, not additional
biological examples. Centered bootstrap two-sided p-values use a +1 correction.
Confirmation applies Holm correction separately within the four primary and
four secondary hypotheses at alpha 0.025 each. Support requires the corrected
test, a positive interval lower bound and improvement at least 0.01 nats/token.
Small seed counts limit tail calibration; pilot variance is not a license to
change thresholds or add more seeds after inspecting a result.

## External affinity analysis and outputs

The existing HER2/PD1/TIGIT AlphaSeq measurements are exploratory because they
have already been inspected. Report Pearson and Spearman separately per
target using heavy-domain perplexity and the existing reliable measurement
aggregation. SciPy independently checks coefficients. Seed plots show ±1
training-seed SD, explicitly distinct from biological uncertainty. Independent
parent-lineage identifiers are unavailable, so no row-wise biological CI is
invented. Better held-out NLL alone does not establish better binding affinity.

The output root contains `protocol.json`, `data_manifest.json`,
`cohort_audit.json`, `records.csv` when feasible, per-run checkpoint/score hashes,
`state.json`, `selection.json` when selection is reached, `run_metrics.csv`,
`uncertainty.json`, `affinity_correlations.csv`, `results_report.json` and
`findings.md`. Plots include seed dots and error bars, paired-difference forest,
factorial interaction, wrong-antigen penalties and per-target affinity plots.
The v1 repair writes only to the new root's `amendment_v1/` directory.

The controller blocks on subprocess completion rather than polling metrics.
`review_on_study_exit.py` can attach to the launch PID and queue one review in
the current Codex conversation after process exit. Its Linux PID descriptor
wait makes no model calls while experiments run. `completion_review/watcher.json`
records attachment and submission. The automatic review may diagnose results
and write a review, but does not change this frozen protocol or restart jobs.
