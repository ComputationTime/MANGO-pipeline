# Sequential affinity-improvement study

This is a separate, predeclared experiment. It preserves existing production
models, figures and binding analyses. Run the plumbing check first:

```bash
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./experiments/run_experiment.sh affinity-improvement-smoke
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./experiments/run_experiment.sh affinity-improvement-study
```

The controller blocks until each Snakemake invocation exits. It then validates
outputs, records the outcome, and starts the next independent stage. It does not
poll losses, tail metrics, query an assistant, or require routine approvals.
All GPU rules use `gpu=1` and existing isolated rule environments. Keep the
launcher process running, or start it under a persistent shell/service with
stdout redirected to an artifact log. `state.json` changes only at stage exits.
The controller itself does not wake the assistant. Attach the separate watcher
below to request an assistant review after launcher completion or failure.

For a completion-triggered assistant review in an existing Codex conversation,
attach the separate watcher (it does not alter the frozen experiment):

```bash
python experiments/scripts/review_on_study_exit.py \
  --root artifacts/experiments/affinity_improvement_v1 \
  --thread-id "$CODEX_THREAD_ID"
```

The watcher waits on the launcher's Linux process-exit event with no timer or
model calls, then uses `codex queue --resume` to request one review in that
conversation. It also handles an already-exited launcher. It checks kernel
process start time to avoid PID reuse, takes an exclusive watcher lock, and
records delivery under `completion_review/watcher.json`. Resume warnings are
recorded as warnings even when Codex returns exit code zero. The queued request
asks for review and interpretation, without modifying the protocol or restarting
training. The assistant's review is saved under
`completion_review/assistant_review.md` when workspace permissions allow.
The machine, watcher process and local Codex daemon must remain available.
This uses Codex's documented [queue command](https://learn.chatgpt.com/docs/developer-commands#codex-queue).

## Fixed scientific protocol

`experiments/config/affinity_improvement_study.yaml` declares sizes, seeds, exclusions,
objectives and advancement thresholds. The controller freezes this configuration
and implementation hashes before running anything. Changed inputs require a new
experiment directory. Source hashes, sequence-search commands and results,
exclusions, exact cohort membership, target masks, checkpoint hashes and training
exposure are retained. Scientific outputs live under
`artifacts/experiments/affinity_improvement_v1/`; smoke has a separate root.

1. Probe all latest eight MANGO checkpoints and scratch IgLM on the original
   native test set, including two deterministic wrong-antigen draws. These
   results diagnose existing models; they do not select new training settings.
2. Audit a bounded 32,000-record candidate sample from the existing protected
   ASD cohort. Exclude AlphaSeq-derived sources, BUZZ, external antigen matches,
   and external or reserved antibody-lineage matches. MMseqs uses the pinned
   thresholds in the configuration; existing CATH, native-PDB and group
   protections remain inherited from the verified source audit. Development
   groups retain upstream splits. Whole validation groups overlapping the
   confirmation panel's antibody lineages are removed.
3. Fit 2,000-record (or smaller eligible) pilots for seeds 0, 1 and 2, two epochs
   each. Establish ESM2 cross-attention MANGO and antigen-free IgLM baselines.
   Independently test antigen balancing, length/mask-matched data diversity,
   frozen-decoder antigen adapters, affinity ranking, and CDR weighting. Relevant
   IgLM controls receive the same data/loss intervention. Decoder comparisons
   use matched IgLM continuation and correctly paired/shuffled antigen branches.
4. Advance only complete, consistent candidates meeting the declared development
   threshold and minimum independent-group count. Confirm at up to 4,000 records
   with fresh seeds 10–14, at most ten epochs, and patience two. Checkpoints use
   validation data; confirmation outcomes use the previously unused source test
   partition as a confirmation development panel. Apply Holm correction across
   all five intervention hypotheses, retaining infeasible hypotheses in the
   family. Reserve alpha 0.025 for this family and 0.025 for combination testing;
   combination tests use Bonferroni over the 31 possible nonempty subsets,
   including paths not executed. Combine supported changes incrementally. Larger training is bounded
   at 16,000 eligible records and requires a supported combination.
5. Freeze selection, then evaluate the external HER2/PD1/TIGIT panel once. The
   already-inspected panel is exploratory. Related variants are not independent
   biological replicates, and three antigens do not establish broad transfer.

The affinity endpoint is equal-target mean within-assay Pearson correlation
between heavy-domain perplexity and log10 Kd. Without an independent verified
development affinity panel, advancement uses heavy-domain NLL and is explicitly
labeled **sequence-modeling only**. CDR/CDR3 NLL, wrong-antigen penalties and
matched IgLM/shuffled controls remain visible. Better sequence NLL alone does
not establish better binding or antigen-specific learning.

Bootstrap uncertainty resamples entire held-out source groups using paired
model scores. Seeds measure optimization variability separately. Low group
counts, degenerate bootstrap draws, missing seeds or failed arms cannot advance.
Passing these thresholds is not a prospective power guarantee.

## Independent affinity labels

The downloaded aggregate AbRank table lacks sufficient assay identifiers for
strict ranking. The default ranking experiment is therefore reported as
data-infeasible, without silently using the external benchmark.

An independently curated optional CSV can be configured through
`verified_affinity_records`. Required columns are `id`, `assay_id`,
`assay_provenance_url`, `log10_kd_nm`, and `affinity_operator`. IDs must refer to
audited canonical pool rows; splits and sequences cannot be replaced by this
file. Only uncensored `=` Kd measurements qualify. A curator must establish
assay comparability; a nonempty identifier is not scientific verification.
Comparisons require the same antigen sequence, assay and provenance, with at
least a threefold affinity separation. IC50 values are never treated as Kd.
Repeated measurements should be curated into one qualifying estimate per pair
before supplying this table.

## Failure, resume and outputs

The controller retries a transient infrastructure failure once. Training uses
one target example per microbatch, so OOM at this minimum is reported instead
of silently changing the objective or falling back to CPU. Failed dependencies
skip their descendants; independent interventions continue. SIGTERM terminates
the active child process group before releasing the controller lock. Restarted
unfinished training starts deterministically from its declared initialization;
completed runs and embeddings are reused. There is no partial-optimizer resume.

Read `findings.md`, `results_report.json`, `run_comparisons.csv`,
`development_nll.png`, and, when available, `affinity_comparison.png` and
`external/correlations.csv`. Per-run directories contain completion markers,
checkpoints, epoch histories, live CSV curves, atomically refreshed PNG curves,
per-antigen/chain exposure, and correct/wrong/no-context scores. Each stage has
an isolated log under `logs/` and an invocation log under `controller_logs/`.
No large expansion occurs when screening or confirmation is negative.
