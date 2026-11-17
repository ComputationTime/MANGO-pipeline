# Short affinity screen v3a: accelerated protection audit

This is a versioned execution amendment to `experiments/docs/short_affinity_tests_v3.md`.
The original v3 run and all its sources remain frozen. Its 900-second audit
timed out in the CPU antibody-search prefilter. Cohort feasibility was unknown;
no scientific model was trained. Completed v3 diagnostics showed no pilot flags.

The only scientific pipeline change here is the validated search backend.
Antibody searches use the same MMseqs2 binary, GPU gapless prefilter, 90%
identity, 80% bidirectional coverage, E <= 0.001, and up to 100,000 candidates.
The antigen search remains the original CPU implementation. Existence-only
exclusion queries stop after one accepted hit; the full lineage graph retains
all accepted hits. No biological protection, cohort capacity, independence,
model arm, training schedule, seed, effect threshold, or inference rule changes.

Before this amendment, the backend probe recovered every CPU-positive query
on 176 validation queries (including exact and near-positive controls), and
all 6,039 CPU edges from the prior completed lineage graph. This finite probe
does not prove universal equality of heuristic searches. Its frozen hashes are
required by this run. New additional GPU hits would conservatively increase
exclusions; they would not be used to relax any criterion.

The audit receives 1,800 seconds and an exclusive GPU resource. Its elapsed
wall time counts toward a 7,200-second total GPU-stage budget. This adds the
audit allowance to the original 5,400-second budget; individual fit, embedding,
and diagnostic limits remain unchanged. Timeout is incomplete, not evidence of
cohort infeasibility. GPU backend failure does not silently fall back to CPU.

The six completed frozen-checkpoint diagnostics from v3 are reused byte for
byte after checking selection, input and checkpoint hashes, rather than scored
again. This is explicitly recorded as reuse, not six new fits. All actual model
fits still start from scratch and run only if the unchanged audit passes.
Ranking remains ineligible without independent curator-verified affinity data.
Training comparisons remain exploratory; formal confirmation requires a separate
frozen study and the rules in `experiments/docs/mango_advantage_confirmation.md`.

The controller blocks on each launched stage and writes a final report. The
completion watcher resumes this conversation only after the controller exits.
