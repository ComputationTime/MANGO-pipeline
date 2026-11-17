# Candidate witnesses for bounded homology audits

This prospective backend does not change a frozen model study. It is prepared
while the v3a launcher runs and must be validated before a new audit uses it.

The expensive external-antibody step asks whether each query has *any* accepted
target match. A small preliminary target subset can provide positive witnesses
without declaring unmatched sequences safe. Select at most 512 real target
sequences using sequence-only prefix/suffix/length buckets: allocate up to three
quarters to query-present buckets, then fill by deterministic sequence hashes.
No binding measurements or model outcomes enter this selection.

Small-database E-values are not sufficient evidence. MMseqs2 computes E-values
using the target database residue count. Every proposed witness is therefore
realigned by the same binary against the complete target sequence database,
using a sparse candidate-pair database. Identity >= 0.9, both coverages >= 0.8,
and E <= 0.001 are rechecked. Only a witness accepted in this full-database
alignment can remove a query from subsequent work.

Every unresolved query is searched against all original targets using the
previously validated full GPU backend and unchanged cutoffs/candidate limits.
This includes queries whose small-subset candidate fails full revalidation.
No-hit evidence from the subset is never treated as evidence of safety. If
candidate search or revalidation fails, the stage fails visibly. Final hit
tables preserve the verified witness or full-search evidence for every
excluded query. The full lineage-graph step bypasses this shortcut entirely,
retaining all accepted edges as before.

The separate CPU-only validation tests sparse realignment and full-database
E-values on the already completed backend probe, including shuffled-sequence
negative pairs. It does not establish full-scale speed or end-to-end GPU
performance. Unit tests cover unknown IDs, full fallback for rejected witness
candidates, bounded deterministic target choice, and bypass for full graphs.
A later versioned study must freeze this code, validation hashes and budgets
before running; current study outputs are not altered.

Primary implementation references:

- https://github.com/soedinglab/MMseqs2/wiki#prefiltering-format
- https://raw.githubusercontent.com/soedinglab/MMseqs2/18-8cc5c/src/alignment/Alignment.cpp
