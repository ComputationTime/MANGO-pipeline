# Full MANGO/IgLM affinity comparison

After the smoke and smoke-all gates, train the eight supported antigen
representations and a matched IgLM from scratch:

```bash
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./run_gpu.sh train-all
./experiments/run_experiment.sh openalphaseq-affinity
```

`train-all` uses the same scientific configurations, cluster-separated cohort,
run identities, epochs and validation stopping rule as `study`. It trains and
evaluates likelihood models without the downstream design-generation analyses.
Each representation runs in a separate invocation on one GPU; a failure does
not stop the remaining methods. IgLM uses the full published architecture,
native vocabulary, heavy/light tokens, constant species token and the same
full-chain training targets and optimization settings. No pretrained IgLM
weights are used.

The affinity workflow uses `experiments/config/openalphaseq_affinity.yaml` and writes to
`artifacts/experiments/openalphaseq_affinity_full_v1/`. It keeps every experiment
from the pinned official OpenAlphaSeq release, including the RBD mutagenesis
experiment: 2,441 distinct target sequences. Targets without finite measurements
from eligible antibodies have no correlation; their absence is retained in the
source and cohort audits.

The existing expression-domain ANARCI assignments are reused only after their
input hash matches the full source snapshot. Training overlap is recalculated
against the current SAbDab **train split**, with canonical IMGT domain matches
and complete-training-chain substring checks for expression overhangs. The
primary cohort requires every expressed domain to be unseen. Heavy-chain
perplexity is reported for that cohort, with token-weighted paired-domain and
the broader novel-heavy cohort as separate sensitivity analyses. This is exact
sequence exclusion, not lineage or sequence-identity-threshold exclusion.

Only antigen sequences and separately predicted antigen-only Chai-1 coordinates
enter the corresponding conditioning inputs. Every structural target is folded
as its exact sequence, without an antibody or affinity value. No mutant borrows
its parent's coordinates. Structural inference is therefore a substantial
additional stage. Its parameters and per-target structures are cached and
audited. The same production antigen embedders and likelihood objective used
in training apply at inference.

Scoring evaluates only observed chain/antigen combinations. All residues and
EOS contribute to NLL. Perplexity is `exp(mean target-token NLL)`; combined
chain scores sum losses and token counts **before** exponentiating. Score
databases commit each batch and resume completed comparisons. Checkpoint,
cohort, sequence, embedding and implementation provenance are frozen, and
batched scores are checked against scalar production scores for each antigen
and chain selector. IgLM is antigen-free and scores each unique chain once.

The affinity axis is the publisher's `alphaseq_affinity`, already expressed as
log10(estimated Kd in nM). Lower means stronger binding. Repeated qualifying
measurements are averaged per normalized antibody/antigen pair, giving each
pair one vote. Primary quality requires a finite measurement and sufficient
replicate observations; both above-background and background-level observations
contribute. All-finite and above-background-only results are separate sensitivity
analyses. Pearson r uses **perplexity**, alongside separately labeled NLL results.
Pooled, per-antigen, per-family and antigen-centered correlations distinguish
overall association from antigen/library offsets.

Final results are `correlations/primary_correlations.csv`,
`correlations/pearson_correlations.csv`, `correlations/findings.md` and
`correlations/perplexity_affinity.png`. A strict summary requires all nine
complete score databases and verifies their hashes. Related library variants
are not independent biological replicates, so the report does not attach
row-independence p-values or confidence intervals. The affinity belongs to the
expressed VHH/scFv construct; the primary model score belongs to its heavy
variable domain.
