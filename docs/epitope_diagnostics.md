# Epitope and paratope diagnostics

```bash
./run_gpu.sh epitope-diagnostics-smoke
./run_gpu.sh epitope-diagnostics
```

This inference-only study analyzes the 20 frozen checkpoints from the ranking
pilot. Results are in `artifacts/experiments/epitope_diagnostics_v1/`. No new
models are fitted and no original checkpoint, cohort, or protocol is changed.
Configuration: `config/epitope_diagnostics.yaml`.

## Questions and interventions

The study distinguishes using coarse antigen identity from using local
antigen features in a way that transfers to affinity ranking.

For each of 1,000 frozen examples per target (HER2, PD1, TIGIT), compare correct
antigen memory, two alternative antigens, one averaged token repeated 32 times,
32 zero tokens, and complete absence of the cross-attention input. IgLM remains
an unconditional control. Antigen swaps are counterfactual interventions, not
assertions that the alternative proteins are experimentally verified nonbinders.

ANARCI IMGT labels decompose per-token losses into framework, CDR1, CDR2, CDR3,
and all-CDR scores. Whole-chain scores include EOS; regional scores exclude it.
Compare affinity Spearman correlations and correct/absent score rank agreement.
Also test specificity residuals: mean alternative-antigen NLL minus correct-
antigen NLL. These residuals are not assumed to improve affinity prediction.

For 35 held-out heavy-chain examples and the two RNase-A training-pair examples,
map observed structure residues exactly onto the actual model sequences. Define
heavy-chain paratope and antigen epitope contacts using minimum non-hydrogen
atom distance: 5 A primary, with 4/8 A epitope sensitivity checks. The native
examples come from 16 held-out protection groups. Coordinates map successfully
for all 37 examples, and all 35 held-out examples have at least three contact
residues on both sides at 5 A. Median epitope/paratope sizes are 20/17 residues.

The complete set of 32 antigen memory positions is occluded in a fixed order,
replacing each token with the mean memory token. This choice uses no interface
labels. **Structural contacts are joined only after the model predictions.**
No epitope crop, antibody-derived embedding, or oracle structural mask is used
as model conditioning. Output-region masks select predictions for analysis and
do not change model inputs. Structural labels are never used to choose a model
or intervention position.

Project epitope fractions through the exact adaptive average-pooling operator,
including multichain breaks and overlapping pooling bins. Compare token
occlusion effects with these fractions. Contact labels at 4/8 A change the
antigen-side annotation; the paratope-query mask remains defined at the primary
5 A cutoff. Average native records within their biological group, then groups
within seed, and report optimization-seed SD separately from biological variation.

Record queries and keys passively through hooks on the existing cross-attention
linear layers, reconstruct their descriptive softmax probabilities, and verify
that recording leaves likelihoods unchanged. The initial smoke version failed
because switching the trained model to an eager attention backend exceeded the
per-token parity guard. That path was rejected. The successful `smoke_v2`
and full study keep the production backend; passive recording changes token
likelihoods by exactly zero in the full study. A separate unit test checks the
reconstruction against directly returned eager attention weights.

The broader interpretability issue is discussed in
[Jain and Wallace](https://aclanthology.org/N19-1357/) and
[Wiegreffe and Pinter](https://aclanthology.org/D19-1002/). Attention weights
alone do not establish causal feature importance; the independent input
interventions are therefore central to this analysis.

## Validation and exports

Five targeted tests cover exact pooling for short/uneven sequences, EOS/region
alignment, causal token-effect controls, per-token likelihood and attention
parity, memory-order invariance, author residue numbering, hydrogen exclusion,
variable-domain cropping, and multichain breaks. The smoke run and all 23 full
workflow jobs completed. Isolated ANARCI, model and analysis environments are
used; GPU probes share `gpu=1`.

Post-run tools in the isolated analysis environment:

```bash
python workflow/scripts/audit_epitope_benchmark.py
python workflow/scripts/validate_epitope_diagnostics.py
python workflow/scripts/plot_epitope_diagnostics.py
```

The benchmark audit verifies that each 1,000-example panel has a fixed light
chain. Independent validation recomputes 1,260 correlations with pandas ranks
instead of SciPy, reconstructs native region scores from token losses, checks
source hashes, and decomposes the ranking gain into positions/regions.
`mechanism_summary.{png,svg}` is the compact export; individual analyses have
their own CSVs, JSON summaries, and plots.

## Interpretation limits

These are exploratory analyses of previously inspected affinity libraries and
a small native holdout. Error bars are sample SD across five optimization
seeds, not biological confidence intervals. No significance tests are used.
Structural contacts are geometric proximity labels, not energetic hot spots.
ESM2 features are contextualized: occluding one pooled token does not remove
all information about its residues from other tokens.

A framework/CDR likelihood is an autoregressive sequence score, not a local
binding-energy measurement. A framework prediction can change after an upstream
CDR mutation. Likewise, the exact positional decomposition of the RNase-A
preference is not a causal energetic attribution. Its two CDR1 and seven CDR3
changes are targets from one parent, not nine independent binding experiments.

These checkpoints contain no explicit antigen coordinates. They receive ESM2
sequence features averaged into 32 memory tokens. Cross-attention is invariant
to reordering those memory tokens (verified within float32 rounding); the ESM2
features still contain contextual information, so this is not evidence that
all sequence or structural information has vanished.

The diagnostics can show weak use of local interface information. They cannot
alone separate lack of useful feature information, insufficient training data,
an objective that rewards coarse sequence priors, or loss of local information
through pooling. A controlled representation-by-contact-supervision experiment
is needed to separate these explanations.
