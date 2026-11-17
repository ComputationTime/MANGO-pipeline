# Recommendations for antigen-conditioned antibody pairing

Written 2026-10-27. Prioritize pairing-sensitive training and antigen balance,
using cross-attention as the first architecture to test. Treat score calibration
as a separate evaluation option.

Updated 2026-11-03: first isolate data variety and focal loss in the controlled
pilot below, then add pairing supervision if correct-antigen dependence remains
weak. The numerical OpenAlphaSeq findings still refer to the original interim
snapshot.

These recommendations draw on the frozen **79,579-query interim snapshot**,
including 46,867 antibodies with reliable positive partners, and the existing
ASD conditioning experiments. They are proposed experiments; the gains from new
training have not been established. See the [pairing protocol](openalphaseq_pairing.md)
and the [diagnostic findings](../artifacts/experiments/openalphaseq_pairing_16k_v1/conditioning_diagnosis/findings.md).

## Evidence motivating the changes

- **Antigen identity has little effect on held-out sequence prediction.** For
  the 16k cross-attention model, heavy-chain NLL is 0.84551 with the correct
  antigen and 0.84787 with unrelated antigen donors. Fixed-position prefix NLL
  changes from 0.83251 to 0.83314. Lower NLL alone does not establish pairing
  specificity.
- **Training is concentrated.** Two antigens provide 66.475% of training records,
  50.236% of scored chain examples/optimizer updates, and 51.669% of target tokens.
  The model sees 1,666 distinct antigen sequences, but their influence is uneven.
- **The objective emphasizes whole-chain reconstruction.** CDR3 supplies
  269,235 of 2,511,806 target tokens, or 10.72%. All current 16k models use
  positive-pair cross-entropy, one epoch, and validation NLL for checkpoint
  selection.
- **Generic antigen preferences inflate some metrics.** Fixed-position MANGO's
  measured-pair AUROC is 0.8871, compared with 0.8883 after antibody reassignment.
  A high AUROC can therefore occur without antibody-specific matching.

## 1. Add a pairing-sensitive training objective

Retain ordinary autoregressive cross-entropy and test an auxiliary loss that
rewards higher antibody likelihood for supported binding pairs than for suitable
comparison pairs. Calculate this from the same decoder likelihoods used for
generation and scoring.

First audit whether independent training sources provide reliable measured
positive/background-negative pairs. Exclude uncertain measurements and mask
known cross-reactive positives. A shuffled or unmeasured pairing is not an
experimentally established nonbinder. If mismatched pairs are used as weak
supervision, disclose that assumption and test its sensitivity separately.

The model task remains **antigen + requested chain type → antibody chain**.
Antibody sequences remain output targets; no antibody-derived embedding or
additional antibody conditioning enters the model. Keep OpenAlphaSeq out of
training for this external comparison.

Binding-supervised fine-tuning has improved specificity prediction in other
antibody architectures. This supports testing the direction, but does not
demonstrate an improvement for MANGO. See [Wang et al. (2025)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012153).

## 2. Reduce antigen and antibody-library imbalance

Compare the current objective with antigen-balanced loss weighting at the same
optimizer-update budget. Weighting offers a first experiment that retains the
existing one-pass target membership. Preserve the individual heavy/light target
masks and report effective weights by antigen and chain.

Within each antigen, examine whether related antibody variants dominate. Test
lineage weighting once sequence-based lineage groups have been defined and
audited. The existing `ab_ag_cluster` split groups must remain intact; they are
not automatically antibody-lineage annotations.

Changing the sampler is a separate option. If sampling repeats or omits targets
within an epoch, update the membership audit explicitly and report the resulting
exposure. Increasing record count without addressing concentration is unlikely
to provide a clean test of additional pairing information.

## 3. Test modest CDR weighting during training

Compare whole-chain cross-entropy with an additional CDR or CDR3 loss, using
IMGT annotations of training targets. Keep framework and EOS prediction in the
objective and retain ordinary full-chain NLL for evaluation.

Choose region weights on a separate development set. CDR3-only inference scoring
was not a consistent remedy in the diagnostic comparison; training with region
weights is a distinct hypothesis. Focal loss also does not explicitly teach
which antigen belongs with an antibody.

## 4. Evaluate calibration as a supplementary scoring method

Removing each antigen's average NLL bias improves some retrieval scores without
retraining. Offsets estimated from the other unlabeled hash partition produced:

| Model | Raw family top-1 | Calibrated family top-1 |
|---|---:|---:|
| Regular prefix | 19.16% | 22.83% |
| Fixed-position prefix | 25.33% | 28.28% |
| Cross-attention | 19.26% | 28.95% |

Family top-1 chooses among four families; uniform random selection gives 25%.
Protein top-1 chooses among 12 proteins. Its uniform random baseline is 24.44%
on this snapshot because multiple measured positive partners can count as
correct; it would be 8.33% with exactly one positive per antibody.

Keep raw scores primary and report calibration separately. The hash partitions
share source libraries and parental lineages, so this is exploratory calibration,
not independent biological validation. The shuffled-training control also
improves, reaching 30.22% calibrated family top-1. Cross-attention's measured-pair
AUROC excess over reassignment changes only from 0.0120 to 0.0133.

Subtracting antigen-free IgLM NLL cannot change the antigen ordering for one
antibody: it subtracts the same value from every antigen score. It can change
within-protein binder classification and should be assessed for that task.

## 5. First test data variety and focal loss

Focal loss is already implemented in `workflow/scripts/training_losses.py`.
The existing setting is `gamma=2`, with uniform token-class weights. It reduces
the influence of easy next-token predictions; it does not rebalance antigen
exposure or directly teach binding specificity. See the [loss comparison
protocol](asd_loss_comparison.md) and [Lin et al.](https://arxiv.org/abs/1708.02002).

The earlier one-seed, small cross-attention pilot was inconclusive for this
purpose. After ASD pretraining and SAbDab2 fine-tuning, primary native test NLL
was **0.99451 with cross-entropy and 1.00099 with focal loss**. These are ordinary
NLL values on the same 362-record test subset, not focal-objective values.
That comparison did not measure OpenAlphaSeq pairing in the current 16k models.
Source: `artifacts/experiments/asd_her2_free_prototype_v1/plots/sabdab2_loss_comparison.csv`.

### Prepare and audit the data before training

1. Define diversity using antigen sequences/families, antibody lineages, source
   balance, and effective exposure. Report largest-antigen shares as well as
   unique counts. Larger row counts alone are insufficient.
2. Use the existing eligible ASD pool as the starting point. The 16k cohort
   already represents all **1,666** antigens eligible under the current length
   and source rules. The ten-record-per-antigen cap permits only **4,007** usable
   training records. Select a pilot size after deduplication; approximately
   4,000 records is a practical first target, subject to matched cohort capacity.
3. Audit additional paired training sources for genuinely new coverage. The
   existing SAbDab2 training split contains 219 exact antigen constructs, 195
   absent from the current 16k cohort. A preliminary length/family/exact-chain
   screen against the frozen ASD and native holdouts leaves 362 candidate rows
   with 168 constructs, 144 absent from the 16k cohort. These are candidates,
   not an approved mixed training cohort or proof of new antigen families.
4. Before mixing sources, complete the existing PDB, antigen homology, antibody
   overlap, source/HER2-policy, and group checks against **every** frozen
   validation/test set. Native training records are not automatically safe
   against ASD internal holdouts. Preserve `ab_ag_cluster` boundaries and
   recompute individual-chain target masks and deduplication across sources.
5. Freeze equal-sized current-mixture and diversified training cohorts using
   training information only. Keep evaluation membership fixed. A changed
   source mixture and antigen balance constitute a combined data-policy
   intervention; report both rather than attributing all effects to diversity.

### Run a two-by-two comparison

Use cross-attention with frozen ESM2 650M and the same antigen-only H/L selector
contract in all four arms:

| Arm | Training data policy | Token loss |
|---|---|---|
| A | Current-mixture pilot | Cross-entropy |
| B | Same current-mixture pilot | Focal, gamma=2 |
| C | Balanced/diversified pilot | Cross-entropy |
| D | Same balanced/diversified pilot | Focal, gamma=2 |

Use three matched initialization seeds: **12 runs**. A one-seed screen can
precede confirmation, but cannot establish a reproducible gain. Match optimizer
updates and initialization/optimizer settings, and report scored chain examples,
target tokens, and runtime. Record counts and nominal epochs alone do not match
compute because chain masks and antigen lengths differ. Reusing cached ESM2
tensors avoids unnecessary encoder work.

Keep CDR weighting and pairing loss out of this first matrix so the comparisons
identify the effects of data policy, focal loss, and their interaction. If needed,
test `gamma=1` as a prespecified development-set sensitivity check.

Select settings on group-aware development data using ordinary full-chain/CDR3
NLL, annotated-partner retrieval, and correct-versus-wrong antigen gaps. Assess
promising arms with matched shuffled-training controls and antibody reassignment.
Keep raw scoring primary and calibration separate. Use OpenAlphaSeq for external
comparison after choices are fixed; these already-inspected results are
exploratory, and stronger claims need independent lineage/source confirmation.

No GPU training was launched for this proposal. New cohorts and configurations
need a new experiment root and a launcher entry before execution; there is no
existing command that runs this exact matrix.

## 6. Follow with a pairing-loss comparison

After the data/focal pilot, add explicit pairing supervision if held-out
correct-antigen dependence remains weak. A focal-loss improvement in sequence
prediction is insufficient evidence of pairing specificity.

Start with cross-attention and retain fixed-position prefix as an architecture
reference. Existing experiments support fixing antibody positions for prefix
models, but position changes alone have not established strong pairing ability.

Use the same protected 16k cohort and compare:

| Arm | Antigen weighting | Training objective | Main comparison |
|---|---|---|---|
| A | Current | Whole-chain cross-entropy | Reproduced baseline |
| B | Balanced | Whole-chain cross-entropy | Effect of antigen balance |
| C | Balanced | Cross-entropy + pairing loss | Effect of pairing supervision |
| D | Balanced | Cross-entropy + pairing loss + CDR weighting | Added effect of region weighting |

Run three seeds per arm. Hold training membership, split protection, initialization
policy, and update budget constant. Record forward-pass cost as well: pairing
loss may require additional computation. Add matched shuffled-training controls
for promising arms, recording how many correct antigen identities remain in
each shuffle. The existing 16k shuffle retains 8.9875% correct identities.

Select weights and checkpoints on a separate group-aware development pairing
evaluation. Report pairing quality alongside full-chain NLL, rather than selecting
only the lowest validation NLL. Prespecify the acceptable NLL tradeoff before
comparing external results.

## Criteria for advancing a model

A candidate should improve held-out partner retrieval and show an advantage
over antibody reassignment and matched shuffled-training controls. Report
per-family results, measured positive/background comparisons, CDR3 NLL, and
correct-versus-wrong antigen effects alongside aggregate scores.

Check consistency across seeds and independent antibody lineages where those
groups are available. Retain all `ab_ag_cluster` split protections. Confidence
intervals should use appropriate biological groups; row counts and reassignment
variation do not establish independent biological replication.

Extra epochs, mean pooling, and adding an antigen branch have already produced
mixed results in the [earlier conditioning experiments](antigen_diagnostics.md).
Prioritize the controlled loss/weighting comparison before expanding those
experiments or increasing model size.

## Supporting artifacts

The CPU diagnosis is under
`artifacts/experiments/openalphaseq_pairing_16k_v1/conditioning_diagnosis/`:

- `analyze.py`: reproducible analysis of the frozen interim matrices.
- `score_diagnostics.csv`: 34 model, region, and scoring comparisons.
- `per_family_diagnostics.csv`: family-specific outcomes.
- `score_decomposition.csv`: generic antigen-preference contributions.
- `summary.json`: source hashes, training concentration, and limitations.

The diagnostic reproduced the existing raw/centered metrics within 1e-12 and
checked that IgLM subtraction preserves within-antibody antigen score differences.
The numerical results above describe that interim snapshot, regardless of later
progress in the full benchmark.
