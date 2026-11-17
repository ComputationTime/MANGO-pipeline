# Predicted interface routing

Run the isolated experiment without modifying previous MANGO code:

```bash
./run_interface_attention.sh smoke
./run_interface_attention.sh study
```

Configuration: `config/interface_attention.yaml`. Results:
`artifacts/experiments/interface_attention_v1/`. Every implementation file is
new; existing modules, antigen inputs, contact labels and checkpoints are read
only. A preservation manifest verifies old source files and trained weights.
The launcher reuses isolated workflow environments and serializes GPU jobs.

## Question and controls

Does feeding predicted interface information directly into cross-attention
improve affinity correlation beyond giving the same model contact supervision?

* `control`: train contact predictors at every layer; do not route predictions.
* `shared_tag`: add the same learned vector to query and antigen features,
  scaled by predicted soft paratope and epitope memberships, before Q/K
  projection. Leave V unchanged to isolate the attention mechanism.
* `contact_bias`: add `2 * sigmoid(predicted_pair_logit)` to attention logits.
* `iglm`: scratch antigen-free decoder, parameter matched within 0.1% to the
  total conditional architecture including the routing predictors.

All conditional arms have identical initial parameters, full-residue ESM2
memory, antigen-only geometry encoder, contact labels, auxiliary gradients,
training exposure and update budget. All use the same attention implementation.
Each layer predicts the NEXT residue's contacts from its available decoder
prefix before that layer's cross-attention. No native antibody structure,
epitope tag, full antibody embedding or future residue enters forward().
Loss uses 0.5 balanced pair BCE, 0.25 balanced paratope BCE and 0.25 balanced
epitope BCE, averaged across layers, with overall coefficient 0.1. Marginal
labels come from the same native contact map, only in the loss. Predicted
memberships are soft and are not calibrated contact probabilities.

The tag starts with per-coordinate standard deviation 0.1; direct bias is
bounded in [0,2]. These fixed choices are recorded before training and are not
tuned on evaluation outcomes. The two interventions differ in parameterization,
strength, and information: tags route region marginals, whereas pair bias routes
residue-pair scores. Their direct comparison cannot isolate injection location
alone; the clean tests are each routing mechanism against the matched control.
This pilot does not exhaust either method's hyperparameters.

Five new seeds (400–404), 1,000 native updates plus 300 adaptation updates,
native batch 4 and identical RNase-pair CE exposure are fixed. There is no
affinity ranking loss or held-out checkpoint selection. Existing protected
splits are reused: 99 training records/33 groups, 35 evaluated heavy chains/16
held-out groups. Native geometry is canonical experimental antigen geometry;
external geometry is the previously validated antigen-alone Chai prediction.

## Endpoints and interventions

Primary: paired changes in equal-target macro Spearman correlation of negative
full-heavy NLL versus negative supplied log10 KD on the frozen 1,000-row
HER2/PD1/TIGIT panels. These panels were inspected previously; this is an
exploratory mechanism experiment, not fresh confirmation.

Secondary: group-averaged native contact AP, attention AP and attention mass
enrichment, plus heavy/CDR NLL. AP ranks antigen residues separately within
each true contacting antibody residue, then averages within complex and
biological group. True contact labels select evaluation rows only after the
forward pass. Attention is reconstructed passively from Q/K without switching
the model's attention backend; capture and padding parity are checked.

Evaluate every conditional model normally, with routing disabled, and with
only the routed antigen memberships/pair scores permuted. Memory, values,
sequences and auxiliary predictions keep their original alignment. The sham
interventions must have exactly zero effect in `control`. A shuffled signal
tests dependence on alignment; it is an out-of-distribution intervention and
does not by itself prove a biological mechanism.

Figures show sample SD across paired optimization seeds. Native paired effects
also receive a 95% bootstrap interval resampling biological groups, conditional
on these fitted seeds; this does not represent training-population uncertainty.
No biological affinity confidence interval is claimed. Separate generation
samples demonstrate full-prefix antigen-only decoding, without target prompts.

The independent validator recomputes correlations with ranked Pearson
correlation and AP with positive-score thresholds, verifies exact training
exposure/objective arithmetic, checks sham interventions, and confirms previous
source files and weights were preserved.

## Completed pilot

All 20 fits completed and passed independent validation. Read
`artifacts/experiments/interface_attention_v1/interpretation.md` for the full
result. Shared tags produced little localization change. Direct bias improved
native attention alignment, but its mean affinity correlation (0.32537) remained
below matched scratch IgLM (0.33202). Disabling or shuffling the direct bias
improved affinity correlation in every seed, so its small gain versus MANGO
control is not evidence that correctly localized inference routing helped.
Previous source files and checkpoints were preserved.
