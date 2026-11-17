# Antigen geometry by contact-supervision experiment

```bash
./experiments/run_experiment.sh interface-supervision-smoke
./experiments/run_experiment.sh interface-supervision
```

The new experiment root is `artifacts/experiments/interface_supervision_v1/`.
It preserves the prior models, study protocols and data protections. The
configuration is `experiments/config/interface_supervision.yaml`.

## Factors and controls

Four MANGO conditions separate antigen geometry from contact supervision:

| Arm | Antigen graph | Contact gradients into generator |
|---|---|---|
| `seq_ce` | Sequence neighbors | No; head-only probe |
| `seq_contact` | Sequence neighbors | Yes |
| `geom_ce` | 3D neighbors | No; head-only probe |
| `geom_contact` | 3D neighbors | Yes |

All four conditional models have identical architecture and initial tensors. A fifth arm,
`iglm_ce`, is an antigen-free scratch IgLM whose feed-forward width matches
MANGO's generation parameter count within 0.1%. Contact-head parameters are
reported separately because they do not participate in sequence generation.

Every MANGO model trains the identical contact-prediction head on exactly the
same labels. CE controls detach the decoder states and antigen memory at the
head input, preventing its loss from updating the generator. Contact arms
allow the same loss to update the shared decoder and antigen encoder. This
avoids comparing a trained head with an untrained random head.

## Antigen input and geometry

Keep one frozen ESM2 650M embedding per antigen residue. Remove interchain CLS
break tokens consistently from every arm; use real residues only. No fixed
32-token pooling is applied. Full residue memory is common to all four arms,
so its improvement relative to the older pooled model is not isolated here.

Use an identical two-layer residual graph encoder in all MANGO models. Each
layer aggregates 64-dimensional projected neighbor features using gates from
16 distance radial basis features, a same-chain indicator, and sequence
separation. The sequence control uses up to 16 neighbors along the same chain,
with 3.8 A per sequence step as a virtual distance. The geometry condition uses
16 nearest C-alpha neighbors and actual distances, including antigen multichain
contacts. This is a simple invariant graph encoder, not a pretrained structure
model. Tests verify rotation/translation invariance and that the sequence
control ignores coordinates.

All records with the same exact antigen sequence use one canonical antigen
conformation, chosen deterministically by record ID. Thus, different target
antibodies cannot be identified through their own bound-antigen conformations.
Graph inputs read antigen atoms only. Antibody coordinates are accessed only
by a separate routine that writes contact-label files. A test moves antibody
atoms while leaving antigen input unchanged and verifies that only labels
change.

The native input is experimental antigen geometry. External HER2/PD1/TIGIT
structures are cached Chai predictions of antigen alone; their manifests
explicitly exclude antibody and affinity inputs. This source difference is an
important limitation of the affinity transfer analysis.

## Supervision and optimization

Contact targets are native non-hydrogen atom pairs within 5 A, mapped exactly
onto observed variable-domain and antigen sequences. The head uses decoder
states after consuming each generated residue and the antigen memory. It is an
output head; no antibody embedding, structure or interface mask enters the
generator's conditioning input. Sequence generation remains causal and uses
the learned heavy/light selector.

Contact BCE gives equal weight to mean positive and negative losses within an
example. An example without a positive contact contributes zero contact loss;
it remains in sequence training. The fixed contact coefficient is 0.1 for both
head-only probes and shared-backbone supervision. All models use Adam at 1e-4.

Five seeds (300–304) and fixed group-balanced schedules are shared by all arms.
Each model gets 1,000 native CE updates followed by 300 updates with the same
native batches and the same RNase-A pair CE exposure used previously. Pair CE
has weight 0.25. No measured-affinity ranking term is included. Contact
supervision/probe training is active throughout. There are 25 complete fits;
no checkpoint selection or early stopping is based on any held-out outcome.

The cohort is unchanged: 99 deduplicated train records in 33 protection groups,
36 held-out records in 16 groups. Training includes the previously selected
heavy/light examples; the primary native evaluation uses 35 heavy examples.
No original `ab_ag_cluster` or protection component crosses splits, and no
prior reserved component enters training. This remains a small exploratory
study, not a replacement for the infeasible confirmatory study.

## Endpoints

The primary endpoint is **within-heavy-residue contact average precision**:
for each true contacting antibody residue, rank all antigen residues by
predicted contact score; average these AP values within the complex, within
biological group, and across groups. This tests epitope localization while
controlling for simply identifying CDR/paratope residues. Whole-matrix AP and
native heavy/CDR NLL are secondary endpoints. Contact AP, rather than accuracy,
avoids rewarding an all-negative classifier on sparse contact maps.

Geometry models are also evaluated with antigen coordinates permuted among
residues within each chain, leaving sequences and true contact labels fixed.
This is an inference intervention that tests use of geometry-residue alignment.
It is not an alternative native structure or a new training condition.

The secondary affinity endpoint is equal-target macro Spearman correlation of
negative full-heavy NLL with negative supplied log10 KD on the same frozen
1,000-row HER2/PD1/TIGIT panels. An additional pre-implemented, uncalibrated
contact readout is mean CDR log-mean-exp contact logit over antigen residues.
Its definition is recorded in `scoring_addendum.json` before any evaluation
completed. It is not a binding-energy model or a model-selection criterion.

Predeclared factorial effects compare geometry without shared contact
gradients, contact gradients without geometry, their conditional counterparts,
and the interaction. All error bars are sample SD across five paired
optimization seeds. There are no biological confidence intervals, p-values or
confirmatory claims from the 16 native groups or previously inspected affinity
panels. Each trained run flushes its CSV every update and refreshes its curve
plot every 100 updates and at completion.

Independent validation in the isolated analysis environment:

```bash
python experiments/scripts/validate_interface_supervision.py
```

It checks frozen inputs and weights, canonical antigen inputs, split boundaries,
exact training exposure/token counts, logged loss coefficients, rank
correlations and AP via an independent positive-threshold calculation. It also
reports the exact finite-sample expectation for uniformly random within-row
rankings as a descriptive reference. Further data and broader structural
architectures would be needed to establish generality.
