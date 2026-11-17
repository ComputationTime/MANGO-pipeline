# External antigen pairing with the frozen 16k models

Run or resume with:

```bash
./run_gpu.sh openalphaseq-pairing
```

This evaluates the five completed, one-epoch, scratch-trained ASD 16k models:
regular prefix MANGO, fixed-position prefix MANGO, cross-attention MANGO,
the shuffled-training cross-attention control, and antigen-free IgLM.
It does not retrain or change their checkpoints. The four conditioned models
use the same frozen ESM2 650M antigen representations as during training.
Only antigen representations and the learned heavy/light selector enter
biological conditioning. Antibody residues are teacher-forced output targets.

The configuration is `config/openalphaseq_pairing.yaml`. The workflow requires
the existing 16k training CSV, split audit, checkpoints, and training markers.
It checks checkpoint hashes, common cohort fingerprints, and checkpoint epoch.
The official OpenAlphaSeq snapshot is downloaded automatically if absent;
see [the downloader](openalphaseq_download.md) for its pinned source revision.

For interpretation of the interim results and proposed improvements, see
[recommendations for antigen-conditioned pairing](openalphaseq_recommendations.md).

## Candidate proteins

The requested target-protein panel is the union of the actual nonempty target
sequences in the optimization experiments: **12 proteins**. These comprise
eight coronavirus RBD proteins/variants, human and mouse TIGIT, PD1, and HER2.
The 2,431 individual RBD mutagenesis targets from YM_0005 are not additional
candidates in this first comparison. Antibodies from all nine experiments
remain eligible. An observation contributes a label only when its exact target
sequence belongs to the 12-protein panel.

`antigens.csv` contains the exact protein sequences, original publisher names,
and target families. `scoring_manifest.json` freezes the antigen-column order.

## Removing overlap with training

OpenAlphaSeq expresses both single-domain antibodies and scFvs, while the
training objective scores separate variable domains. Comparing complete scFvs
with the training VH/VL strings would miss overlap. The workflow therefore:

1. Splits the explicitly expressed `GGGGSGGGGSGGGGS` scFv linker when exactly
   one match leaves plausible 70–170-residue components on both sides.
2. IMGT-numbers every expression component and every distinct **training-split**
   VH/VL with ANARCI. Each component must contain exactly one assigned domain.
   Single-domain constructs must be heavy domains; scFvs must have one heavy
   and one light domain, in either orientation. Non-antibody controls,
   noncanonical residues, and unassigned/multiple domains are excluded.
3. Compares canonical domain sequences against both the original and
   IMGT-normalized training chains. It also excludes an expressed component
   containing a complete training chain as a contiguous substring. This catches
   terminal expression overhangs. The match is against any training antibody,
   regardless of its antigen or assigned chain.
4. Consolidates identical normalized H/L pairs, merging their observations.
   Antibodies with every domain already seen are excluded. Seen chains of an
   otherwise retained antibody do not contribute to its reported NLL.

The snapshot contains 251,418 unique expressed antibody strings. Five have
only training-seen domains, one contains an invalid residue, and one is the
ACE2 protein control. The 251,411 retained strings consolidate to **251,387
unique antibodies**, all with novel heavy domains. Their novel domains give
251,410 distinct chain targets. **193,682 antibodies** also pass the stricter
condition that every expressed antibody domain is unseen.

| Source target family | Novel-heavy antibodies | All domains unseen |
|---|---:|---:|
| RBD | 66,631 | 66,630 |
| TIGIT | 62,624 | 62,624 |
| PD1 | 64,428 | 64,428 |
| HER2 | 57,704 | 0 |

The HER2 library shares a light chain present in training; an expression tail
initially hides this match. Its unseen heavy domains remain eligible for the
primary heavy-chain evaluation. RBD has one antibody with a training-seen light
domain. The strict cohort therefore covers three target families, and its
macro accuracy must not be compared with the four-family cohort without noting
the different family composition.

These are **sequence-disjoint** results. Close relatives, shared germlines,
and parental antibodies can still occur across datasets. Large optimization
libraries contain many related variants of relatively few parental antibodies.
No row-level confidence intervals are used to imply independent biological
replication.

## Ground truth and metrics

A reliable positive observation has finite affinity, `above_background=True`,
and `sufficient_replicate_observations=True`. A reliable background-negative
has finite affinity, sufficient replicates, and `above_background=False`.
Missing affinity or insufficient replicates are uncertain. If reliable positive
and negative observations conflict, **any positive makes the pair positive**.
This preserves binding evidence despite conflicting affinity measurements.
The normalized cohort contains 431,838 positive, 97,918 negative and 220,748
uncertain antibody/protein pairs; 360 pairs have conflicting reliable labels.

Every retained novel chain is scored against every candidate protein. NLL
includes all variable-domain residues and EOS; lower is better. Separate IMGT
CDR3 NLL excludes framework residues and EOS. Scores for a novel-domain composite
sum chain losses and divide by the combined target-token count. A training-seen
light chain is never added to that composite. The strict composite additionally
requires every expressed chain to be novel.

The primary cohort is novel heavy chains. Results distinguish:

- **Antigen-family top-1**, balanced equally across represented families.
  Family scores average NLL over the candidate proteins in that family. This
  avoids weighting the eight RBD candidates as eight independent families.
  The four-family tied baseline is 25%.
- **Protein top-1 and MRR** among all 12 candidates, crediting any reliably
  positive protein when an antibody binds multiple targets. An unmeasured
  alternative is unknown, not an experimentally proven negative. This is
  annotated-partner retrieval.
- **Within-antibody AUROC** comparing only the antibody's explicitly measured
  positive and background-negative proteins. Only antibodies with both types
  of label contribute. HER2/PD1 have just one measured target, so they cannot
  individually support this metric.
- **Within-protein binder AUROC** comparing measured positive and negative
  antibodies. Results include raw negative NLL and `IgLM NLL − model NLL`.
  The latter asks whether conditioning improves on the antigen-free antibody
  prior. These binder-classification results measure a different property
  from choosing an antigen for one antibody.

IgLM receives the same score for all candidate proteins. Top-1 and MRR give
the exact expected credit under uniform ordering of ties; taking the first
`argmin` would introduce arbitrary antigen-order bias. Comparisons use a
3e-6 nats/token tolerance for negligible floating-point differences.

Raw NLL is primary. A separate, explicitly labeled `column_centered` diagnostic
subtracts each protein's mean NLL across the evaluated queries without using
labels. This is transductive calibration, not a separately trained model.
Per-family results and the shuffled-training control reveal when a model
mostly prefers one antigen rather than reliably distinguishing partners.

`specificity_controls.csv` also compares measured-pair AUROC with constant
global/library mean antigen scores and 16 random antibody reassignments within
the same source target library. Each reassignment uses a cyclic shift without
self assignments. If the measured-pair score remains strong after reassignment,
the result can be explained by a generic antigen preference instead of
antibody-specific compatibility. Reassignment variation is a diagnostic; it is
not a confidence interval over independent biological lineages.

## Runtime, caching, and outputs

ANARCI runs in its isolated environment. ESM2 embeddings are computed once per
unique antigen and share the existing sequence-keyed cache. GPU scoring
streams committed annotation batches and caches antigen/selector/BOS keys and
values. Sequence-sorted batches additionally share identical teacher-forced
antibody prefixes; those residues still contribute their original target loss.
This changes computation reuse, not biological conditioning or the objective.
Each frozen decoder runs in float32 with highest matmul precision.

Production scalar scores, ragged batches, both chain selectors, all model
architectures, fixed learned positions, CDR3 indexing, and immutable prompt
caches are covered by scorer parity checks. Faster TF32/FP16 modes were
benchmarked but are not used because they changed some antigen rankings.
Scores flush to SQLite after every batch, allowing an interrupted job to
resume without repeating completed model/chain/antigen comparisons. The
one-time v1→v2 cache migration preserves validated full-precision scores and
archives the previous manifest and migration reason.

GPU jobs retain `gpu=1` and run serially. CPU annotation can overlap the GPU
scorer. The download, data, ANARCI, ESM2, and model environments remain isolated.

Outputs are under `artifacts/experiments/openalphaseq_pairing_16k_v1/`:

- `cohort_audit.json`: exact exclusion counts and training-disjointness checks.
- `heldout_antibodies.csv`: normalized H/L sequences, IDs and novelty masks.
- `heldout_chains.csv`: only novel chain sequences eligible for scoring.
- `excluded.csv`, `labels.csv`: exclusions and measured annotation aggregates.
- `{model}_scores.npz`: all query × antigen NLL matrices, CDR3 matrices, query
  IDs, antigen IDs, measured labels and strict/heavy novelty masks. Array names
  beginning `heavy_` use heavy-chain loss; `novel_` use all novel chains.
- `metrics.csv`, `per_family_metrics.csv`, `binder_metrics.csv`, `rankings.csv`:
  comparisons for raw/calibrated scores, all three cohorts and both regions.
- `specificity_controls.csv`: measured-pair AUROC, antibody-independent antigen
  preferences and within-library antibody reassignment, for full-chain/CDR3 NLL.
- `pairing_comparison.png`, `findings.md`, `summary.json`: final comparison.
- `verification.json`: source/cohort hash checks, output hashes, exact IgLM tie
  checks and independent scalar CPU-versus-cached-GPU NLL checks across all
  models, all panel antigens, both chain selectors and all four source families.
- `numbering_progress.json`, `scoring_progress.json`, `logs/`: progress and logs.

The report refuses to claim completion before `scoring_complete.json` exists
and every query has scores from every model. An explicitly requested interim
report goes into `partial/` and records its coverage separately:

```bash
python workflow/scripts/report_openalphaseq_pairing.py --partial
```

Use the isolated model environment for that command. To read a score matrix:

```python
import numpy as np
s = np.load('artifacts/experiments/openalphaseq_pairing_16k_v1/fixed_scores.npz')
# Rows follow s['query_ids']; columns follow s['antigen_ids'].
nll = s['heavy_nll']
# labels: +1 reliable binding, -1 reliable background-negative, 0 uncertain/unmeasured.
labels = s['labels']
```
