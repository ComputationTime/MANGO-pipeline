# Small antigen-conditioning diagnostics

These experiments investigate the HER2-free one-epoch MANGO/IgLM prototype.
They preserve its cohorts, target masks, cached frozen ESM2 tensors and original
checkpoints. No larger ASD training run is launched. All GPU jobs request
`gpu=1` and run sequentially; annotation/summary use their isolated environments.

Run `./run_gpu.sh antigen-diagnostics` for six checkpoint probes and the region
analysis. It scores all 56 validation and 426 test records, plus a seeded
96-record training inspection subset balanced across existing `ab_ag_cluster`
groups. This subsample is an inspection set; no split assignments are changed.
Primary test evaluation uses the same 362 records and 54 groups as the pilot.

For each MANGO checkpoint, score the original antibody target with:

- Its correct antigen.
- Two deterministic unrelated antigen draws, both from the same split and a
  different antigen sequence and `ab_ag_cluster`. Donors are selected from the
  nearest-length candidates. Prefix memory is interpolated to its original
  token count so changing antigen content does not also shift target positions.
- Antigen context removed. This is an inference intervention and can be outside
  the training distribution; it is not a separately trained unconditional model.
- Mean-pooled antigen memory. Cross-attention uses one memory token; the prefix
  probe repeats the mean to the original length to preserve target positions.

Every per-token scorer must reproduce the saved official validation/full-test
NLL, examples and token counts before its completion marker is written. Scores
include target residues and EOS. ANARCI labels targets using IMGT numbering
solely for analysis; region labels are never model inputs. Inclusive CDR ranges
are 27–38, 56–65 and 105–117, retaining insertions, per the
[IMGT definition](https://www.imgt.eu/IMGTrepertoire/Proteins/rules_Proteindisplay_V.pdf).
Domain overhangs remain explicitly unassigned, rather than being silently
classified as framework or removed from overall NLL.

Run `./run_gpu.sh antigen-diagnostic-training` for eight follow-up models:

| Arm | Training change | Budget |
|---|---|---|
| `prefix_rebased` | Original full antigen prefix; antibody selector starts at absolute position 0, BOS at 1, independent of antigen length | Fresh one-epoch ASD + one-epoch native |
| `prefix_causal_control` | Original target positions with the same explicit causal mask as rebasing, controlling for mask backend/dropout differences | Fresh one-epoch ASD + one-epoch native |
| `cross_pooled` | Mean-pool ESM2 into one cross-attention memory token | Fresh one-epoch ASD + one-epoch native |
| `cross_continue` | Continue the existing native cross-attention model | One additional native epoch |
| `iglm_continue` | Continue the existing native IgLM model | One additional native epoch |
| `iglm_cross_frozen` | Add zero-output antigen cross-attention to trained IgLM; freeze every original decoder parameter | One additional native epoch |
| `iglm_cross_frozen_shuffled` | Same frozen adapter initialization, trained with unrelated antigen pairings | One additional native epoch |
| `iglm_cross_joint` | Same zero-output branch, update both decoder and antigen branch | One additional native epoch |

All stages use ordinary CE, fresh Adam with learning rate `1e-4`, seed 0,
both chain targets and the complete original training target masks. The extra
native epochs use the seed-0 epoch-1 ordering. New ASD/native arms use epoch-0
ordering, matching the original prototype. Frozen paired/shuffled arms preserve
the original memory length during training, matching attention/dropout tensor
shapes. Validation always uses correct antigens.

New IgLM branches retain its native architecture/vocabulary and full trained
decoder. Only added cross-attention output projections are zero-initialized.
Actual initial validation NLL must match the source IgLM NLL before training.
The report checks frozen base tensors against the source checkpoint exactly,
as well as one-pass training membership, source hashes, scored target/token
counts and validation-checkpoint NLL. Experimental checkpoints have a separate
format and are not loaded by ordinary production generation.

The rebased prefix passes an explicit all-valid attention mask. Transformers
4.57 otherwise interprets the position-ID reset as a packed-sequence boundary
and disconnects antigen from antibody. A regression check requires nonzero
antigen gradients and verifies that future antibody residues remain masked.
The first disconnected position run is archived separately and excluded from
the reported position result.

All outputs live under `artifacts/experiments/antigen_conditioning_diagnostics_v1/`:

- `diagnostics.png`, `summary.json`, `paired_comparisons.csv`: antigen sensitivity,
  region differences and observational length associations.
- `probes/*/token_scores.jsonl`: target-level residue/EOS NLL and donor identities.
- `regions.json`: complete target annotations and explicit unassigned positions.
- `training/*/{asd,native}/`: independent checkpoints, live curves and metrics.
- `training/training_diagnostics.png`, `training/summary.json`: controlled
  follow-up results, separated by the amount of native training.

Uncertainty uses a paired bootstrap of entire `ab_ag_cluster` groups (2,000
resamples), retaining the token-weighted estimand. These are exploratory,
single-initialization experiments; the intervals capture held-out cluster
variation, not training-seed uncertainty, and are not adjusted for multiple
comparisons. Compare equal training budgets. Improvements from a shuffled
antigen adapter do not establish antigen-specific binding information.
