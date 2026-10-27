# ASD-only 2,000-to-4,000-record scaling experiment

Run `./run_gpu.sh asd-scaling`. Outputs are isolated under
`artifacts/experiments/asd_scaling_2k_4k_v2/`; earlier MANGO/IgLM and native
fine-tuning experiments remain intact. This experiment trains and evaluates
only ASD 2.0, using positive-binding examples and ordinary cross-entropy.

The prior 512-residue pool contains only 3,694 eligible training records, or
2,761 with the ten-record-per-antigen cap. Exact doubling therefore requires a
broader shared training eligibility rule. Both new sizes use an antigen limit
of **768 residues**, the same HER2/BUZZ exclusions and a cap of ten records per
antigen. Both sizes are trained afresh, rather than comparing an old short-only
baseline with the expanded pool. An abandoned initial preparation under `v1`
was never used for model training.

Within the original protected training split, the preparer selects antigens in
seeded round-robin order, recomputes exact individual-chain target masks, then
takes nested prefixes of that stream. This preserves every small-cohort record
and target mask in the large cohort. Original group/family split assignments
are retained, and the SAbDab2 validation/test protection is rechecked.

| ASD training records | Scored chain examples | Distinct antigens | Distinct heavy/light targets |
|---|---:|---:|---:|
| 2,000 | 3,988 | 1,666 | 1,176 / 1,155 |
| 4,000 | 7,876 | 1,666 | 2,111 / 2,020 |

Both cohorts cover 600 CATH families and 79 split-graph components. The increase
adds antibody examples for the same antigen set. These component counts are
not antibody-lineage counts. Validation remains byte-equivalent at the record
level to the original 95-record/185-target cohort, and test to the original
200-record/393-target cohort. Scored test tokens remain 44,934, including EOS.

At each size, train IgLM, paired ESM2 cross-attention MANGO and shuffled-pairing
cross-attention MANGO for one epoch. All begin from random model weights, use
the original optimizer settings/seed and score both heavy and light targets.
ESM2 650M representations remain frozen; all required tensors already exist.
Only small cohort and shuffled-donor indexes are created.

The training shuffle changes antigen identity within the training split and
preserves the exact multiset of antigen records. Validation/test context stays
correct. A distinct split-graph component cannot be required while preserving
marginals because one training component contains most records; the component
spans many CATH families. In these cohorts, 98.7% / 97.8% of shuffled training
pairs also have disjoint CATH families. Shuffling is a context control and does
not assign experimentally established negative-binding labels.

Evaluate every model on the identical validation/test targets with correct
context. Cross-attention models additionally receive two marginal-preserving
antigen permutations, two donor draws with disjoint CATH families and split
components, and no antigen context. The different-family draws need not
preserve antigen marginals; both kinds of intervention are reported separately.

Two additional 2,000-record models, IgLM and paired MANGO, train for two epochs.
These are approximate optimization-budget controls: 7,976 updates versus
7,876 for 4,000 records once. Their **final** checkpoint is evaluated explicitly
to preserve that budget; their validation-selected checkpoints are retained
separately. One-epoch models evaluate their sole validation-selected checkpoint.

ANARCI annotates held-out targets for analysis only. The report includes
overall, framework, CDR and CDR3 NLL by chain; correct-versus-wrong antigen
effects; paired-versus-shuffled training effects; scale differences; changes in
the MANGO–IgLM gap; and approximate equal-compute controls. Confidence intervals
use 2,000 paired bootstrap resamples of whole `ab_ag_cluster` groups. The test
set contains only ten such groups. Results are exploratory and use one training
seed; intervals do not include initialization uncertainty or multiple-comparison
adjustments.

Inspect `asd_scaling.png`, `summary.json`, `scores.csv` and
`paired_comparisons.csv`. `results_report.json` exposes partial progress.
Each run retains ordinary live training curves, best/latest checkpoints,
per-token scores, target-cohort hashes and evaluated checkpoint provenance.

Recheck the final results using the model environment:

```bash
python workflow/scripts/report_asd_scaling.py --require-complete
```

## 8,000-record extension

Run `./run_gpu.sh asd-scaling-8k`. It trains three fresh, one-epoch models:
IgLM, paired ESM2 cross-attention and shuffled-training ESM2 cross-attention.
Ordinary CE, frozen ESM2, seed and optimizer settings remain the same. It
reuses the completed eight reference runs without changing their configuration,
cohorts, checkpoints or scores. New outputs, including the joined 2k/4k/8k
figure, live under `artifacts/experiments/asd_scaling_8k_v1/`.

The audited source has 598,978 training records before removing HER2/BUZZ,
and **73,586 afterward**. Of these, **72,378 records / 1,666 antigens** meet
the existing 768-residue antigen limit. Recomputing individual-chain masks in
the round-robin order retains 72,375 usable records, or 73,583 without the
prototype length limit. These are available training records after the original
binding, CATH-family, antibody-homology, PDB and split protections; they exclude
held-out records.

The ten-record-per-antigen cap only permits **4,007** usable training records.
The 8k extension therefore raises the cap to 1,500, continues the same seeded
round-robin stream and takes its first 8,000 usable records. It explicitly
verifies that all 2k/4k records, their ordering and their individual target masks
remain identical, and that held-out records retain the same SHA256. HER2/BUZZ
and native reserved-family exclusions are checked again.

The resulting 8k cohort contains 13,176 chain targets (7,953 heavy / 5,223 light),
the same 1,666 antigens, 600 CATH families and 79 training split components.
Two antigens each supply 1,318 records: **32.95%** of the cohort together.
Consequently this tests additional examples with increased antigen imbalance;
it does not isolate dataset size under a fixed balanced antigen distribution.
The one-epoch budget also increases optimizer updates. No additional 8k
compute-matched controls or training seeds are included.

`availability.json` records the counts and concentration. The joined report
verifies each original/new checkpoint, full per-epoch target membership, token
counts, fixed held-out hashes and different-family donors before calculating
paired cluster-bootstrap intervals. Inspect `findings.md`, `asd_scaling.png`,
`scores.csv`, `paired_comparisons.csv` and `results_report.json` in the new root.

```bash
python workflow/scripts/report_asd_scaling_8k.py --require-complete
```

## 16,000-record extension and regular MANGO

Run `./run_gpu.sh asd-scaling-16k`. Seven new models train from random weights:
16k IgLM, paired cross-attention and shuffled cross-attention, plus **regular
MANGO at each of 2k/4k/8k/16k**. Regular MANGO uses the supported
`antigen_chain_gpt2_v3` architecture: projected ESM2 antigen tokens precede the
learned chain selector and antibody decoder tokens. It retains the standard
absolute target positions. Cross-attention uses its existing separate antigen
memory. Every new run uses ordinary CE and exactly one epoch, with the original
seed/optimizer settings and frozen ESM2 650M. Existing checkpoints and evaluation
scores are reused. All GPU rules remain serialized.

New artifacts live under `artifacts/experiments/asd_scaling_16k_v1/`. The joined
report verifies all 18 runs: eleven existing references and seven new models.
The figure contains **two panels**, overall test NLL and CDR3 NLL. The wrong
antigen probes, uncertainty estimates and old optimization controls remain in
the numerical outputs, with their former bottom panels removed.

The 16k cohort preserves the literal 8k/4k/2k prefixes, including each chain mask,
and the same validation/test cohort and family/group protections. The antigen
limit remains 768 residues. Raising the per-antigen cap to 10,000 supplies the
full 16k: one antigen contributes 8,719 records, another 1,917. They jointly
account for **66.475%**. The cohort still covers 1,666 antigens and supplies
21,176 training chain targets (15,953 heavy / 5,223 light). Every added 8k record
contributes a new heavy-chain target; the light-chain target count is unchanged.

A fully deranged antigen shuffle with unchanged record marginals is impossible
when one antigen occupies more than half the records. For largest antigen count
`L` in a cohort of `N`, at least `max(0, 2L − N)` records must retain that antigen.
The maximum shuffle achieves this bound: at 16k, **1,438 records / 8.9875%**
retain their correct antigen; the other 91.0125% change identity. Donors remain
within train and preserve the exact multiset of antigen records. Earlier sizes
have zero retained correct identities and use the same mappings as their
original controls. The figure and `availability.json` disclose the partially
shuffled 16k control; it carries some true-pairing signal.

For regular MANGO's wrong-antigen probes, donor representations are linearly
interpolated to the recipient antigen length to preserve target token positions.
Correct-context scoring uses the original unmodified embeddings and must
reproduce checkpoint validation NLL. Cross-attention probes retain their existing
protocol. ANARCI annotates held-out targets for analysis only.

The report checks all nested cohorts, frozen input/configuration hashes, complete
per-epoch target membership, held-out score membership and different-family
donors. The same single-seed, ten-test-group and increased-compute limitations
apply, with stronger antigen imbalance at 16k. Reproduce only the joined report
inside the model environment using:

```bash
python workflow/scripts/report_asd_scaling_8k.py --config config/asd_scaling_16k.yaml --require-complete
```

## Fixed-position prefix MANGO

Run `./run_gpu.sh asd-scaling-fixed-positions` to add four one-epoch models on
the exact same 2k/4k/8k/16k cohorts. They use the improved fixed-position prefix
from the earlier controlled diagnostics, starting from random weights under
the current ASD-only CE protocol. All full-length frozen ESM2 tokens remain
present. The learned chain selector has position 0, BOS has position 1 and
antibody residues begin at position 2, regardless of antigen length.

The explicit all-valid attention mask preserves ordinary causal attention across
the antigen/antibody boundary. It prevents Transformers from interpreting the
position reset as a packed-sequence boundary. Production loss and scoring match
the corrected diagnostic implementation; tests check antigen gradients and
future-token masking. The original prefix comparison also differs in mask
handling, so its entire NLL difference should not be attributed solely to the
position reset.

The architecture has a distinct checkpoint identity,
`antigen_chain_fixed_position_gpt2_v1`. Checkpoint loading restores the position
policy. Cached generation keeps physical KV-cache locations monotonic while
resetting learned antibody position indices; generation tests compare every
step's logits with full teacher-forced forward passes.

The existing 18 runs are read as frozen references. Only four new models train,
for 22 total evaluated runs and five curves in each of the two figure panels.
Results live under `artifacts/experiments/asd_scaling_fixed_positions_v1/`.
Training, held-out sets, HER2 exclusions and antigen-balance limitations remain
as above. Rebuild the joined report in the model environment using:

```bash
python workflow/scripts/report_asd_scaling_8k.py --config config/asd_scaling_fixed_positions.yaml --require-complete
```
