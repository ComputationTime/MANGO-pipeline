# AbRank pretraining and CATH-protected SAbDab2 fine-tuning

`./run_gpu.sh abrank-transfer` trains eight conditioning controls on an audited
AbRank cohort, then initializes each SAbDab2 model from its matching AbRank
checkpoint and fine-tunes it. It uses separate Snakemake invocations and isolated
ESM2, model, and plotting environments; GPU rules have `gpu=1`. No downstream
structure-prediction or generation jobs are part of this focused NLL experiment.

The configuration is `config/abrank_transfer.yaml`; generated data and results
are under `artifacts/experiments/abrank_cath_transfer_v1/`. Existing Figure 8 and
Figure 9 inputs, trained runs, and plots are never outputs of this workflow.
The experiment manifest pins the configuration and split audit; changing these
requires a new experiment root rather than replacing an existing experiment.

## Cohorts and contamination controls

The downloaded Kaggle v4 table contains 342,356 rows, but 192,559 RBD-escape
rows store mutation labels rather than amino-acid sequences in `Ag_seq`.
Those rows are excluded. The large HIV and SARS-CoV-2 groups overlap protected
SAbDab2 test antigen families and are also excluded. After sequence validation,
CATH filtering, IMGT variable-domain extraction, and exact H/L/antigen triple
deduplication, 901 usable records remain:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| AbRank | 817 | 21 | 63 |
| SAbDab2 transfer | 509 | 56 | 426 |

The primary SAbDab2 test subset contains 362 records whose antigen chains have
confident CATH assignments. All 426 original test rows remain unchanged in the
secondary result; 64 lack confident CATH assignments. The original SAbDab2
records file is never rewritten.

“CATH family” means the four-level homologous superfamily code C.A.T.H, not the
three-level fold or the finer S35 sequence family. Sequence hits transfer domain
assignments from the pinned CATH 4.4.0 S100 reference. Native PDB antigen-chain
annotations from the downloaded daily snapshot are added conservatively; the
snapshot retains its reported v4_3_0 version. Antibody-chain CATH assignments
are never used as antigen annotations.

Confident assignments require at least 35% identity, 80% CATH-domain coverage,
and E ≤ 1e-5. Exclusion uses the union of possible domain matches at 20%
identity, 60% domain coverage, and E ≤ 0.001. Development records without a
confident assignment for every antigen chain are quarantined. Any family
shared with any assigned domain of any of the 426 protected tests excludes a
candidate, including its entire SAbDab2 `ab_ag_cluster` when applicable.
A direct antigen-sequence screen additionally excludes possible test homologs
at 20% identity, 60% coverage of either antigen, and E ≤ 0.001, including matches
to the unclassified tests. Exact antibody target-chain and PDB overlap with the
protected tests is forbidden.

AbRank connected components join shared antigen CATH families, upstream
`Ag25_cluster`, and exact individual antibody target sequences. Multi-domain
antigens join all their families transitively. The largest component stays in
training. Seeded, deterministic subset-sum selection targets 10% of rows for
test and then validation, with each holdout capped at 10% of independent
components so the dominant HER2 group does not put every other family into
holdouts. All split choices precede model training and use counts, not model
outcomes.

SAbDab2 removes unsafe source groups in full. Its validation split is chosen
from independent family/antibody/source-group components absent from both
AbRank training and checkpoint-selection validation. Thus fine-tuning
validation has not been used in pretraining development. SAbDab2 source
`ab_ag_cluster` identifiers remain intact and no retained group crosses splits.

`cohort/split_audit.json` records source/output SHA-256 digests, exclusions,
counts, thresholds and pairwise checks. `cohort/exclusions.csv` records a reason
for every rejected row. CATH disjointness is verified against these pinned
annotations and sequence screens; unannotated remote homology is not a proven
negative. This is the reason for reporting the assigned subset as primary.

## Model and evaluation protocol

The eight arms are the ESM2 prefix baseline, full-shape constant/random,
one-token constant/random, mean-pooled ESM2, shuffled ESM2, and cross-attention
ESM2. ESM2 650M stays frozen. Only antigen sequences enter biological
conditioning; heavy/light sequences are IMGT variable-domain targets and a
learned selector requests the output chain.

Shared architecture, seed, optimizer, epoch limit, learning rate, and early
stopping match the preceding conditioning study. Cross-attention retains its
additional attention parameters. Every transfer run loads all matching decoder,
projection and selector weights from the best AbRank validation checkpoint,
checks architecture/task/width/target/control compatibility, and starts a fresh
optimizer. No original SAbDab2-trained checkpoint initializes this experiment.
Source checkpoint hashes and source cohort fingerprints appear in checkpoints,
model configuration and evaluations.

NLL is target-token weighted over VH/VL residues plus EOS. Antigen context is
never scored. Test sets do not select checkpoints. Transfer evaluations include
`test_cath_assigned` as primary and `test` (all original records) as secondary.

The new plots and their numerical CSV files are:

- `plots/abrank_controls.png` / `.csv`: AbRank train and internal test NLL.
- `plots/sabdab2_finetuned_controls.png` / `.csv`: SAbDab2 fine-tuning train,
  CATH-assigned test, and full secondary test NLL.

`results_report.json` and `.csv` retain partial results when an independent
arm fails. Logs are under `logs/`; each model retains its live training curve.
Rerun the same launcher command to continue incomplete arms. After completion,
`python3 workflow/scripts/verify_abrank_transfer.py --root
artifacts/experiments/abrank_cath_transfer_v1` independently checks pinned input
hashes, original test/figure preservation, split overlaps, checkpoint selection,
transfer initialization, evaluation membership/token counts, and plotted NLLs.

The strict filter retains a small training set: 758 of
817 AbRank training rows are from the HER2 AbSci collection. The AbRank test
contains only five independent connected components. Treat this experiment as
a limited transfer test, not evidence that AbRank solves the antigen-diversity
problem. Filtering also changes the SAbDab2 training cohort relative to the
earlier figures, so those figures cannot isolate the effect of AbRank
pretraining. Within each new figure, all controls share exactly the same
training and evaluation cohort.

The AbRank prefix context averages 602 tokens in training versus 257 in test.
Prefix models consequently also face a shift in the absolute decoder positions
used for target residues; cross-attention and one-token controls retain their
short decoder prefixes. These models differ in both conditioning and context
layout, as in the earlier control study.

A within-split record shuffle can also retain the identical antigen
sequence in this imbalanced cohort; `biological_pairing_audit.json` reports the
actual matching-antigen fraction per split instead of assuming every donor ID
breaks the biological pairing.

## Rebuilding cohort preparation

1. `prepare_cath_sequences.py` snapshots the original tests and creates unique
   antigen queries and their row/chain membership table.
2. `mmseqs easy-search` in `workflow/envs/cath_split.yaml` searches the CATH S100
   FASTA with `-s 7.5 -e .001 --min-seq-id .2 -c .6 --cov-mode 1 --max-seqs 5000`.
   Export columns `query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov`
   without a header as `cath_sequence_hits.tsv`. A second search against unique
   protected-test antigen sequences, without `-c` or `--cov-mode`, writes
   `protected_antigen_hits.tsv` with the same columns. Use eight threads and a
   bounded MMseqs memory budget; these searches do not require the GPU.
3. `number_abrank_candidates.py`, in the isolated ANARCI environment, creates
   `numbered_antibody_chains.json` using IMGT numbering. Ambiguous/missing domains
   are quarantined rather than labeling full deposited chains as variable targets.
4. `prepare_abrank_transfer.py` builds both records files and the split audit.
   Its `--seed 0` is recorded. Source changes are rejected against the original
   input snapshot. Freeze the outputs before running training.

Reference download URLs, sizes and SHA-256 digests are retained under
`artifacts/data/cath/reference_manifest.json`. Downloaded release provenance is
under `artifacts/data/abrank/`. No generated reference, dataset, model weight,
or result should be committed.
