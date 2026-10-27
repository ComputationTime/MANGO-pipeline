# ASD 2.0 pretraining → SAbDab2 fine-tuning

Run `./run_gpu.sh asd-transfer` for the pretraining comparison using ASD instead
of AbRank. Configuration lives in `config/asd_transfer.yaml`; all new cohorts,
weights, logs and figures live in `artifacts/experiments/asd_cath_transfer_v1/`.
The AbRank experiment and existing Figures 8 and 9 remain available.

The launcher first prepares the cohort from the downloaded, audited ASD 2.0
Delta version 12 snapshot. It consumes:

- `artifacts/data/asd/sabdab2_family_audit_v1/antigen_rows.sqlite` and its
  antigen FASTA, CATH annotations and eligibility decisions;
- `artifacts/data/asd/sabdab2_binding_any_affinity_v1/counts.json`, which pins
  the read-only audit inputs and the revised any-binding policy;
- the fixed SAbDab2 transfer cohort under
  `artifacts/experiments/abrank_cath_transfer_v1/cohort/`. This supplies native
  splitting and test coverage, with no AbRank pretraining records or weights.

The [ASD downloader](asd_download.md) obtains the raw snapshot. Downloading alone
does not recreate the scientific family/target audit; the preparation script
requires those audited inputs and rejects missing or changed inputs. A prepared
cohort is immutable with respect to its source hashes and preparation settings.

## Binding and sequence policy

Keep any qualifying positive binding observation. Differing reported affinities
do not reject an otherwise supported sequence pair. BUZZ HER2 high, medium and
low FACS affinity bins all qualify. Negative-only, assay-floor-only, missing and
uninterpreted relative-assay-only rows do not qualify. This is binding evidence
for the source antibody, not proof that each isolated output chain binds alone.

Retain paired antibodies with validated ASD RIOT IMGT variable-domain targets.
These use `primary_sequence_aa`, not full deposited chains; an independent
ANARCI re-numbering has not been performed. Exact antigen/VH/VL triples are
deduplicated across sources. After splitting, each exact `(antigen sequence,
requested chain, target sequence)` becomes one training/evaluation example.
`example_target_chains` records which targets remain on each row, so repeated
light chains in BUZZ are not scored hundreds of thousands of times. Affinity
and source labels are never conditioning inputs or extra examples.
Record IDs use the minimum qualifying source row in the frozen snapshot, not a
target-sequence hash; random-control seeds therefore do not derive from targets.

The audited pool contains 601,886 paired triples with BUZZ, or 77,568 without.
Protecting the fixed native validation set removes 68 more triples. Another 318
exceed the common decoder capacity bound, leaving 601,500 paired triples with
BUZZ, or 77,182 without, before internal holdout allocation. An optional
`--without-buzz --output <new-cohort-directory>` on `prepare_asd_transfer.py`
prepares the alternative; use a separate experiment root and matching config
to train it. The default includes BUZZ and retains ordinary token NLL loss;
no focal/Dice objective is introduced in this dataset substitution.

## Splits and comparability

All original 426 SAbDab2 tests remain fixed, with the same 362-record
CATH-assigned primary subset and 64 additional secondary-only records. The
native fine-tuning split remains 509 training / 56 validation records.

The ASD audit excludes protected native test CATH homologous superfamilies,
direct antigen homologs, exact antibody chains and PDB overlap. Unknown antigen
families are excluded. Preparation additionally excludes native fine-tuning
validation families, exact targets, antigen sequences and PDBs from **all** ASD
splits. CATH assignments and thresholds match the [transfer protocol](abrank_transfer.md).

Internal ASD connected components join every shared CATH family and exact
individual antibody target, including transitive connections across antigens.
Components cannot cross splits. The largest stays in training; seeded count
selection targets 10% of rows per holdout, capped at 10% of independent groups
per holdout to retain training diversity. The actual counts are:

| ASD split | Paired triples | Rows after individual-chain deduplication | Scored chain examples | Independent groups |
|---|---:|---:|---:|---:|
| Train | 598,993 | 598,978 | 605,585 | 87 |
| Validation | 95 | 95 | 185 | 11 |
| Test | 2,412 | 2,410 | 2,710 | 11 |

Training contains 1,925 distinct antigen sequences, but 524,346 paired triples
have a BUZZ observation. Exact target sharing and multi-domain family matches
connect most of the pool into one component. Row counts therefore do not
measure independent diversity, and the internal validation set is small.
The split audit reports both paired and individual-chain counts explicitly.

All arms use the same whole-antigen capacity filter. The longest eligible
target determines an antigen bound that fits the decoder's 2,048 positions,
including selector/BOS/EOS, even when a shuffled donor supplies a different
antigen. Oversized examples are excluded with a recorded reason, rather than
truncating sequences or allowing only some arms to use them.

## Training and outputs

The same eight controls, model initialization, seed, optimizer, learning rate,
10-epoch limit and early stopping are retained. Frozen ESM2 650M encodes only
antigens. The requested H/L selector is learned; antibodies remain decoder
targets. Fine-tuning loads each arm's best ASD checkpoint, validates its cohort
and architecture, and starts a fresh optimizer. Native NLL scoring is unchanged.

Each unique ASD antigen is embedded once. A read-only SQLite index resolves
record embeddings into that cache; pooled and shuffled controls share tensors,
and constant/random tensors are generated deterministically on load. This
avoids storing a full matrix for every antibody variant. The shuffled control
records its actual identical-antigen fraction in `biological_pairing_audit.json`.

The launcher serializes GPU work and attempts each arm separately, retaining a
partial `results_report.json` / `.csv` on failures. Rerun the same command to
reuse prepared inputs, frozen tensors and completed jobs. Full ASD training is
substantially larger than the previous 817-record AbRank training run.

New figures and matching numerical CSVs are `plots/asd_controls.png` and
`plots/sabdab2_finetuned_controls.png`. Their titles identify ASD pretraining;
the native figure keeps primary and secondary tests separate.

Validate preparation without training:

```bash
python3 workflow/scripts/prepare_asd_transfer.py
python3 workflow/scripts/verify_abrank_transfer.py \
  --root artifacts/experiments/asd_cath_transfer_v1 --cohort-only
.snakemake/gpu-driver/bin/snakemake -s workflow/asd_transfer.smk \
  --profile workflow/profiles/gpu --dry-run
```

Omit `--cohort-only` after training to verify completed evaluations, source
checkpoint hashes, token counts and plotted values as well.
