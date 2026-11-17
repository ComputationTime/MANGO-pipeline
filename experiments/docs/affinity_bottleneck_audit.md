# Affinity data and grouping audit

Run from the repository root:

```bash
./experiments/run_experiment.sh affinity-bottleneck-audit
```

This mode uses the existing isolated analysis environment and completed,
hash-verified homology searches. It performs no model fitting, changes no
existing biological groups, and supplies no antibody features to a model.
Results are under `artifacts/experiments/affinity_curation_v1/`.

`audit_protocol.json` freezes input and implementation hashes. A changed input
or implementation requires a new experiment root. The separate lineage review
has its own protocol. Existing native/adapter studies remain intact.
For a new root, set `MANGO_AFFINITY_AUDIT_CONFIG` and
`MANGO_AFFINITY_LINEAGE_CONFIG` to corresponding new configuration files.

## What is measured

The grouping census reconstructs both the 115-component inventory and the
53-component full graph exactly. Factor nodes retain the original source
cluster, exact antigen, CATH family, heavy sequence, and light sequence.
Antibody alignments retain the frozen 90% identity and 80% bidirectional
coverage thresholds. Every propagated antigen exclusion has a shortest typed
witness path in `antigen_exclusion_witnesses.csv`.

`graph_ablation.csv` isolates removal of individual relations. The resulting
counts are diagnostic upper bounds; they never become a new training split.
For example, removing CATH still leaves 3,058 records in the antigen-hit
component, and removing near-antibody links leaves 2,653. Broad connections
in more than one relation therefore matter. Source clusters remain intact in
all diagnostic partitions.

The lineage review independently traverses the graph with BFS, checks every
propagation witness, and annotates near-antibody edges with cached IMGT CDR3s
where available. Its adapter explicitly removes the scoring caches' trailing
EOS label. The parent audit's stricter equal-length cache check has no usable
CDR3-pair annotations; use `lineage_review_v1/results.json` for this supplement.
Coverage is incomplete and nonrandom. CDR3 differences alone cannot establish
independent lineages: V/J, CDR3 length and experimental parent information are
still needed.

## Affinity curation

The audit checks all 423 SAbDab links against their raw source row, including
PDB, heavy/light chain, affinity, method, temperature and publication ID.
There are 142 links to 82 numeric PubMed IDs; 281 have no usable publication
ID. Construct/measurement deduplication collapses 106 extra crystal copies,
leaving 317 candidate links. Neither this deduplication nor a shared PMID
establishes biological independence or assay compatibility.

The raw SKEMPI audit finds 970 locally indexed antibody rows from 38 complexes.
Native annotations resolve chain roles for 926 rows; 298 have mutations only
on the heavy chain, before variable-domain and residue-number validation.
None pass the unchanged protected native whitelist. Exact sequence-prefix
checks verify 363 processed rows whose heavy/light columns are reversed.
The source processing is audited, not reused as a trusted training table.

`experiments/config/affinity_primary_ledger_v1.json` records six measurements from the
[2007 RNase-A VHH study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2148503/)
and its [2008 scaffold study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2442005/).
They represent one parent. Their published +/- values are retained without
assuming SD, SEM, confidence level, or replicate count. The 2007 clone-a
narrative reports 0.18 nM, whereas Table 1 and SAbDab give 0.15 nM; both
values remain explicitly flagged. Yeast display is kept separate from
soluble SPR, display constructs require reconciliation with crystal termini
and tags, and yeast induction temperature is not binding-assay temperature.

`bottleneck_audit.png` shows the deterministic grouping census and the
published affinity error bars. Census counts do not have sampling error bars.
The reported affinity bars are not model-performance confidence intervals.
Partial full-text downloads under `sources/` are not treated as validated
article archives or as evidence for these labels; the ledger identifies the
primary pages inspected through the web tool.

## Four-arm comparison and current stop condition

`four_arm_design.json` specifies scratch IgLM and MANGO, each with CE alone
and CE plus within-assay affinity ranking. Decoder size, initialization seed,
CE schedule, ranking pairs and optimization budget are shared. The planned
interaction contrast asks whether ranking improves MANGO more than IgLM.
The PLM/geometric representation stays fixed while measuring that objective
effect. Heavy/light output selection and antigen-only conditioning remain
the model contract.

The primary endpoint is macro within-assay, within-parent Spearman correlation
with negative log10 KD. Biological uncertainty must resample independent
held-out parents, paired across arms, and report optimization-seed SD
separately. A single parent or a collection of crystal copies cannot support
that confidence interval. The study's existing minimum counts are feasibility
guards, not a completed power calculation.

No new scientific fits were launched: zero affinity labels have all required
assay, construct and uncertainty provenance, and 227 protected native records
cannot satisfy disjoint minimums of 150 training and 100 heavy test examples.
The four-arm artifact is a design specification, not a claim that training
has run or that a complete new training executor was implemented here.

The next data work should prioritize measured VHH or fixed-light heavy-variant
panels with independent parent IDs, complete assay constructs, uncensored KD,
and explicit replicate/error definitions. Any proposed grouping revision must
state whether the claim concerns new antigen families, new antibody lineages,
or both, justify its lineage definition biologically, and undergo a fresh
audit while preserving original source-cluster and prior-holdout protection.
