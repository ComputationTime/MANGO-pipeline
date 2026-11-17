# Local binding-affinity analysis

Run the supplied HER2, PD1 and TIGIT CSVs against the latest completed run of
each of the eight MANGO representations and the scratch-trained IgLM:

```bash
MANGO_EXECUTION_CONFIG=config/local_12gb.yaml ./run_gpu.sh binding-affinity
```

The workflow freezes source and checkpoint hashes under
`artifacts/experiments/mango_binding_affinity/`. It reuses verified exact-sequence
ANARCI assignments, training annotations, antigen-only structures, embeddings,
and likelihoods from `openalphaseq_affinity_full_v1` when available. It does not
include that study's other observations. Missing embeddings and likelihoods use
the production implementations in their existing isolated environments, one
GPU job at a time. Rerun the same command to resume; use a new experiment root
when changing frozen inputs or checkpoints.

The primary score is heavy-variable-domain perplexity, including EOS. Only
finite measurements with sufficient assay replicates contribute, averaged per
normalized antibody–antigen pair. Every expressed domain must be absent from
the SAbDab train split for the strict primary analysis. In these source files,
all retained HER2 and PD1 scFvs have training-seen light chains, so their strict
correlations are undefined. The separately labeled **novel-heavy sensitivity**
requires an unseen heavy domain and permits a training-seen light domain.
Neither criterion establishes lineage independence.

Outputs in `correlations/` include:

- `findings.md`: strict and novel-heavy tables and interpretation limits.
- `pearson_by_model_target.csv`: strict Pearson r per target and pooled.
- `novel_heavy_pearson_by_model_target.csv`: the corresponding sensitivity table.
- `pearson_correlations.csv`: all cohort, quality, NLL and paired-domain results.
- `*_matched_pairs.csv.gz`: normalized pair scores, affinity aggregates and eligibility.
- `*_perplexity_affinity.png`: pooled and per-target nine-model plots.
- `local_summary.json`: source/checkpoint hashes, selected epochs and pair counts.

The affinity axis retains published `alphaseq_affinity` (log10 estimated Kd in
nM; lower means stronger binding). Positive r means higher perplexity associates
with weaker binding. Inspect per-target results alongside pooled results, which
can reflect library/antigen offsets. No naive row-independent p-values or
confidence intervals are attached to related library variants.

To draw the upstream FLAb blue-scatter/red-regression style using these same
heavy-domain scores, run the adapter in the existing affinity data environment:

```bash
python workflow/scripts/analysis/plot_mango_binding_flab.py
```

It uses `plot_flab.py` without changing the scoring definition. The 27 figures
and their verified correlations are written to `correlations/flab_style/`.
HER2 and PD1 use the explicitly labeled novel-heavy sensitivity cohort;
TIGIT also satisfies the strict domain-exclusion criterion. The adapter checks
each plotted coefficient and pair count against the completed affinity report.

Each model is attempted even if another fails. A failed run exits nonzero and
attempts a clearly marked partial report under `partial/`; inspect the model's
embedding/scoring logs under `logs/`. The strict nine-model summary is produced
only after all nine score databases are complete and verified.
