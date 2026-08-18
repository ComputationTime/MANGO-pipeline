# Structure-prediction integration plan

Structure prediction is a bounded default layer after training and inference. Boltz-2 and
Chai-1 confidence execution and normalization are implemented; AF3 remains a
cluster-side ingestion target. It should be a separate workflow layer with three
stable boundaries:

```text
designs.csv + held-out target light/antigen sequences
    → predictor-specific input bundle
    → predictor execution
    → normalized <predictor>_confidence.csv
```

## Recommended architecture

Keep input preparation and output normalization in ordinary Python scripts.
Run each predictor through its own Snakemake rule and container. Do not combine
AF3, Boltz, and Chai in one Conda environment.

Each backend should implement the same normalized output columns:

```text
embedder, run_id, target_id, design_index, sequence, predictor, sample_index,
confidence_score, ptm, iptm, complex_plddt, mean_pae,
has_inter_chain_clashes, structure_path, status
```

This lets downstream code consume predictor results without knowing their
native directory layouts.

## AF3

AF3 should run on the cluster through an Apptainer image and a Snakemake
executor profile. Model weights and sequence databases are installation-level
resources and must not be downloaded by individual jobs. Their locations
should come from the cluster profile or environment variables, not the study
configuration committed to Git.

The eventual rule should declare abstract resources such as `gpu`, `mem_mb`,
and `runtime`. A Slurm profile can translate those resources into the local
partitions and account settings. Keep a second ingestion-only mode so results
produced by a centrally managed AF3 service can enter through the same
normalization step.

## Boltz and Chai

Boltz and Chai can use the same prepare/run/normalize contract. They are better
first implementations because their distribution is less tied to gated model
weights and large local databases. Each still has a separate pinned Conda environment and resource declaration.

## Implemented local confidence contract

The GPU launcher selects the same deterministic design ranks for each model,
folds three designs from each of ten held-out antigen clusters in study mode
(one target/design in smoke), combines heavy + cognate light + every antigen
chain, and writes one normalized row per diffusion sample:

```text
embedder, run_id, target_id, design_index, sequence, predictor, sample_index,
confidence_score, ptm, iptm, complex_plddt, mean_pae,
has_inter_chain_clashes, structure_path, status
```

Raw predictor files remain under the corresponding `*_raw/` directory. The plot
uses each design's highest-ranking diffusion sample. Single-sequence mode is the
default for reproducibility; the external MSA service is explicit opt-in. GPU
rules request `gpu=1` and therefore remain serialized on the supported profile.

## Suggested remaining implementation order

1. Validate Boltz-2 and Chai-1 environments and confidence outputs on the smoke
   cohort, then run the configured study subset.
2. Add AF3 ingestion mode against the normalized confidence contract.
3. Add direct AF3 cluster execution once the image, weights, databases, and
   scheduler policy are known.

This isolates cluster-specific decisions from the core MANGO training DAG and
allows structure prediction to be retried or moved between systems without
retraining the model.
