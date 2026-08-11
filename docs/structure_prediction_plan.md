# Structure-prediction integration plan

Structure prediction should be added after the one-hot training and inference
milestone is reproducible. It should be a separate workflow layer with three
stable boundaries:

```text
designs.csv + target structure
    → predictor-specific input bundle
    → predictor execution
    → normalized structures_and_scores.csv
```

## Recommended architecture

Keep input preparation and output normalization in ordinary Python scripts.
Run each predictor through its own Snakemake rule and container. Do not combine
AF3, Boltz, and Chai in one Conda environment.

Each backend should implement the same normalized output columns:

```text
target_id, design_index, sequence, predictor, status,
structure_path, ptm, iptm, pae
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
weights and large local databases. Each should still have a separate pinned
container and resource declaration.

## Suggested implementation order

1. Define and test the predictor input bundle and normalized output schema.
2. Implement Boltz execution and normalization in a pinned container.
3. Implement Chai against the same contract.
4. Add AF3 ingestion mode.
5. Add direct AF3 cluster execution once the image, weights, databases, and
   scheduler policy are known.

This isolates cluster-specific decisions from the core MANGO training DAG and
allows structure prediction to be retried or moved between systems without
retraining the model.
