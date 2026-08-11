# =============================================================================
# standardize.smk -- raw dataset -> one dataset-agnostic table.
# =============================================================================
# This is the seam that lets a new dataset drop in without touching anything
# downstream. Everything after this point reads `standardized.csv` and knows
# nothing about SAbDab2's column names or directory layout.
#
# Contract (see scripts/standardize_common.py):
#   id, pdb_path, antigen_chains, expected_heavy_seq, expected_light_seq,
#   expected_ag_seq, split
# Every column except id/pdb_path may be empty. Adapters may carry additional
# dataset-specific columns through; `process` uses them when present and falls
# back to parsing the structure when they are not.
#
# Add a dataset by writing scripts/standardize_<dataset.name>.py with the same
# entrypoint shape and registering it below.

_STANDARDIZERS = {
    "sabdab2": "../scripts/standardize_sabdab2.py",
}

if DATASET_NAME not in _STANDARDIZERS:
    raise ValueError(
        f"No standardizer for dataset '{DATASET_NAME}'. "
        f"Known: {sorted(_STANDARDIZERS)}. Add workflow/scripts/standardize_"
        f"{DATASET_NAME}.py and register it in workflow/rules/standardize.smk."
    )


rule standardize:
    # The dependency edge is the fetch marker, not the split CSV itself: fetch
    # unpacks thousands of files and validates the split table before writing
    # the marker, so depending on the marker keeps the DAG small and avoids
    # requiring a rule that "produces" an archive member.
    input:
        marker=FETCH_MARKER,
    output:
        standardized=STANDARDIZED_CSV,
    params:
        split_csv=SPLIT_CSV,
        splits_dir=SPLITS_DIR,
        split_column=config["dataset"]["split_column"],
        dataset_name=DATASET_NAME,
    log:
        f"{LOG_DIR}/standardize_{DATASET_NAME}.log",
    conda:
        "../envs/process.yaml"
    script:
        _STANDARDIZERS[DATASET_NAME]
