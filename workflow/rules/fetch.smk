# =============================================================================
# fetch.smk -- download + stage the SAbDab2 ML dataset from Zenodo.
# =============================================================================
# Produces a marker file once splits_final/ (structures + split tables) is
# present and the configured split CSV has been verified. Downstream rules
# depend on FETCH_MARKER, not on the thousands of individual .cif files.


rule fetch_sabdab2:
    output:
        marker=FETCH_MARKER,
    params:
        artifact_root=ARTIFACT_ROOT,
        version=DATASET_VERSION,
        url=config["dataset"]["zenodo_url"],
        split_file=config["dataset"]["split_file"],
    log:
        f"{ARTIFACT_ROOT}/logs/fetch_sabdab2.log",
    conda:
        "../envs/fetch.yaml"
    script:
        "../scripts/fetch_sabdab2.py"
