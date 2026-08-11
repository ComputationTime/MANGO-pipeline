# =============================================================================
# process.smk -- standardized.csv -> filtered, resolved, split-assigned records.
# =============================================================================
# Applies the configured filters, guarantees resolved sequences for heavy /
# light / antigen, carves a validation set out of train, and writes the one
# table that every downstream module reads.
#
# This is a *checkpoint*: the set of kept ids is not known until it runs, and
# the embed rules fan out over exactly those ids via checkpoints.get(...).


checkpoint process_records:
    input:
        standardized=STANDARDIZED_CSV,
    output:
        records=RECORDS_CSV,
    params:
        processing=config["processing"],
    log:
        f"{LOG_DIR}/process_records.log",
    conda:
        "../envs/process.yaml"
    script:
        "../scripts/process_records.py"
