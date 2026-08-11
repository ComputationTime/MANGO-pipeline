# =============================================================================
# targets.smk -- Aim 2 held-out therapeutic targets (grant Table 1).
# =============================================================================
# Downloads the eight therapeutic complexes and describes them with a
# records-shaped table whose `split` is "target". That single trick means the
# eight antigen embedder rules in embed.smk serve these structures unchanged --
# their embeddings land at embeddings/antigen/<tag>/target/<PDB>.pt next to the
# train/val/test folders, and generation reads them like any other embedding.
#
# These structures are deliberately NOT in the training records: they are the
# held-out evaluation set the whole study builds toward.


rule fetch_target_structure:
    output:
        cif=f"{TARGETS_DIR}/structures/{{pdb}}.cif",
    params:
        base_url=config["generation"]["rcsb_url"],
    log:
        f"{LOG_DIR}/fetch_target_{{pdb}}.log",
    conda:
        "../envs/fetch.yaml"
    script:
        "../scripts/fetch_target_structure.py"


rule standardize_targets:
    input:
        cifs=[target_structure(p) for p in TARGET_PDBS],
    output:
        records=TARGETS_RECORDS,
    params:
        targets=GEN_TARGETS,
        structures_dir=f"{TARGETS_DIR}/structures",
        split=TARGET_SPLIT,
    log:
        f"{LOG_DIR}/standardize_targets.log",
    conda:
        "../envs/process.yaml"
    script:
        "../scripts/standardize_targets.py"


# --- aggregation targets -----------------------------------------------------
rule targets:
    input:
        TARGETS_RECORDS,


rule embed_targets:
    input:
        [target_emb_path(t, p) for t in ACTIVE_EMBEDDERS for p in TARGET_PDBS],
