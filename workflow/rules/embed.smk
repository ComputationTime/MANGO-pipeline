# =============================================================================
# embed.smk -- precompute per-structure antigen embeddings.
# =============================================================================
# Output layout (the embedder contract):
#
#   embeddings/antigen/<tag>/embedder_config.json
#   embeddings/antigen/<tag>/{train,val,test}/<id>.pt
#
# One file per structure, so Snakemake fans out over exactly the ids the process
# checkpoint kept and a single new structure costs a single job.
#
# Every method rule writes the SAME output pattern, so each constrains its
# {embedder} wildcard to just its own active tags (method_constraint). That
# keeps the DAG unambiguous without ruleorder, and lets two tags share a method
# (e.g. two ESM2 sizes) for free.
#
# Heavy/conflicting dependencies stay isolated in per-rule conda envs -- ESM2
# and ESM3 both `import esm` from different packages, and PyRosetta needs its
# own channel, so they can never share an environment.

_MPNN_REPO_RAW = config["weights"]["proteinmpnn_repo_raw"]


def _spec(w):
    return embedder_spec(w.embedder)


def _records(w):
    """Targets live in their own records table; everything else in records.csv."""
    return records_for_split(w.split)


def _structure(w):
    """Structure path, from whichever records table backs this split."""
    if w.split == TARGET_SPLIT:
        return target_structure(w.instance)
    return structure_path(w.instance)


# --- one config JSON per embedder --------------------------------------------
rule embedder_config:
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/embedder_config.json",
    params:
        spec=lambda w: embedder_spec(w.embedder),
        seq_source=SEQ_SOURCE,
    conda:
        "../envs/process.yaml"
    script:
        "../scripts/write_embedder_config.py"


# --- external weights --------------------------------------------------------
rule fetch_ablang2_weights:
    output:
        marker=ABLANG2_WEIGHTS_MARKER,
    params:
        url=config["weights"]["ablang2_url"],
        weights_dir=ABLANG2_WEIGHTS_DIR,
    log:
        f"{LOG_DIR}/fetch_ablang2_weights.log",
    conda:
        "../envs/fetch.yaml"
    script:
        "../scripts/fetch_ablang2_weights.py"


rule fetch_proteinmpnn_weights:
    output:
        f"{WEIGHTS_ROOT}/proteinmpnn/{{model}}/v_48_{{noise3}}.pt",
    params:
        model=lambda w: w.model,
        noise=lambda w: int(w.noise3),
        repo_raw=_MPNN_REPO_RAW,
    log:
        f"{LOG_DIR}/fetch_proteinmpnn_{{model}}_{{noise3}}.log",
    conda:
        "../envs/embed_proteinmpnn.yaml"
    script:
        "../scripts/fetch_proteinmpnn_weights.py"


def _mpnn_weights(w):
    spec = embedder_spec(w.embedder)
    return (
        f"{WEIGHTS_ROOT}/proteinmpnn/{spec['model']}/v_48_{int(spec['noise']):03d}.pt"
    )


rule prefetch_esm2_weights:
    output:
        marker=f"{WEIGHTS_ROOT}/pretrained/{{embedder}}.json",
    params:
        method="esm2",
        model_name=lambda w: f"esm2_{embedder_spec(w.embedder)['size']}_UR50D",
    wildcard_constraints:
        embedder=method_constraint("esm2"),
    resources:
        weight_download=1, mem_mb=12000,
    conda:
        "../envs/embed_esm2.yaml"
    script:
        "../scripts/prefetch_embedder_weights.py"


rule prefetch_esmif_weights:
    output:
        marker=f"{WEIGHTS_ROOT}/pretrained/{{embedder}}.json",
    params:
        method="esmif",
        model_name=lambda w: embedder_spec(w.embedder)["model_name"],
    wildcard_constraints:
        embedder=method_constraint("esmif"),
    resources:
        weight_download=1, mem_mb=12000,
    conda:
        "../envs/embed_esmif.yaml"
    script:
        "../scripts/prefetch_embedder_weights.py"


rule prefetch_esm3_weights:
    output:
        marker=f"{WEIGHTS_ROOT}/pretrained/{{embedder}}.json",
    params:
        method="esm3",
        model_name=lambda w: embedder_spec(w.embedder)["model_name"],
    wildcard_constraints:
        embedder=method_constraint("esm3"),
    resources:
        weight_download=1, mem_mb=24000,
    conda:
        "../envs/embed_esm3.yaml"
    script:
        "../scripts/prefetch_embedder_weights.py"


rule verify_pyrosetta_assets:
    output:
        marker=f"{WEIGHTS_ROOT}/pretrained/{{embedder}}.json",
    params:
        method="pyrosetta_pre",
        model_name="pyrosetta-quarterly",
    wildcard_constraints:
        embedder=method_constraint("pyrosetta_pre"),
    resources:
        weight_download=1, mem_mb=8000,
    conda:
        "../envs/embed_pyrosetta.yaml"
    script:
        "../scripts/prefetch_embedder_weights.py"


def _active_weight_targets(wildcards):
    targets = [ABLANG2_WEIGHTS_MARKER]
    for tag in ACTIVE_EMBEDDERS:
        method = embedder_method(tag)
        if method in {"esm2", "esm3", "esmif"}:
            targets.append(pretrained_weight_marker(tag))
        elif method == "pyrosetta_pre":
            targets.append(pretrained_weight_marker(tag))
        elif method == "proteinmpnn":
            spec = embedder_spec(tag)
            targets.append(
                f"{WEIGHTS_ROOT}/proteinmpnn/{spec['model']}/"
                f"v_48_{int(spec['noise']):03d}.pt"
            )
    return list(dict.fromkeys(targets))


rule weights:
    """Download and verify all weights required by active embedders."""
    input:
        _active_weight_targets,


# --- antigen-free controls ---------------------------------------------------
rule embed_antigen_control:
    input:
        records=_records,
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source="none",
    wildcard_constraints:
        embedder=method_constraint("antigen_control"),
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/embed_antigen_control.py"


# --- naive -------------------------------------------------------------------
rule embed_antigen_one_hot:
    input:
        records=_records,
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source=SEQ_SOURCE,
    wildcard_constraints:
        embedder=method_constraint("one_hot"),
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/embed_antigen_one_hot.py"


# --- biophysical -------------------------------------------------------------
rule embed_antigen_biopython:
    input:
        records=_records,
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source=SEQ_SOURCE,
    wildcard_constraints:
        embedder=method_constraint("biopython"),
    conda:
        "../envs/embed_biopython.yaml"
    script:
        "../scripts/embed_antigen_biopython.py"


rule embed_antigen_pyrosetta_pre:
    input:
        records=_records,
        cif=_structure,
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
    wildcard_constraints:
        embedder=method_constraint("pyrosetta_pre"),
    conda:
        "../envs/embed_pyrosetta.yaml"
    script:
        "../scripts/embed_antigen_pyrosetta_pre.py"


# --- learned -----------------------------------------------------------------
rule embed_antigen_esm2:
    input:
        records=_records,
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source=SEQ_SOURCE,
    wildcard_constraints:
        embedder=method_constraint("esm2"),
    threads: 2
    resources:
        gpu=1,
        mem_mb=24000,
    conda:
        "../envs/embed_esm2.yaml"
    script:
        "../scripts/embed_antigen_esm2.py"


rule embed_antigen_esm3:
    input:
        records=_records,
        cif=_structure,
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source=SEQ_SOURCE,
    wildcard_constraints:
        embedder=method_constraint("esm3"),
    threads: 2
    resources:
        gpu=1,
        mem_mb=40000,
    conda:
        "../envs/embed_esm3.yaml"
    script:
        "../scripts/embed_antigen_esm3.py"


rule embed_antigen_esmif:
    input:
        records=_records,
        cif=_structure,
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
    wildcard_constraints:
        embedder=method_constraint("esmif"),
    threads: 2
    resources:
        gpu=1,
        mem_mb=16000,
    conda:
        "../envs/embed_esmif.yaml"
    script:
        "../scripts/embed_antigen_esmif.py"


rule embed_antigen_proteinmpnn:
    input:
        records=_records,
        cif=_structure,
        weights=_mpnn_weights,
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
    wildcard_constraints:
        embedder=method_constraint("proteinmpnn"),
    threads: 2
    resources:
        gpu=1,
        mem_mb=12000,
    conda:
        "../envs/embed_proteinmpnn.yaml"
    script:
        "../scripts/embed_antigen_proteinmpnn.py"


rule embed_antigen_afm:
    input:
        records=_records,
    output:
        f"{EMB_DIR}/antigen/{{embedder}}/{{split}}/{{instance}}.pt",
    params:
        spec=_spec,
        seq_source=SEQ_SOURCE,
    wildcard_constraints:
        embedder=method_constraint("afm"),
    conda:
        "../envs/embed_afm.yaml"
    script:
        "../scripts/embed_antigen_afm.py"


# --- persistent-model batches for the single-GPU profile --------------------
# The normal fine-grained rules above remain useful for incremental workstation
# work. On a full study, however, one process per record would reload large
# pretrained models thousands of times. `execution.batch_embeddings` switches
# dependencies to these completion markers; each rule loads its model once and
# writes the same per-record .pt contract beneath the usual split directories.
rule batch_antigen_control:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_control.py",
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="antigen_control",
        spec=_spec, seq_source="none", splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("antigen_control"),
    resources:
        embedder_slot=1, mem_mb=8000,
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_shuffled:
    input:
        records=RECORDS_CSV,
        source_eval=lambda w: ancient(embedder_spec(w.embedder)["source_eval_json"]),
        implementation="workflow/scripts/shuffle_antigen_embeddings.py",
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
        mapping=f"{EMB_DIR}/antigen/{{embedder}}/shuffle_map.csv",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        source_dir=lambda w: embedder_spec(w.embedder)["source_dir"],
        source_tag=lambda w: embedder_spec(w.embedder)["source_embedder"],
        seed=lambda w: int(embedder_spec(w.embedder).get("seed", 0)),
        splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("antigen_shuffled"),
    resources:
        embedder_slot=1, mem_mb=8000,
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/shuffle_antigen_embeddings.py"


rule batch_antigen_pooled:
    input:
        records=RECORDS_CSV,
        source_eval=lambda w: ancient(embedder_spec(w.embedder)["source_eval_json"]),
        implementation="workflow/scripts/pool_antigen_embeddings.py",
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        source_dir=lambda w: embedder_spec(w.embedder)["source_dir"],
        source_tag=lambda w: embedder_spec(w.embedder)["source_embedder"],
        splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("antigen_pooled"),
    resources:
        embedder_slot=1, mem_mb=8000,
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/pool_antigen_embeddings.py"


rule batch_antigen_one_hot:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_one_hot.py",
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="one_hot",
        spec=_spec, seq_source=SEQ_SOURCE, splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("one_hot"),
    resources:
        embedder_slot=1, mem_mb=8000,
    conda:
        "../envs/embed_onehot.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_biopython:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_biopython.py",
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="biopython",
        spec=_spec, seq_source=SEQ_SOURCE, splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("biopython"),
    resources:
        embedder_slot=1, mem_mb=8000,
    conda:
        "../envs/embed_biopython.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_pyrosetta_pre:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_pyrosetta_pre.py",
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="pyrosetta_pre",
        spec=_spec, seq_source="structure", splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("pyrosetta_pre"),
    resources:
        embedder_slot=1, mem_mb=16000,
    conda:
        "../envs/embed_pyrosetta.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_esm2:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_esm2.py",
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="esm2",
        spec=_spec, seq_source=SEQ_SOURCE, splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("esm2"),
    threads: 2
    resources:
        gpu=1, embedder_slot=1, mem_mb=24000,
    conda:
        "../envs/embed_esm2.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_esm3:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_esm3.py",
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="esm3",
        spec=_spec, seq_source=SEQ_SOURCE, splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("esm3"),
    threads: 2
    resources:
        gpu=1, embedder_slot=1, mem_mb=40000,
    conda:
        "../envs/embed_esm3.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_esmif:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_esmif.py",
        weights=lambda w: pretrained_weight_marker(w.embedder),
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="esmif",
        spec=_spec, seq_source="structure", splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("esmif"),
    threads: 2
    resources:
        gpu=1, embedder_slot=1, mem_mb=16000,
    conda:
        "../envs/embed_esmif.yaml"
    script:
        "../scripts/embed_batch.py"


rule batch_antigen_proteinmpnn:
    input:
        records=RECORDS_CSV,
        implementation="workflow/scripts/embed_antigen_proteinmpnn.py",
        weights=_mpnn_weights,
    output:
        marker=f"{EMB_DIR}/antigen/{{embedder}}/.batch_complete.json",
    params:
        output_dir=lambda w: antigen_emb_dir(w.embedder),
        kind="antigen", tag=lambda w: w.embedder, method="proteinmpnn",
        spec=_spec, seq_source="structure", splits=EMBED_SPLITS,
    wildcard_constraints:
        embedder=method_constraint("proteinmpnn"),
    threads: 2
    resources:
        gpu=1, embedder_slot=1, mem_mb=12000,
    conda:
        "../envs/embed_proteinmpnn.yaml"
    script:
        "../scripts/embed_batch.py"


# --- aggregation targets -----------------------------------------------------
def _all_antigen_embeddings(wildcards):
    out = []
    for tag in ACTIVE_EMBEDDERS:
        out.append(antigen_emb_config(tag))
        out += antigen_embedding_dependencies(tag)
    return out


rule embed_antigen:
    input:
        _all_antigen_embeddings,


rule embed:
    input:
        _all_antigen_embeddings,
