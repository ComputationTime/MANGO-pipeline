"""Bounded Boltz-2 and Chai-1 confidence predictions for handbook Figure 2."""

_SP = config["structure_prediction"]


rule predict_boltz_confidence:
    input:
        designs=lambda w: designs_csv(tag_for_run(w.run), w.instance),
        records=lambda w: records_for_split(GENERATION_SPLIT),
    output:
        raw=directory(f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/boltz2_raw"),
        scores=f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/boltz2_confidence.csv",
    params:
        n_designs=_SP["n_designs"],
        seed=_SP.get("selection_seed", config["experiment"]["seed"]),
        samples=_SP.get("diffusion_samples", 5),
        recycles=_SP.get("recycling_steps", 3),
        use_msa_server=_SP.get("use_msa_server", False),
        cache=f"{ARTIFACT_ROOT}/cache/boltz",
    log:
        f"{LOG_DIR}/structure_boltz2_{{run}}_{{instance}}.log",
    threads: 4
    resources:
        gpu=1,
        mem_mb=48000,
    conda:
        "../../envs/structure_boltz.yaml"
    script:
        "../../scripts/analysis/run_boltz_confidence.py"


rule predict_chai_confidence:
    input:
        designs=lambda w: designs_csv(tag_for_run(w.run), w.instance),
        records=lambda w: records_for_split(GENERATION_SPLIT),
    output:
        raw=directory(f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/chai_raw"),
        scores=f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/chai_confidence.csv",
    params:
        n_designs=_SP["n_designs"],
        seed=_SP.get("selection_seed", config["experiment"]["seed"]),
        samples=_SP.get("diffusion_samples", 5),
        recycles=_SP.get("recycling_steps", 3),
        use_msa_server=_SP.get("use_msa_server", False),
        cache=f"{ARTIFACT_ROOT}/cache/chai",
    log:
        f"{LOG_DIR}/structure_chai_{{run}}_{{instance}}.log",
    threads: 4
    resources:
        gpu=1,
        mem_mb=48000,
    conda:
        "../../envs/structure_chai.yaml"
    script:
        "../../scripts/analysis/run_chai_confidence.py"


rule structure_confidence_run:
    """Completion marker for every configured target/predictor in one run."""
    input:
        lambda w: structure_confidence_outputs(tag_for_run(w.run)),
    output:
        touch(f"{ANALYSIS_DIR}/{{run}}/structures/.confidence_complete"),


rule structure_confidence_scores:
    """Normalized Boltz/Chai tables for every active embedder, without plotting."""
    input:
        [struct_run_marker(tag) for tag in ANALYSIS_EMBEDDERS],


rule structure_confidence:
    input:
        scores=[struct_run_marker(tag) for tag in ANALYSIS_EMBEDDERS],
        figure=figure_path("fig2_structure_confidence"),
        data=figure_data_path("fig2_structure_confidence"),
