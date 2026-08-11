"""Independent antibody-language-model scorers for handbook Figure 3."""

_ABLIKE = config["analysis"]["ab_likeness"]


rule analysis_score_iglm:
    input:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/metrics/iglm.csv",
    params:
        chain_token=_ABLIKE["iglm"]["chain_token"],
        species_token=_ABLIKE["iglm"]["species_token"],
    resources:
        gpu=1,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/analysis_iglm_{{run}}.log",
    conda:
        "../../envs/analysis_iglm.yaml"
    script:
        "../../scripts/analysis/score_iglm.py"


rule analysis_score_antiberty:
    input:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/metrics/antiberty.csv",
    params:
        batch_size=_ABLIKE["antiberty"]["batch_size"],
    resources:
        gpu=1,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/analysis_antiberty_{{run}}.log",
    conda:
        "../../envs/analysis_antiberty.yaml"
    script:
        "../../scripts/analysis/score_antiberty.py"


rule analysis_score_ablang2:
    input:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
        weights=ABLANG2_WEIGHTS_MARKER,
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/metrics/ablang2.csv",
    params:
        model_dir=lambda w, input: str(PurePosixPath(input.weights).parent),
        mode=_ABLIKE["ablang2"]["mode"],
    resources:
        gpu=1,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/analysis_ablang2_{{run}}.log",
    conda:
        "../../envs/analysis_ablang2.yaml"
    script:
        "../../scripts/analysis/score_ablang2.py"
