"""ANARCI nearest-heavy-germline assignment and LD for handbook Figure 4."""

_GERMLINE = config["analysis"]["germline"]


rule analysis_score_germline:
    input:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/metrics/germline.csv",
    params:
        scheme=_GERMLINE["scheme"],
        allowed_species=_GERMLINE.get("allowed_species"),
    threads: 4
    log:
        f"{LOG_DIR}/analysis_germline_{{run}}.log",
    conda:
        "../../envs/analysis_anarci.yaml"
    script:
        "../../scripts/analysis/score_germline.py"
