"""Sequence-only developability metrics for handbook Figure 5 (TAP deferred)."""


rule analysis_biophysical:
    input:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/metrics/biophysical.csv",
    params:
        charge_ph=config["analysis"]["developability"]["charge_ph"],
    log:
        f"{LOG_DIR}/analysis_biophysical_{{run}}.log",
    conda:
        "../../envs/analysis_core.yaml"
    script:
        "../../scripts/analysis/score_biophysical.py"
