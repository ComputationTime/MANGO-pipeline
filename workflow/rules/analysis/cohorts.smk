"""Deterministic shared design cohorts for handbook Figures 3-5."""

rule analysis_cohort:
    input:
        designs=lambda w: analysis_design_inputs(w.run),
        records=RECORDS_CSV,
    output:
        cohort=f"{ANALYSIS_DIR}/{{run}}/cohort.csv",
    params:
        tag=lambda w: tag_for_run(w.run),
        n_per_target=config["analysis"]["cohort"].get("n_per_target"),
        seed=config["analysis"]["cohort"].get("seed", config["experiment"]["seed"]),
    log:
        f"{LOG_DIR}/analysis_cohort_{{run}}.log",
    conda:
        "../../envs/analysis_core.yaml"
    script:
        "../../scripts/analysis/select_cohort.py"
