# LEGACY prototype: not included by workflow/Snakefile. Active analysis rules
# are split across workflow/rules/analysis/*.smk.
# =============================================================================
# analysis.smk -- designs/evals -> comparison figures.
# =============================================================================
# Analysis is where the study's question gets answered, so every rule here fans
# IN over `analysis.embedders`: each figure compares antigen representations
# against each other rather than describing one in isolation.
#
# Figure families map 1:1 onto the handbook figure plan:
#
#   fig1_nll             NLL bars vs. Ag embedding        <- runs/*/eval.json
#   fig2_ptm             pTM x {AF3, Boltz2, Chai}        <- structure prediction
#   fig3_ablikeness      IgLM / AbLang2 conf., % charged  <- seq metrics
#   fig4_ld_germline     LD-from-germline distributions   <- seq metrics
#   fig5_developability  GRAVY, charge@pH, TAP            <- seq metrics
#
# DAG shape:
#   evaluate  (per tag) ------------------------------> fig1
#   generate  (per tag) -> seq_metrics (per tag) -----> fig3, fig4, fig5
#                       -> predict_structures -> complex_metrics -> fig2

_SEQ_CFG = config["analysis"]["seq_metrics"]
_SP = config["structure_prediction"]


rule analysis_seq_metrics:
    """Per-design sequence screening metrics for one run (grant Aim 2)."""
    input:
        designs=lambda w: [
            designs_csv(tag_for_run(w.run), i) for i in generation_instances()
        ],
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/seq_metrics.csv",
    params:
        tag=lambda w: tag_for_run(w.run),
        label=lambda w: embedder_label(tag_for_run(w.run)),
        seq_cfg=_SEQ_CFG,
    log:
        f"{LOG_DIR}/seq_metrics_{{run}}.log",
    conda:
        "../envs/analysis.yaml"
    script:
        "../scripts/analysis_seq_metrics.py"


rule analysis_predict_structures:
    """Fold designs in complex with the structure they were designed against."""
    input:
        designs=lambda w: designs_csv(tag_for_run(w.run), w.instance),
        target=lambda w: generation_structure(w.instance),
    output:
        scores=f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/{{sp_method}}_scores.csv",
    params:
        tag=lambda w: tag_for_run(w.run),
        method=lambda w: w.sp_method,
        n_designs=_SP["n_designs"],
        scores=_SP["scores"],
    log:
        f"{LOG_DIR}/predict_structures_{{run}}_{{instance}}_{{sp_method}}.log",
    conda:
        "../envs/structure_prediction.yaml"
    script:
        "../scripts/analysis_predict_structures.py"


rule analysis_complex_metrics:
    """Interface/developability metrics on the predicted complexes (Aim 3)."""
    input:
        scores=lambda w: [
            struct_scores_csv(tag_for_run(w.run), w.instance, m) for m in sp_methods()
        ],
    output:
        metrics=f"{ANALYSIS_DIR}/{{run}}/structures/{{instance}}/complex_metrics.csv",
    params:
        tag=lambda w: tag_for_run(w.run),
        metrics=config["analysis"]["complex_metrics"],
    log:
        f"{LOG_DIR}/complex_metrics_{{run}}_{{instance}}.log",
    conda:
        "../envs/analysis.yaml"
    script:
        "../scripts/analysis_complex_metrics.py"


# --- figures (fan-in over every compared embedder) ---------------------------
def _figure_inputs(figure):
    """Which artifacts a figure needs, across all compared embedders."""
    if figure == "fig1_nll":
        return {"evals": [eval_json(t) for t in ANALYSIS_EMBEDDERS]}
    if figure == "fig2_ptm":
        return {
            "scores": [
                struct_scores_csv(t, i, m)
                for t in ANALYSIS_EMBEDDERS
                for i in sp_targets()
                for m in sp_methods()
            ]
        }
    if figure in ("fig3_ablikeness", "fig4_ld_germline", "fig5_developability"):
        return {"metrics": [seq_metrics_csv(t) for t in ANALYSIS_EMBEDDERS]}
    raise ValueError(
        f"unknown figure {figure!r}; register its inputs in _figure_inputs()"
    )


rule analysis_plot:
    input:
        unpack(lambda w: _figure_inputs(w.figure)),
    output:
        figure=f"{FIGURES_DIR}/{{figure}}.{config['analysis']['format']}",
        data=f"{FIGURES_DIR}/{{figure}}_data.csv",
    params:
        figure=lambda w: w.figure,
        embedders=ANALYSIS_EMBEDDERS,
        labels=lambda w: {t: embedder_label(t) for t in ANALYSIS_EMBEDDERS},
        classes=lambda w: {t: embedder_spec(t).get("class", "") for t in ANALYSIS_EMBEDDERS},
        plot_cfg=lambda w: config["analysis"]["plots"][w.figure],
        dpi=config["analysis"]["dpi"],
    log:
        f"{LOG_DIR}/plot_{{figure}}.log",
    conda:
        "../envs/analysis.yaml"
    script:
        "../scripts/analysis_plots.py"


wildcard_constraints:
    figure="|".join(re.escape(f) for f in config["analysis"]["plots"]),


# --- aggregation targets -----------------------------------------------------
rule seq_metrics:
    input:
        [seq_metrics_csv(t) for t in ANALYSIS_EMBEDDERS],


# Input functions, not lists: sp_targets() follows the generation set, which
# only exists once the process checkpoint has run.
rule structures:
    input:
        lambda w: [
            struct_scores_csv(t, i, m)
            for t in ANALYSIS_EMBEDDERS
            for i in sp_targets()
            for m in sp_methods()
        ],


rule figures:
    input:
        [figure_path(f) for f in enabled_figures()],


rule analysis:
    input:
        [figure_path(f) for f in enabled_figures()],
