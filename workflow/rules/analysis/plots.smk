"""Pure plotting rules for the currently active handbook figures."""

_PLOT_LABELS = {tag: embedder_label(tag) for tag in ANALYSIS_EMBEDDERS}
_DPI = config["analysis"]["dpi"]


rule plot_fig2_structure_confidence:
    input:
        scores=lambda w: [
            struct_scores_csv(tag, instance, method)
            for tag in ANALYSIS_EMBEDDERS
            for instance in sp_targets()
            for method in sp_methods()
        ],
    output:
        figure=figure_path("fig2_structure_confidence"),
        data=figure_data_path("fig2_structure_confidence"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        methods=config["structure_prediction"]["methods"],
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig2_structure_confidence.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_structure_confidence.py"


rule plot_fig1_nll:
    input:
        evals=[eval_json(tag) for tag in ANALYSIS_EMBEDDERS],
    output:
        figure=figure_path("fig1_nll"),
        data=figure_data_path("fig1_nll"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig1_nll.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_nll.py"


rule plot_fig3_ablikeness:
    input:
        iglm=[analysis_metric_csv(tag, "iglm") for tag in ANALYSIS_EMBEDDERS],
        antiberty=[analysis_metric_csv(tag, "antiberty") for tag in ANALYSIS_EMBEDDERS],
        ablang2=[analysis_metric_csv(tag, "ablang2") for tag in ANALYSIS_EMBEDDERS],
    output:
        figure=figure_path("fig3_ablikeness"),
        data=figure_data_path("fig3_ablikeness"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig3_ablikeness.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_ab_likeness.py"


rule plot_fig4_germline:
    input:
        metrics=[analysis_metric_csv(tag, "germline") for tag in ANALYSIS_EMBEDDERS],
    output:
        figure=figure_path("fig4_ld_germline"),
        data=figure_data_path("fig4_ld_germline"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig4_ld_germline.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_germline.py"


rule plot_fig5_developability:
    input:
        metrics=[analysis_metric_csv(tag, "biophysical") for tag in ANALYSIS_EMBEDDERS],
    output:
        figure=figure_path("fig5_developability"),
        data=figure_data_path("fig5_developability"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig5_developability.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_developability.py"
