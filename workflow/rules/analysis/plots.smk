"""Pure plotting rules for the currently active handbook figures."""

from pathlib import Path

_PLOT_LABELS = {tag: embedder_label(tag) for tag in ANALYSIS_EMBEDDERS}
_DPI = config["analysis"]["dpi"]
_ESM2_ABLATION_CFG = config["analysis"].get("esm2_ablation", {})
_ESM2_ABLATION_TAGS = list(_ESM2_ABLATION_CFG.get("embedders", []))
_ANTIGEN_CONTROL_CFG = config["analysis"].get("antigen_controls", {})
_ANTIGEN_CONTROL_TAGS = list(_ANTIGEN_CONTROL_CFG.get("embedders", []))


def _esm2_ablation_evals(wildcards):
    if not _ESM2_ABLATION_TAGS:
        raise ValueError(
            "the esm2_ablation target requires config/esm2_ablation.yaml"
        )
    unknown = [tag for tag in _ESM2_ABLATION_TAGS if tag not in EMBEDDERS]
    if unknown:
        raise ValueError(f"unregistered ESM2 ablation tag(s): {unknown}")
    inactive = [tag for tag in _ESM2_ABLATION_TAGS if tag not in ACTIVE_EMBEDDERS]
    if inactive:
        raise ValueError(
            f"ESM2 ablation tag(s) are not active: {inactive}; "
            "load config/esm2_ablation.yaml"
        )
    wrong_method = [
        tag for tag in _ESM2_ABLATION_TAGS if embedder_method(tag) != "esm2"
    ]
    if wrong_method:
        raise ValueError(f"non-ESM2 tag(s) in the ESM2 ablation: {wrong_method}")
    reuse_tag = _ESM2_ABLATION_CFG.get("reuse_tag")
    reuse_eval = _ESM2_ABLATION_CFG.get("reuse_eval_json")
    paths = []
    for tag in _ESM2_ABLATION_TAGS:
        if tag == reuse_tag:
            if not reuse_eval or not Path(reuse_eval).is_file():
                raise ValueError(
                    f"the completed {reuse_tag!r} evaluation must exist at "
                    f"{reuse_eval!r}; the ablation will not retrain it"
                )
            paths.append(ancient(reuse_eval))
        else:
            paths.append(eval_json(tag))
    return paths


def _antigen_control_evals(wildcards):
    if not _ANTIGEN_CONTROL_TAGS:
        raise ValueError(
            "the antigen_controls target requires config/antigen_controls.yaml"
        )
    unknown = [tag for tag in _ANTIGEN_CONTROL_TAGS if tag not in EMBEDDERS]
    if unknown:
        raise ValueError(f"unregistered antigen-control plot tag(s): {unknown}")
    reuse_tag = _ANTIGEN_CONTROL_CFG.get("reuse_tag")
    reuse_eval = _ANTIGEN_CONTROL_CFG.get("reuse_eval_json")
    inactive = [
        tag for tag in _ANTIGEN_CONTROL_TAGS
        if tag != reuse_tag and tag not in ACTIVE_EMBEDDERS
    ]
    if inactive:
        raise ValueError(
            f"antigen control tag(s) are not active: {inactive}; "
            "load config/antigen_controls.yaml"
        )
    paths = []
    for tag in _ANTIGEN_CONTROL_TAGS:
        if tag == reuse_tag:
            if not reuse_eval or not Path(reuse_eval).is_file():
                raise ValueError(
                    f"the completed {reuse_tag!r} evaluation must exist at "
                    f"{reuse_eval!r}; the control comparison will not retrain it"
                )
            paths.append(ancient(reuse_eval))
        else:
            paths.append(eval_json(tag))
    return paths


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


rule plot_esm2_size_ablation:
    """Compare MANGO likelihood across frozen ESM2 checkpoint sizes."""
    input:
        evals=_esm2_ablation_evals,
    output:
        figure=figure_path("fig7_esm2_size_ablation"),
        data=figure_data_path("fig7_esm2_size_ablation"),
    params:
        embedders=_ESM2_ABLATION_TAGS,
        specs={tag: embedder_spec(tag) for tag in _ESM2_ABLATION_TAGS},
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig7_esm2_size_ablation.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_esm2_ablation.py"


rule esm2_ablation:
    """Train/evaluate configured ESM2 sizes and render their dedicated plot."""
    input:
        rules.plot_esm2_size_ablation.output.figure,
        rules.plot_esm2_size_ablation.output.data,


rule plot_antigen_controls:
    """Compare no-antigen and shuffled-antigen controls with matched ESM2."""
    input:
        evals=_antigen_control_evals,
    output:
        figure=figure_path("fig8_antigen_controls"),
        data=figure_data_path("fig8_antigen_controls"),
    params:
        embedders=_ANTIGEN_CONTROL_TAGS,
        labels={tag: embedder_label(tag) for tag in _ANTIGEN_CONTROL_TAGS},
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig8_antigen_controls.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_antigen_controls.py"


rule antigen_controls:
    """Train/evaluate antigen controls and render their dedicated plot."""
    input:
        rules.plot_antigen_controls.output.figure,
        rules.plot_antigen_controls.output.data,


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


rule plot_fig6_gene_families:
    input:
        metrics=[analysis_metric_csv(tag, "germline") for tag in ANALYSIS_EMBEDDERS],
    output:
        figure=figure_path("fig6_gene_families"),
        data=figure_data_path("fig6_gene_families"),
    params:
        embedders=ANALYSIS_EMBEDDERS,
        labels=_PLOT_LABELS,
        dpi=_DPI,
    log:
        f"{LOG_DIR}/plot_fig6_gene_families.log",
    conda:
        "../../envs/analysis_plot.yaml"
    script:
        "../../scripts/analysis/plot_genes.py"


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
