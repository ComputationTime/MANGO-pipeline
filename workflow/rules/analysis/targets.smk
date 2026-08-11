"""Explicit aggregate targets for metrics and handbook figures."""

rule analysis_metrics:
    input:
        [analysis_metric_csv(tag, metric)
         for tag in ANALYSIS_EMBEDDERS
         for metric in ("iglm", "antiberty", "ablang2", "germline", "biophysical")],


rule figures:
    input:
        [figure_path(name) for name in enabled_figures()],
        [figure_data_path(name) for name in enabled_figures()],


rule analysis:
    input:
        rules.figures.input,
