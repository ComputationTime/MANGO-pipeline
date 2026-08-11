"""Single-GPU preflight and one-command completion target."""

GPU_PREFLIGHT = f"{ARTIFACT_ROOT}/gpu/preflight.json"
GPU_MANIFEST = f"{ARTIFACT_ROOT}/gpu/results_complete.json"
GPU_REPORT = f"{ARTIFACT_ROOT}/gpu/results_report.json"
GPU_NUMBERS = f"{ARTIFACT_ROOT}/gpu/results_report.csv"


def _gpu_weight_path(tag):
    method = embedder_method(tag)
    if method in {"esm2", "esm3", "esmif", "pyrosetta_pre"}:
        return pretrained_weight_marker(tag)
    if method == "proteinmpnn":
        spec = embedder_spec(tag)
        return (
            f"{WEIGHTS_ROOT}/proteinmpnn/{spec['model']}/"
            f"v_48_{int(spec['noise']):03d}.pt"
        )
    return None


def _gpu_report_specs():
    metric_names = ["iglm", "antiberty", "ablang2", "germline", "biophysical"]
    specs = []
    for tag in ACTIVE_EMBEDDERS:
        run = run_id(tag)
        specs.append(
            {
                "embedder": tag,
                "run_id": run,
                "weights": _gpu_weight_path(tag),
                "embedding": antigen_batch_marker(tag),
                "checkpoint": run_ckpt(tag),
                "training_curve": run_training_curve_csv(tag),
                "training_plot": run_training_curve_png(tag),
                "evaluation": eval_json(tag),
                "predictions": predictions_csv(tag),
                "analysis_metrics": {
                    name: analysis_metric_csv(tag, name) for name in metric_names
                },
                "logs": {
                    "train": f"{LOG_DIR}/train_{run}.log",
                    "evaluate": f"{LOG_DIR}/evaluate_{run}.log",
                    "predict": f"{LOG_DIR}/predict_{run}.log",
                },
            }
        )
    return specs


rule gpu_preflight:
    output:
        report=GPU_PREFLIGHT,
    params:
        min_memory_gb=config.get("execution", {}).get("min_gpu_memory_gb", 20),
    resources:
        gpu=1,
        mem_mb=4000,
    conda:
        "../envs/model.yaml"
    script:
        "../scripts/gpu_preflight.py"


rule gpu_results:
    input:
        preflight=GPU_PREFLIGHT,
        evals=[eval_json(tag) for tag in ACTIVE_EMBEDDERS],
        predictions=[predictions_csv(tag) for tag in ACTIVE_EMBEDDERS],
        training_curves=[run_training_curve_csv(tag) for tag in ACTIVE_EMBEDDERS],
        training_plots=[run_training_curve_png(tag) for tag in ACTIVE_EMBEDDERS],
        figures=[figure_path(name) for name in enabled_figures()],
        figure_data=[figure_data_path(name) for name in enabled_figures()],
    output:
        manifest=GPU_MANIFEST,
    params:
        embedders=ACTIVE_EMBEDDERS,
        artifact_root=ARTIFACT_ROOT,
    conda:
        "../envs/process.yaml"
    script:
        "../scripts/summarize_gpu_results.py"


rule gpu_embedder_result:
    """Core and modular analysis outputs for the currently active method(s)."""
    input:
        evals=[eval_json(tag) for tag in ACTIVE_EMBEDDERS],
        predictions=[predictions_csv(tag) for tag in ACTIVE_EMBEDDERS],
        metrics=[
            analysis_metric_csv(tag, metric)
            for tag in ACTIVE_EMBEDDERS
            for metric in ("iglm", "antiberty", "ablang2", "germline", "biophysical")
        ],


rule gpu_report:
    """Always-written partial report; deliberately has no scientific inputs."""
    output:
        report=GPU_REPORT,
        numbers=GPU_NUMBERS,
    params:
        specs=_gpu_report_specs(),
        preflight=GPU_PREFLIGHT,
        figures=[figure_path(name) for name in enabled_figures()],
        figure_data=[figure_data_path(name) for name in enabled_figures()],
    conda:
        "../envs/process.yaml"
    script:
        "../scripts/report_gpu_results.py"
