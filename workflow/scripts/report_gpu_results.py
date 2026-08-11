"""Always report successful embedder numbers and identify incomplete methods."""

import csv
import json
from collections import Counter
from pathlib import Path


CSV_COLUMNS = [
    "embedder", "status", "failure_stage", "run_id", "split",
    "n_examples", "n_tokens", "nll", "perplexity", "prediction_rows",
    "successful_predictions", "training_curve", "training_plot",
    "evaluation", "predictions",
]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _prediction_counts(path):
    target = Path(path)
    if not target.is_file():
        return None, None
    try:
        with target.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        statuses = Counter(row.get("status", "") for row in rows)
        return {
            "rows": len(rows),
            "statuses": dict(statuses),
            "ok": statuses.get("ok", 0),
        }, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _failure_stage(spec, evaluation, predictions):
    checks = [
        ("weights", spec.get("weights")),
        ("embedding", spec.get("embedding")),
        ("training", spec.get("checkpoint")),
        ("training_curve", spec.get("training_curve")),
        ("training_plot", spec.get("training_plot")),
        ("evaluation", spec.get("evaluation")),
        ("prediction", spec.get("predictions")),
    ]
    for stage, path in checks:
        if path and not Path(path).is_file():
            return stage
    if evaluation is None:
        return "evaluation_invalid"
    if predictions is None:
        return "prediction_invalid"
    return None


def report(specs, preflight_path, figures, figure_data, output_json, output_csv):
    runs = {}
    csv_rows = []
    successful = []
    failed = []
    fully_complete = []
    analysis_incomplete = []
    for spec in specs:
        tag = spec["embedder"]
        evaluation = None
        evaluation_error = None
        if Path(spec["evaluation"]).is_file():
            evaluation, evaluation_error = _read_json(spec["evaluation"])
            if evaluation is not None and not isinstance(evaluation, dict):
                evaluation_error = "evaluation JSON must contain an object"
                evaluation = None
        predictions, prediction_error = _prediction_counts(spec["predictions"])
        core_failure_stage = _failure_stage(spec, evaluation, predictions)
        expected_metrics = spec.get("analysis_metrics", {})
        metrics = {
            name: path for name, path in spec.get("analysis_metrics", {}).items()
            if Path(path).is_file()
        }
        missing_metrics = sorted(set(expected_metrics) - set(metrics))
        if core_failure_stage is not None:
            status = "failed_or_incomplete"
            failure_stage = core_failure_stage
            failed.append(tag)
        elif missing_metrics:
            status = "core_complete_analysis_incomplete"
            failure_stage = "analysis"
            successful.append(tag)
            analysis_incomplete.append(tag)
        else:
            status = "complete"
            failure_stage = None
            successful.append(tag)
            fully_complete.append(tag)
        run = {
            "status": status,
            "failure_stage": failure_stage,
            "run_id": spec["run_id"],
            "weights_ready": not spec.get("weights") or Path(spec["weights"]).is_file(),
            "embedding_ready": Path(spec["embedding"]).is_file(),
            "checkpoint_ready": Path(spec["checkpoint"]).is_file(),
            "training_curve": (
                spec.get("training_curve")
                if spec.get("training_curve") and Path(spec["training_curve"]).is_file()
                else None
            ),
            "training_plot": (
                spec.get("training_plot")
                if spec.get("training_plot") and Path(spec["training_plot"]).is_file()
                else None
            ),
            "evaluation": spec["evaluation"] if evaluation is not None else None,
            "evaluation_error": evaluation_error,
            "splits": evaluation.get("splits", {}) if evaluation else {},
            "predictions": spec["predictions"] if predictions is not None else None,
            "prediction_error": prediction_error,
            "prediction_counts": predictions,
            "analysis_metrics": metrics,
            "missing_analysis_metrics": missing_metrics,
            "expected_paths": spec,
            "logs": spec.get("logs", {}),
        }
        runs[tag] = run

        splits = run["splits"] or {"": {}}
        for split, values in splits.items():
            csv_rows.append(
                {
                    "embedder": tag,
                    "status": status,
                    "failure_stage": failure_stage or "",
                    "run_id": spec["run_id"],
                    "split": split,
                    "n_examples": values.get("n_examples", ""),
                    "n_tokens": values.get("n_tokens", ""),
                    "nll": values.get("nll", ""),
                    "perplexity": values.get("perplexity", ""),
                    "prediction_rows": predictions["rows"] if predictions else "",
                    "successful_predictions": predictions["ok"] if predictions else "",
                    "training_curve": run["training_curve"] or "",
                    "training_plot": run["training_plot"] or "",
                    "evaluation": run["evaluation"] or "",
                    "predictions": run["predictions"] or "",
                }
            )

    preflight, preflight_error = (None, None)
    if Path(preflight_path).is_file():
        preflight, preflight_error = _read_json(preflight_path)
    discovered_figures = [path for path in figures if Path(path).is_file()]
    discovered_figure_data = [path for path in figure_data if Path(path).is_file()]
    is_fully_complete = not failed and not analysis_incomplete
    result = {
        "status": "complete" if is_fully_complete else "partial",
        "task": "antigen+light->heavy",
        "summary": {
            "expected": len(specs),
            "complete": len(fully_complete),
            "core_successful": len(successful),
            "failed_or_incomplete": len(failed),
            "successful_embedders": successful,
            "fully_complete_embedders": fully_complete,
            "analysis_incomplete_embedders": analysis_incomplete,
            "failed_or_incomplete_embedders": failed,
        },
        "preflight": preflight,
        "preflight_error": preflight_error,
        "runs": runs,
        # Global plots are trustworthy only after strict all-embedder completion.
        "figures": discovered_figures if is_fully_complete else [],
        "figure_data": discovered_figure_data if is_fully_complete else [],
        "possibly_stale_figures": discovered_figures if not is_fully_complete else [],
        "possibly_stale_figure_data": (
            discovered_figure_data if not is_fully_complete else []
        ),
        "note": (
            "Per-embedder NLL/perplexity and prediction counts remain reportable "
            "when comparison figures cannot be rebuilt because another method failed."
        ),
    }
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"GPU report: {len(successful)}/{len(specs)} core runs succeeded; "
        f"fully_complete={fully_complete}; analysis_incomplete={analysis_incomplete}; "
        f"failed={failed}",
        flush=True,
    )
    print(f"numbers -> {output_csv}\nreport -> {output_json}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("report_gpu_results.py is intended for Snakemake")
    report(
        specs=list(smk.params.specs),
        preflight_path=smk.params.preflight,
        figures=list(smk.params.figures),
        figure_data=list(smk.params.figure_data),
        output_json=smk.output.report,
        output_csv=smk.output.numbers,
    )


if __name__ == "__main__":
    main()
