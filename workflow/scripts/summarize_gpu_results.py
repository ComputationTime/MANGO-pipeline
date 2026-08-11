"""Write a machine-readable completion manifest for a multi-embedder run."""

import json
from pathlib import Path


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("summarize_gpu_results.py is intended for Snakemake")
    artifact_root = Path(smk.params.artifact_root)
    embedders = list(smk.params.embedders)
    runs = {}
    for tag, eval_path, prediction_path in zip(
        embedders, smk.input.evals, smk.input.predictions
    ):
        evaluation = json.loads(Path(eval_path).read_text())
        runs[tag] = {
            "run_id": evaluation.get("run_id"),
            "evaluation": str(eval_path),
            "predictions": str(prediction_path),
            "splits": evaluation.get("splits", {}),
        }
    result = {
        "status": "complete",
        "task": "antigen+light->heavy",
        "embedders": embedders,
        "preflight": str(smk.input.preflight),
        "runs": runs,
        "training_curves": [str(path) for path in smk.input.training_curves],
        "training_plots": [str(path) for path in smk.input.training_plots],
        "figures": [str(path) for path in smk.input.figures],
        "figure_data": [str(path) for path in smk.input.figure_data],
        "artifact_root": str(artifact_root),
    }
    Path(smk.output.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(smk.output.manifest).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"GPU result manifest -> {smk.output.manifest}", flush=True)


if __name__ == "__main__":
    main()
