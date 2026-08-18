"""Render a deliberately small Rosetta interface-energy plumbing figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(paths, figure_path, data_path, dpi=160):
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    data = data.loc[data["status"] == "ok"].copy()
    data["dG_separated"] = pd.to_numeric(data["dG_separated"], errors="coerce")
    data = data.dropna(subset=["dG_separated"])
    if data.empty:
        raise RuntimeError("no successful Rosetta interface scores to plot")

    predictors = list(dict.fromkeys(data["predictor"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for x, predictor in enumerate(predictors):
        values = data.loc[data["predictor"] == predictor, "dG_separated"]
        ax.scatter([x] * len(values), values, s=42, alpha=0.8, label=predictor)
    for _, paired in data.groupby(["embedder", "target_id", "design_index"]):
        if len(paired["predictor"].unique()) == len(predictors):
            paired = paired.set_index("predictor").reindex(predictors)
            ax.plot(range(len(predictors)), paired["dG_separated"],
                    color="0.72", linewidth=0.8, zorder=0)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xticks(range(len(predictors)), predictors)
    ax.set_ylabel("Rosetta dG_separated (REU)")
    ax.set_title("Fixed-backbone interface score (plumbing check)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    figure = Path(figure_path)
    table = Path(data_path)
    figure.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    data.to_csv(table, index=False)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_rosetta_interface.py must run through Snakemake")
    plot(list(smk.input.scores), smk.output.figure, smk.output.data, smk.params.dpi)


if __name__ == "__main__":
    main()
