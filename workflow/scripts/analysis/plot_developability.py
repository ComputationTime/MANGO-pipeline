"""Plot handbook Figure 5: GRAVY and charge-at-pH distributions (TAP deferred)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def plot_developability(paths, embedders, labels, dpi, out_figure, out_data):
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    valid = data.loc[data["metric_status"] == "ok"].copy()
    cmap = pc.colors(embedders)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    panels = [("gravy", "GRAVY hydrophobicity", "GRAVY"),
              ("charge_at_pH", "Charge at pH 7.4", "predicted net charge")]
    for ax, (column, title, xlabel) in zip(axes, panels):
        if valid.empty:
            pc.empty_panel(ax, title, "No valid generated sequences",
                           xlabel=xlabel, ylabel="density")
            continue
        edges = pc.shared_edges(valid, column, bins=30)
        for tag in embedders:
            values = valid.loc[valid.embedder == tag, column].dropna()
            ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                    color=cmap[tag])
        pc.style(ax, title, xlabel=xlabel, ylabel="density")
    fig.suptitle("Developability properties of generated heavy chains", x=0.01, ha="left")
    pc.add_legend(fig, embedders, labels, cmap)
    columns = ["embedder", "run_id", "target_id", "design_index", "sequence",
               "gravy", "charge_at_pH", "charge_ph", "metric_status"]
    pc.save(fig, data[[c for c in columns if c in data]], out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_developability.py is intended to run through Snakemake")
    plot_developability(list(smk.input.metrics), list(smk.params.embedders),
                        dict(smk.params.labels), smk.params.dpi,
                        smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
