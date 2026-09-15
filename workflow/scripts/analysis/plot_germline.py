"""Plot handbook Figure 4: LD to the nearest ANARCI heavy V/J germline."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def plot_germline(paths, embedders, labels, dpi, out_figure, out_data):
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "ld_germline" not in data and "levenshtein_distance" in data:
        data["ld_germline"] = pd.to_numeric(
            data["levenshtein_distance"], errors="coerce"
        )
    if "germline_status" not in data:
        data["germline_status"] = data["ld_germline"].notna().map(
            {True: "ok", False: "error"}
        )
    valid = data.loc[data["germline_status"] == "ok"].copy()
    cmap = pc.colors(embedders)
    n_columns = min(4, len(embedders))
    n_rows = math.ceil(len(embedders) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    title = "Distance from nearest heavy-chain germline by embedder"
    if valid.empty:
        for ax, tag in zip(axes, embedders):
            pc.empty_panel(
                ax,
                labels.get(tag, tag),
                "ANARCI assigned no generated sequence",
                xlabel="Levenshtein distance",
                ylabel="Frequency",
            )
    else:
        edges = pc.shared_edges(valid, "ld_germline", bins=30)
        for index, (ax, tag) in enumerate(zip(axes, embedders)):
            values = pd.to_numeric(
                valid.loc[valid.embedder == tag, "ld_germline"], errors="coerce"
            ).dropna()
            if values.empty:
                pc.empty_panel(
                    ax,
                    labels.get(tag, tag),
                    "No valid distances",
                    xlabel="Levenshtein distance",
                    ylabel="Frequency" if index % n_columns == 0 else "",
                )
                continue
            ax.hist(
                values,
                bins=edges,
                alpha=0.88,
                color=cmap[tag],
                edgecolor="white",
                linewidth=0.8,
            )
            pc.style(
                ax,
                labels.get(tag, tag),
                xlabel="Levenshtein distance",
                ylabel="Frequency" if index % n_columns == 0 else "",
            )
            ax.tick_params(labelbottom=True)
    for ax in axes[len(embedders):]:
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.18, hspace=0.38)
    fig.suptitle(title, x=0.01, ha="left")
    columns = ["embedder", "run_id", "target_id", "design_index", "sequence",
               "germline_species", "v_gene", "v_identity", "j_gene", "j_identity",
               "germline_reference_sequence", "ld_germline",
               "ld_germline_normalized", "germline_status"]
    pc.save(fig, data[[c for c in columns if c in data]], out_figure, out_data, dpi)

def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_germline.py is intended to run through Snakemake")
    plot_germline(
        list(smk.input.metrics),
        list(smk.params.embedders),
        dict(smk.params.labels),
        smk.params.dpi,
        smk.output.figure,
        smk.output.data,
    )


if __name__ == "__main__":
    main()
