"""Plot LD to the nearest ANARCI heavy V/J germline as one violin plot.

Every embedder is a violin on a single shared axes (no per-embedder panels).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    data["ld_germline"] = pd.to_numeric(data["ld_germline"], errors="coerce")
    valid = data.loc[
        (data["germline_status"] == "ok") & data["embedder"].isin(embedders)
    ].dropna(subset=["ld_germline"])

    cmap = pc.colors(embedders)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(embedders) + 3.0), 6.5))
    title = "Distance from nearest heavy-chain germline by embedder"
    ylabel = "Levenshtein distance from germline"
    if valid.empty:
        pc.empty_panel(
            ax, title, "ANARCI assigned no generated sequence",
            xlabel="Embedder", ylabel=ylabel,
        )
    else:
        present = [tag for tag in embedders if (valid["embedder"] == tag).any()]
        sns.violinplot(
            data=valid,
            x="embedder",
            y="ld_germline",
            order=present,
            hue="embedder",
            hue_order=present,
            palette={tag: cmap[tag] for tag in present},
            inner="quartile",
            cut=0,
            legend=False,
            ax=ax,
        )
        # Keep every requested embedder on the axis, even one with no valid data.
        ax.set_xticks(
            range(len(present)),
            [labels.get(tag, tag) for tag in present],
            rotation=45,
            ha="right",
        )
        ax.set_ylim(bottom=0)
        pc.style(ax, title, xlabel="Embedder", ylabel=ylabel)
        missing = [tag for tag in embedders if tag not in present]
        if missing:
            ax.text(
                0.99, 0.98,
                "No valid distances: " + ", ".join(labels.get(t, t) for t in missing),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=pc.ANNOTATION_SIZE,
            )
    columns = ["embedder", "run_id", "target_id", "design_index", "sequence",
               "germline_species", "v_gene", "v_identity", "j_gene", "j_identity",
               "germline_reference_sequence", "ld_germline",
               "ld_germline_normalized", "germline_status"]
    pc.save(fig, data[[c for c in columns if c in data]], out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_germline.py is imported by run_analysis.py")
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
