"""Plot all generated-chain biophysical properties as violin distributions."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


METRICS = [
    ("net_charge_pH7.4", "Net Charge pH7.4"),
    ("instability_index", "Instability Index"),
    ("isoelectric_point", "Isoelectric Point"),
    ("gravy_hydrophobicity", "Gravy Hydrophobicity"),
    ("aromaticity", "Aromaticity"),
    ("aliphatic_index", "Aliphatic Index"),
]


def plot_developability(paths, embedders, labels, dpi, out_figure, out_data):
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    aliases = {
        "gravy_hydrophobicity": "gravy",
        "net_charge_pH7.4": "charge_at_pH",
    }
    for canonical, legacy in aliases.items():
        if canonical not in data and legacy in data:
            data[canonical] = pd.to_numeric(data[legacy], errors="coerce")
    for column, _ in METRICS:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "metric_status" not in data:
        data["metric_status"] = "ok"
    valid = data.loc[data["metric_status"] == "ok"].copy()

    sns.set_theme(style="whitegrid")
    cmap = pc.colors(embedders)
    columns = 3
    rows = math.ceil(len(METRICS) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (metric, display_name) in zip(axes, METRICS):
        panel = valid.dropna(subset=[metric])
        if panel.empty:
            pc.empty_panel(
                ax,
                f"Distribution of {display_name}",
                "No valid values",
                xlabel="Embedder",
                ylabel=display_name,
            )
            continue
        sns.violinplot(
            data=panel,
            x="embedder",
            y=metric,
            order=embedders,
            hue="embedder",
            hue_order=embedders,
            palette=cmap,
            inner="quartile",
            cut=0,
            legend=False,
            ax=ax,
        )
        ax.set_xticks(
            range(len(embedders)),
            [labels.get(tag, tag) for tag in embedders],
            rotation=45,
            ha="right",
        )
        pc.style(
            ax,
            f"Distribution of {display_name}",
            xlabel="Embedder",
            ylabel=display_name,
        )

    output_columns = [
        "embedder",
        "run_id",
        "target_id",
        "design_index",
        "sequence",
        *[metric for metric, _ in METRICS],
        "metric_status",
    ]
    pc.save(
        fig,
        data[[column for column in output_columns if column in data]],
        out_figure,
        out_data,
        dpi,
    )


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_developability.py is intended to run through Snakemake")
    plot_developability(list(smk.input.metrics), list(smk.params.embedders),
                        dict(smk.params.labels), smk.params.dpi,
                        smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
