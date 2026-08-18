"""Plot Figure 2: normalized Boltz-2 and Chai-1 confidence distributions."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc

METRICS = [
    ("confidence_score", "Ranking confidence"),
    ("ptm", "pTM"),
    ("iptm", "ipTM"),
]


def _best_samples(data):
    valid = data.loc[data["status"] == "ok"].copy()
    for column in [name for name, _ in METRICS]:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=["confidence_score"])
    if valid.empty:
        return valid
    keys = ["embedder", "target_id", "design_index", "predictor"]
    return valid.loc[valid.groupby(keys)["confidence_score"].idxmax()].copy()


def plot(paths, embedders, labels, methods, dpi, out_figure, out_data):
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    best = _best_samples(data) if not data.empty else data
    methods = [method for method in methods if method in set(best.get("predictor", []))]
    shown_methods = methods or [None]
    cmap = pc.colors(embedders)
    fig, axes = plt.subplots(
        len(shown_methods), len(METRICS),
        figsize=(4.0 * len(METRICS), 3.7 * len(shown_methods)),
        squeeze=False, sharey="col",
    )
    for row_idx, method in enumerate(shown_methods):
        for col_idx, (column, title) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            if method is None or best.empty:
                pc.empty_panel(ax, title, "No successful structure predictions")
                continue
            sub = best.loc[best["predictor"] == method]
            values = [sub.loc[sub["embedder"] == tag, column].dropna().to_numpy()
                      for tag in embedders]
            nonempty = [(idx + 1, vals) for idx, vals in enumerate(values) if len(vals)]
            if not nonempty:
                pc.empty_panel(ax, title, f"No {column} values for {method}")
                continue
            positions, series = zip(*nonempty)
            boxes = ax.boxplot(series, positions=positions, widths=0.65,
                               patch_artist=True, showfliers=False)
            for patch, position in zip(boxes["boxes"], positions):
                patch.set_facecolor(cmap[embedders[position - 1]])
                patch.set_alpha(0.72)
            ax.set_xticks(range(1, len(embedders) + 1))
            ax.set_xticklabels([labels.get(tag, tag) for tag in embedders],
                               rotation=30, ha="right")
            ax.set_ylim(-0.02, 1.02)
            pc.style(ax, f"{method}: {title}", ylabel=column if col_idx == 0 else "")
    fig.suptitle("Predicted complex confidence", x=0.01, ha="left")
    pc.add_legend(fig, embedders, labels, cmap)
    pc.save(fig, best, out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_structure_confidence.py must run through Snakemake")
    plot(list(smk.input.scores), list(smk.params.embedders),
         dict(smk.params.labels), list(smk.params.methods), smk.params.dpi,
         smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
