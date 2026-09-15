"""Shared visual grammar for handbook plots; contains no metric computation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Seaborn's colorblind palette, kept as explicit hex values so every plot uses
# the same embedder colors without requiring seaborn at runtime.
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#D55E00",
           "#CC78BC", "#CA9161", "#FBAFE4", "#949494"]
EMBEDDER_COLORS = {
    "one_hot": PALETTE[0],
    "biopython": PALETTE[1],
    "pyrosetta_pre": PALETTE[2],
    "esm2": PALETTE[3],
    "esm3": PALETTE[4],
    "esmif": PALETTE[5],
    "proteinmpnn": PALETTE[6],
    "afm": PALETTE[7],
}
FONT_FAMILY = "DejaVu Sans"
TEXT = "#262626"
TITLE_SIZE = 14
SUPTITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9
INK = TEXT
MUTED = TEXT
GRID = "#D9D9D9"
PANEL = "#F7F7F7"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlepad": 15,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    "figure.titlesize": SUPTITLE_SIZE,
    "font.family": FONT_FAMILY,
})


def colors(embedders):
    unknown = [tag for tag in embedders if tag not in EMBEDDER_COLORS]
    if unknown:
        raise ValueError(f"no fixed plot color is registered for {unknown}")
    return {tag: EMBEDDER_COLORS[tag] for tag in embedders}


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=15)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)


def label_bars(ax, bars, values, fmt="{:.3f}"):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.annotate(fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=ANNOTATION_SIZE, color=TEXT)


def add_legend(fig, embedders, labels, color_map):
    handles = [
        matplotlib.patches.Patch(color=color_map[tag], label=labels.get(tag, tag))
        for tag in embedders
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.01))


def shared_edges(frame, column, bins=30):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    low, high = float(values.min()), float(values.max())
    if low == high:
        width = max(abs(low) * 0.05, 0.5)
        low, high = low - width, high + width
    return np.linspace(low, high, int(bins) + 1)


def empty_panel(ax, title, message, xlabel="", ylabel=""):
    """Render an explicit empty-state panel while preserving the plot artifact."""
    style(ax, title, xlabel=xlabel, ylabel=ylabel)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=TEXT, fontsize=ANNOTATION_SIZE, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def normalize_typography(fig):
    """Apply one font and text color with consistent role-based sizes."""
    for ax in fig.axes:
        ax.title.set(
            fontfamily=FONT_FAMILY,
            fontsize=TITLE_SIZE,
            fontweight="normal",
            color=TEXT,
            horizontalalignment="center",
            x=0.5,
        )
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set(
                fontfamily=FONT_FAMILY,
                fontsize=LABEL_SIZE,
                fontweight="normal",
                color=TEXT,
            )
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set(
                fontfamily=FONT_FAMILY,
                fontsize=TICK_SIZE,
                fontweight="normal",
                color=TEXT,
            )
        for annotation in ax.texts:
            annotation.set(
                fontfamily=FONT_FAMILY,
                fontweight="normal",
                color=TEXT,
            )
        legend = ax.get_legend()
        if legend is not None:
            legend.get_title().set(
                fontfamily=FONT_FAMILY,
                fontsize=LEGEND_SIZE,
                fontweight="normal",
                color=TEXT,
            )
            for label in legend.get_texts():
                label.set(
                    fontfamily=FONT_FAMILY,
                    fontsize=LEGEND_SIZE,
                    fontweight="normal",
                    color=TEXT,
                )
    for legend in fig.legends:
        legend.get_title().set(
            fontfamily=FONT_FAMILY,
            fontsize=LEGEND_SIZE,
            fontweight="normal",
            color=TEXT,
        )
        for label in legend.get_texts():
            label.set(
                fontfamily=FONT_FAMILY,
                fontsize=LEGEND_SIZE,
                fontweight="normal",
                color=TEXT,
            )
    if fig._suptitle is not None:
        fig._suptitle.set(
            fontfamily=FONT_FAMILY,
            fontsize=SUPTITLE_SIZE,
            fontweight="normal",
            color=TEXT,
            horizontalalignment="center",
            x=0.5,
        )


def save(fig, data, out_figure, out_data, dpi):
    normalize_typography(fig)
    Path(out_data).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)
    Path(out_figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(
        out_figure,
        dpi=max(300, int(dpi)),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {out_figure} and {out_data}", flush=True)
