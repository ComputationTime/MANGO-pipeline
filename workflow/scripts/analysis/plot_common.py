"""Shared visual grammar for handbook plots; contains no metric computation."""

from pathlib import Path
import re

import matplotlib
from matplotlib.transforms import Bbox

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
TITLE_SIZE = 18
SUPTITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
TICK_LENGTH = 5
TICK_WIDTH = 1.1
LEGEND_SIZE = 12
ANNOTATION_SIZE = 11
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
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
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
    ax.tick_params(
        colors=MUTED,
        labelsize=TICK_SIZE,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )
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


def add_legend(fig, embedders, labels, color_map, location="lower center"):
    handles = [
        matplotlib.patches.Patch(color=color_map[tag], label=labels.get(tag, tag))
        for tag in embedders
    ]
    if location == "upper right":
        fig.legend(
            handles=handles,
            loc="upper right",
            ncol=1,
            frameon=False,
            fontsize=LEGEND_SIZE,
            bbox_to_anchor=(0.995, 0.94),
        )
        fig._mango_legend_location = "upper right"
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, -0.01),
    )


def add_axes_legend(ax, embedders, labels, color_map):
    """Place an embedder legend inside an axes at its upper-right corner."""
    handles = [
        matplotlib.patches.Patch(color=color_map[tag], label=labels.get(tag, tag))
        for tag in embedders
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        fontsize=LEGEND_SIZE,
    )


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
        ax.tick_params(
            axis="both",
            colors=MUTED,
            labelsize=TICK_SIZE,
            length=TICK_LENGTH,
            width=TICK_WIDTH,
        )
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


def _subplot_slug(ax, index):
    """Return a stable, readable filename component for one subplot."""
    candidates = [ax.get_title(), ax.get_ylabel(), ax.get_xlabel()]
    label = next((value for value in candidates if str(value).strip()), "")
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    return slug[:80] or f"panel_{index:02d}"


def _save_subplots(fig, out_figure, dpi):
    """Save each panel of a multi-panel figure as an independent PNG."""
    output = Path(out_figure)
    subplot_dir = output.parent / f"{output.stem}_subplots"
    visible_axes = [candidate for candidate in fig.axes if candidate.get_visible()]
    if len(visible_axes) <= 1:
        if subplot_dir.is_dir():
            for stale_path in subplot_dir.glob("*.png"):
                stale_path.unlink()
            subplot_dir.rmdir()
        return []

    subplot_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in subplot_dir.glob("*.png"):
        stale_path.unlink()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    written = []

    for index, ax in enumerate(visible_axes, start=1):
        temporary_legend = None
        if ax.get_legend() is None and fig.legends:
            source_legend = fig.legends[0]
            handles = getattr(source_legend, "legend_handles", None)
            if handles is None:
                handles = getattr(source_legend, "legendHandles", [])
            labels = [text.get_text() for text in source_legend.get_texts()]
            temporary_legend = ax.legend(
                handles,
                labels,
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.85,
                fontsize=LEGEND_SIZE,
            )
            fig.canvas.draw()

        bbox = ax.get_tightbbox(renderer)
        if temporary_legend is not None:
            bbox = Bbox.union([bbox, temporary_legend.get_window_extent(renderer)])
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(
            1.04, 1.08
        )
        subplot_path = subplot_dir / (
            f"{output.stem}_subplot_{index:02d}_{_subplot_slug(ax, index)}.png"
        )
        fig.savefig(
            subplot_path,
            dpi=max(300, int(dpi)),
            bbox_inches=bbox_inches,
            facecolor="white",
        )
        written.append(subplot_path)
        if temporary_legend is not None:
            temporary_legend.remove()
            fig.canvas.draw()
    return written


def save(fig, data, out_figure, out_data, dpi):
    normalize_typography(fig)
    Path(out_data).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)
    Path(out_figure).parent.mkdir(parents=True, exist_ok=True)
    if getattr(fig, "_mango_legend_location", None) == "upper right":
        fig.tight_layout(rect=(0, 0.02, 0.82, 0.97))
    else:
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(
        out_figure,
        dpi=max(300, int(dpi)),
        bbox_inches="tight",
        facecolor="white",
    )
    subplot_paths = _save_subplots(fig, out_figure, dpi)
    plt.close(fig)
    print(
        f"wrote {out_figure}, {len(subplot_paths)} subplot PNG(s), and {out_data}",
        flush=True,
    )
