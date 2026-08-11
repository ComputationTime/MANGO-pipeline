"""Shared visual grammar for handbook plots; contains no metric computation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"


def colors(embedders):
    if len(embedders) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} embedders fit the fixed palette")
    return dict(zip(embedders, PALETTE))


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.xaxis.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def label_bars(ax, bars, values, fmt="{:.3f}"):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.annotate(fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=MUTED)


def add_legend(fig, embedders, labels, color_map):
    handles = [
        matplotlib.patches.Patch(color=color_map[tag], label=labels.get(tag, tag))
        for tag in embedders
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))


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
            color=MUTED, fontsize=9, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, data, out_figure, out_data, dpi):
    Path(out_data).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)
    Path(out_figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out_figure, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_figure} and {out_data}", flush=True)
