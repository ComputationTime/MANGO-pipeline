"""LEGACY combined plot prototype; not included by the active Snakefile.

Comparison figures: one function per handbook figure family.

Every figure answers the same question in a different currency -- does the
antigen representation matter? -- so they share a visual grammar:

  * colour encodes the EMBEDDER (identity), assigned in config order and never
    cycled, so a given representation is the same colour in every panel;
  * one measure per axis, never a second y-scale;
  * a legend is always present, and bars are directly labelled when there are
    few enough to read;
  * every figure also writes `<figure>_data.csv` -- the exact numbers behind the
    panels, which doubles as the accessible table view.

Palette: the validated 8-slot categorical set (adjacent-pair CVD dE 9.1, normal
vision 19.6 in light mode). Three slots sit below 3:1 contrast on white, which
is why direct labels + the data CSV are mandatory rather than optional.

Adding a figure: write `_fig_<name>(...)`, register it in PLOTS, and declare its
inputs in `_figure_inputs()` in workflow/rules/analysis.smk.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on a cluster node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical palette, light mode. Slots are assigned in config order.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2df"

# Splits keep a fixed order and a fixed lightness ramp within a figure.
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_ALPHA = {"train": 1.0, "val": 0.62, "test": 0.34}

DIRECT_LABEL_MAX = 4  # bars get value labels when at most this many embedders


def _colors(embedders):
    if len(embedders) > len(PALETTE):
        raise ValueError(
            f"{len(embedders)} embedders exceeds the {len(PALETTE)}-slot validated "
            "palette. Fold the extras into a separate figure rather than "
            "generating new hues."
        )
    return {t: PALETTE[i] for i, t in enumerate(embedders)}


def _style(ax, xlabel="", ylabel="", title=""):
    """Recessive axes: horizontal grid only, no top/right spines."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=8)


def _empty_panel(ax, message, title=""):
    """A panel whose metric could not be computed says so, rather than lying.

    The title is kept: a reader looking at a blank panel needs to know which
    measure is missing, especially when the whole figure is one panel.
    """
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
        color=INK_SECONDARY, fontsize=9, wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=8)


def _label_bars(ax, bars, values, fmt="{:.2f}"):
    for bar, value in zip(bars, values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 3),
            ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
        )


def _legend(fig, embedders, labels, colors):
    handles = [
        matplotlib.patches.Patch(facecolor=colors[t], edgecolor="none",
                                 label=labels.get(t, t))
        for t in embedders
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 4),
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.5, -0.02),
    )


# --- figure 1: NLL vs antigen embedding --------------------------------------
def _fig_nll(inputs, embedders, labels, colors, cfg):
    """Grouped bars: one group per embedder, one bar per split."""
    rows = []
    for path in inputs["evals"]:
        with open(path) as fh:
            data = json.load(fh)
        for split, values in data.get("splits", {}).items():
            rows.append(
                {
                    "embedder": data.get("embedder", ""),
                    "split": split,
                    "nll": values.get("nll"),
                    "perplexity": values.get("perplexity"),
                    "n_examples": values.get("n_examples"),
                    "n_tokens": values.get("n_tokens"),
                }
            )
    df = pd.DataFrame(rows)

    splits = [s for s in SPLIT_ORDER if s in set(df["split"])]
    fig, ax = plt.subplots(figsize=(1.7 * max(len(embedders), 3) + 2, 4.2))

    # Groups occupy 70% of the slot so adjacent embedders read as separate groups.
    group_span = 0.7
    width = group_span / max(len(splits), 1)
    x = np.arange(len(embedders))
    all_vals = []
    for j, split in enumerate(splits):
        vals = [
            df.loc[(df.embedder == t) & (df.split == split), "nll"].mean()
            for t in embedders
        ]
        all_vals += [v for v in vals if v == v]
        bars = ax.bar(
            x + j * width - group_span / 2 + width / 2, vals, width * 0.9,
            color=[colors[t] for t in embedders], alpha=SPLIT_ALPHA.get(split, 1.0),
            edgecolor="white", linewidth=2, label=split,
        )
        if len(embedders) <= DIRECT_LABEL_MAX:
            _label_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(t, t) for t in embedders], fontsize=9)
    # Headroom so the direct labels never collide with the top of the axes.
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.15)
    _style(ax, ylabel="NLL (nats / token)",
           title="Antigen representation vs. held-out likelihood")

    # Split is encoded by alpha, so it needs its own legend. It sits ABOVE the
    # axes: inside, it collides with whichever bar happens to be tallest.
    split_handles = [
        matplotlib.patches.Patch(facecolor=INK_SECONDARY,
                                 alpha=SPLIT_ALPHA.get(s, 1.0), label=s)
        for s in splits
    ]
    ax.legend(
        handles=split_handles, frameon=False, fontsize=9,
        labelcolor=INK_SECONDARY, loc="lower right",
        bbox_to_anchor=(1.0, 1.01), ncol=len(split_handles),
    )
    return fig, df


# --- figure 2: pTM across structure predictors -------------------------------
def _fig_ptm(inputs, embedders, labels, colors, cfg):
    """Small multiples: one panel per predictor, bars = mean pTM per embedder."""
    frames = [pd.read_csv(p) for p in inputs["scores"]]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    methods = sorted(df["sp_method"].unique()) if not df.empty else []
    n = max(len(methods), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True, squeeze=False)

    for ax, method in zip(axes[0], methods or [None]):
        if method is None or df.empty:
            _empty_panel(ax, "no structure-prediction scores yet",
                         title=str(method) if method else "pTM")
            continue
        sub = df[df.sp_method == method]
        means = [sub.loc[sub.embedder == t, "ptm"].mean() for t in embedders]
        errs = [sub.loc[sub.embedder == t, "ptm"].sem() for t in embedders]
        bars = ax.bar(
            np.arange(len(embedders)), means, 0.72,
            color=[colors[t] for t in embedders], edgecolor="white", linewidth=2,
        )
        ax.errorbar(
            np.arange(len(embedders)), means, yerr=errs, fmt="none",
            ecolor=INK_SECONDARY, elinewidth=1.5, capsize=3,
        )
        if len(embedders) <= DIRECT_LABEL_MAX:
            _label_bars(ax, bars, means)
        ax.set_xticks(np.arange(len(embedders)))
        ax.set_xticklabels([labels.get(t, t) for t in embedders],
                           rotation=30, ha="right", fontsize=9)
        _style(ax, ylabel="pTM" if ax is axes[0][0] else "", title=str(method))

    fig.suptitle("Predicted complex confidence by antigen representation",
                 color=INK_PRIMARY, fontsize=11, x=0.01, ha="left")
    _legend(fig, embedders, labels, colors)
    return fig, df


# --- shared loader for the seq-metric figures --------------------------------
def _load_metrics(inputs):
    frames = [pd.read_csv(p) for p in inputs["metrics"]]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bar_panel(ax, df, column, embedders, labels, colors, title, ylabel):
    if df.empty or column not in df.columns or df[column].isna().all():
        _empty_panel(ax, f"{column}\nnot available\n(see seq_metrics status columns)",
                     title=title)
        return
    medians = [df.loc[df.embedder == t, column].median() for t in embedders]
    bars = ax.bar(
        np.arange(len(embedders)), medians, 0.72,
        color=[colors[t] for t in embedders], edgecolor="white", linewidth=2,
    )
    if len(embedders) <= DIRECT_LABEL_MAX:
        _label_bars(ax, bars, medians, fmt="{:.3f}")
    ax.set_xticks(np.arange(len(embedders)))
    ax.set_xticklabels([labels.get(t, t) for t in embedders],
                       rotation=30, ha="right", fontsize=9)
    _style(ax, ylabel=ylabel, title=title)


def _dist_panel(ax, df, column, embedders, labels, colors, title, xlabel, bins=40):
    if df.empty or column not in df.columns or df[column].isna().all():
        _empty_panel(ax, f"{column}\nnot available\n(see seq_metrics status columns)",
                     title=title)
        return
    values = df[column].dropna()
    if values.empty:
        _empty_panel(ax, f"{column}: no finite values", title=title)
        return
    edges = np.linspace(values.min(), values.max(), bins + 1)
    for t in embedders:
        series = df.loc[df.embedder == t, column].dropna()
        if series.empty:
            continue
        ax.hist(
            series, bins=edges, histtype="step", linewidth=2,
            color=colors[t], density=True,
        )
    _style(ax, xlabel=xlabel, ylabel="density", title=title)


# --- figure 3: antibody-likeness ---------------------------------------------
def _fig_ablikeness(inputs, embedders, labels, colors, cfg):
    df = _load_metrics(inputs)
    panels = [
        ("iglm_confidence", "IgLM confidence", "median confidence"),
        ("ablang2_confidence", "AbLang2 confidence", "median confidence"),
        ("frac_charged", "% charged residues", "median fraction"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (col, title, ylabel) in zip(axes, panels):
        _bar_panel(ax, df, col, embedders, labels, colors, title, ylabel)
    fig.suptitle("Are the designs antibody-like?", color=INK_PRIMARY,
                 fontsize=11, x=0.01, ha="left")
    _legend(fig, embedders, labels, colors)
    return fig, df


# --- figure 4: distance from germline ----------------------------------------
def _fig_ld_germline(inputs, embedders, labels, colors, cfg):
    df = _load_metrics(inputs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _dist_panel(
        ax, df, "ld_germline", embedders, labels, colors,
        title="How far from germline does each representation push generation?",
        xlabel="Levenshtein distance to nearest germline",
    )
    _legend(fig, embedders, labels, colors)
    return fig, df


# --- figure 5: developability ------------------------------------------------
def _fig_developability(inputs, embedders, labels, colors, cfg):
    df = _load_metrics(inputs)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    _dist_panel(axes[0], df, "gravy", embedders, labels, colors,
                "GRAVY hydrophobicity", "GRAVY")
    _dist_panel(axes[1], df, "charge_at_pH", embedders, labels, colors,
                "Charge at pH 7.4", "net charge")
    _bar_panel(axes[2], df, "tap_score", embedders, labels, colors,
               "TAP developability", "median TAP score")
    fig.suptitle("Developability of de novo designs", color=INK_PRIMARY,
                 fontsize=11, x=0.01, ha="left")
    _legend(fig, embedders, labels, colors)
    return fig, df


PLOTS = {
    "fig1_nll": _fig_nll,
    "fig2_ptm": _fig_ptm,
    "fig3_ablikeness": _fig_ablikeness,
    "fig4_ld_germline": _fig_ld_germline,
    "fig5_developability": _fig_developability,
}


def make_plot(figure, inputs, embedders, labels, classes, plot_cfg, dpi,
              out_figure, out_data):
    if figure not in PLOTS:
        raise ValueError(
            f"unknown figure {figure!r}; known: {sorted(PLOTS)}. "
            "Register new figures in PLOTS and in _figure_inputs()."
        )
    colors = _colors(list(embedders))
    fig, data = PLOTS[figure](inputs, list(embedders), dict(labels), colors,
                              dict(plot_cfg or {}))

    Path(out_data).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)

    Path(out_figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_figure, dpi=int(dpi), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote {out_figure} and {out_data} ({len(data)} rows)", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        inputs = {k: list(v) for k, v in smk.input.items()}
        make_plot(
            figure=smk.params.figure,
            inputs=inputs,
            embedders=list(smk.params.embedders),
            labels=dict(smk.params.labels),
            classes=dict(smk.params.classes),
            plot_cfg=dict(smk.params.plot_cfg),
            dpi=smk.params.dpi,
            out_figure=smk.output.figure,
            out_data=smk.output.data,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", required=True, choices=sorted(PLOTS))
    p.add_argument("--inputs", required=True, help="JSON {key: [paths]}")
    p.add_argument("--embedders", nargs="+", required=True)
    p.add_argument("--labels", default="{}", help="JSON {tag: label}")
    p.add_argument("--out-figure", required=True)
    p.add_argument("--out-data", required=True)
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    make_plot(
        a.figure, json.loads(a.inputs), a.embedders, json.loads(a.labels),
        {}, {}, a.dpi, a.out_figure, a.out_data,
    )


if __name__ == "__main__":
    main()
