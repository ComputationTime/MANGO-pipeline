"""Plot handbook Figure 3: IgLM, AntiBERTy, and AbLang2 score panels."""

import sys
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


PANELS = [
    ("iglm", "iglm_log_likelihood", "IgLM Perplexity"),
    ("antiberty", "antiberty_pseudo_log_likelihood", "AntiBERTy Pseudo Perplexity"),
    ("ablang2", "ablang2_pseudo_log_likelihood", "AbLang2 Pseudo Perplexity"),
]
CHAIN_TYPES = ("Heavy", "Light")


def _read(paths, source):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["score_source"] = source
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_ab_likeness(inputs, embedders, labels, dpi, out_figure, out_data):
    raw = {name: _read(inputs[name], name) for name, *_ in PANELS}
    cmap = pc.colors(embedders)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.2))
    summary = []
    for ax, (source, column, ylabel) in zip(axes, PANELS):
        frame = raw[source]
        if "chain_type" not in frame:
            raise ValueError(
                f"{source} metrics must include chain_type for separate Heavy/Light plots"
            )
        if source == "ablang2":
            modes = set(frame["ablang2_mode"].dropna().astype(str))
            if modes != {"pseudo_log_likelihood"} or column not in frame:
                raise ValueError(
                    "AbLang2 pseudo perplexity requires metrics generated with "
                    "analysis.ab_likeness.ablang2.mode=pseudo_log_likelihood"
                )
        chain_statistics = {}
        for chain_type in CHAIN_TYPES:
            medians, lower, upper = [], [], []
            for tag in embedders:
                selected = (
                    frame["embedder"].eq(tag)
                    & frame["chain_type"].astype(str).str.casefold().eq(
                        chain_type.casefold()
                    )
                )
                log_likelihoods = pd.to_numeric(
                    frame.loc[selected, column], errors="coerce"
                ).dropna()
                values = np.exp(-log_likelihoods)
                median = values.median()
                q25, q75 = (
                    values.quantile([0.25, 0.75])
                    if len(values)
                    else (np.nan, np.nan)
                )
                medians.append(median)
                lower.append(median - q25)
                upper.append(q75 - median)
                summary.append({
                    "embedder": tag,
                    "chain_type": chain_type,
                    "score_source": source,
                    "score_column": column,
                    "transform": "exp(-score)",
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "n": len(values),
                })
            chain_statistics[chain_type] = (medians, lower, upper)
        if not any(item["score_source"] == source and item["n"] > 0 for item in summary):
            pc.empty_panel(ax, "", "No valid scores in this smoke cohort",
                           ylabel=ylabel)
            ax.texts[-1].set_position((0.5, 0.25))
            continue
        x = np.arange(len(embedders))
        width = 0.36
        data_tops = []
        for index, chain_type in enumerate(CHAIN_TYPES):
            medians, lower, upper = chain_statistics[chain_type]
            positions = x + (index - 0.5) * width
            bars = ax.bar(
                positions,
                medians,
                width,
                color=[cmap[tag] for tag in embedders],
                alpha=1.0 if chain_type == "Heavy" else 0.45,
                hatch="" if chain_type == "Heavy" else "//",
                edgecolor="white" if chain_type == "Light" else "none",
                linewidth=0.6 if chain_type == "Light" else 0,
            )
            ax.errorbar(
                positions,
                medians,
                yerr=[lower, upper],
                fmt="none",
                color=pc.MUTED,
                capsize=3,
                linewidth=1.2,
            )
            if len(embedders) <= 4:
                pc.label_bars(ax, bars, medians)
            data_tops.extend(np.asarray(medians) + np.asarray(upper))
        ax.set_xticks(x, [labels.get(t, t) for t in embedders], rotation=25, ha="right")
        pc.style(ax, "", ylabel=ylabel)
        ax.legend(
            handles=[
                Patch(facecolor=pc.MUTED, label="Heavy"),
                Patch(
                    facecolor=pc.MUTED,
                    alpha=0.45,
                    hatch="//",
                    edgecolor="white",
                    label="Light",
                ),
            ],
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
        )
        finite_tops = np.asarray(data_tops, dtype=float)
        finite_tops = finite_tops[np.isfinite(finite_tops)]
        if len(finite_tops):
            ax._mango_data_top = float(finite_tops.max())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        data_top = getattr(ax, "_mango_data_top", None)
        if data_top is None:
            continue
        legend = ax.get_legend()
        legend_bottom = legend.get_window_extent(renderer).transformed(
            ax.transAxes.inverted()
        ).y0
        available_fraction = max(0.12, legend_bottom - 0.08)
        bottom = min(0.0, ax.get_ylim()[0])
        required_top = bottom + (data_top - bottom) * 1.05 / available_fraction
        ax.set_ylim(bottom=bottom, top=max(ax.get_ylim()[1], required_top))
    pc.save(fig, pd.DataFrame(summary), out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_ab_likeness.py is intended to run through Snakemake")
    plot_ab_likeness({name: list(smk.input[name]) for name, *_ in PANELS},
                     list(smk.params.embedders), dict(smk.params.labels), smk.params.dpi,
                     smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
