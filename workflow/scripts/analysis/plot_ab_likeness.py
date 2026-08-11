"""Plot handbook Figure 3: IgLM, AntiBERTy, and AbLang2 score panels."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


PANELS = [
    ("iglm", "iglm_log_likelihood", "IgLM", "mean log likelihood"),
    ("antiberty", "antiberty_pseudo_log_likelihood", "AntiBERTy", "mean pseudo-log-likelihood"),
    ("ablang2", None, "AbLang2", "confidence"),
]


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
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    summary = []
    for ax, (source, configured_column, title, ylabel) in zip(axes, PANELS):
        frame = raw[source]
        if source == "ablang2":
            candidates = [c for c in frame.columns if c.startswith("ablang2_")
                          and c not in {"ablang2_status", "ablang2_mode"}]
            if len(candidates) != 1:
                raise ValueError(f"expected one AbLang2 score column, found {candidates}")
            column = candidates[0]
            ylabel = frame["ablang2_mode"].iloc[0].replace("_", " ")
        else:
            column = configured_column
        medians, lower, upper = [], [], []
        for tag in embedders:
            values = pd.to_numeric(frame.loc[frame.embedder == tag, column],
                                   errors="coerce").dropna()
            median = values.median()
            q25, q75 = values.quantile([0.25, 0.75]) if len(values) else (np.nan, np.nan)
            medians.append(median); lower.append(median - q25); upper.append(q75 - median)
            summary.append({"embedder": tag, "score_source": source,
                            "score_column": column, "median": median,
                            "q25": q25, "q75": q75, "n": len(values)})
        if not any(item["score_source"] == source and item["n"] > 0 for item in summary):
            pc.empty_panel(ax, title, "No valid scores in this smoke cohort",
                           ylabel=ylabel)
            continue
        x = np.arange(len(embedders))
        bars = ax.bar(x, medians, 0.72, color=[cmap[t] for t in embedders],
                      edgecolor="white", linewidth=1.5)
        ax.errorbar(x, medians, yerr=[lower, upper], fmt="none", color=pc.MUTED,
                    capsize=3, linewidth=1.2)
        if len(embedders) <= 4:
            pc.label_bars(ax, bars, medians)
        ax.set_xticks(x, [labels.get(t, t) for t in embedders], rotation=25, ha="right")
        pc.style(ax, title, ylabel=ylabel)
    fig.suptitle("Antibody-likeness of generated heavy chains", x=0.01, ha="left")
    pc.add_legend(fig, embedders, labels, cmap)
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
