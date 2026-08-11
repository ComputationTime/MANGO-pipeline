"""Plot handbook Figure 1: train and ab_ag_cluster-held-out test NLL."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


SPLITS = ("train", "test")


def plot_nll(eval_jsons, embedders, labels, dpi, out_figure, out_data):
    rows = []
    for path in eval_jsons:
        with open(path) as handle:
            payload = json.load(handle)
        for split in SPLITS:
            if split not in payload.get("splits", {}):
                raise ValueError(f"{path} has no required {split!r} evaluation")
            values = payload["splits"][split]
            rows.append({"embedder": payload["embedder"], "split": split,
                         "nll": values["nll"], "perplexity": values["perplexity"],
                         "n_examples": values["n_examples"], "n_tokens": values["n_tokens"]})
    data = pd.DataFrame(rows)
    cmap = pc.colors(embedders)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(embedders) + 2), 4.4))
    x = np.arange(len(embedders))
    width = 0.34
    for offset, split, alpha in [(-width / 2, "train", 1.0),
                                  (width / 2, "test", 0.48)]:
        values = [data.loc[(data.embedder == tag) & (data.split == split), "nll"].iloc[0]
                  for tag in embedders]
        bars = ax.bar(x + offset, values, width, color=[cmap[tag] for tag in embedders],
                      alpha=alpha, edgecolor="white", linewidth=1.5, label=split)
        if len(embedders) <= 4:
            pc.label_bars(ax, bars, values)
    ax.set_xticks(x, [labels.get(tag, tag) for tag in embedders], rotation=25, ha="right")
    pc.style(ax, "Antigen representation vs. heavy-chain likelihood",
             ylabel="NLL (nats / predicted heavy-chain token)")
    ax.legend(frameon=False, title="split")
    pc.save(fig, data, out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_nll.py is intended to run through Snakemake")
    plot_nll(list(smk.input.evals), list(smk.params.embedders), dict(smk.params.labels),
             smk.params.dpi, smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
