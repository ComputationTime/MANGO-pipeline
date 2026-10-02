"""Plot MANGO antigen-conditioning controls against the matched ESM2 run."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


SPLITS = ("train", "test")


def plot_antigen_controls(
    eval_jsons, embedders, labels, dpi, out_figure, out_data
):
    if len(eval_jsons) != len(embedders):
        raise ValueError("one eval.json is required for every control-plot tag")

    payloads = {}
    for path in eval_jsons:
        with open(path) as handle:
            payload = json.load(handle)
        tag = payload.get("embedder")
        if tag in payloads:
            raise ValueError(f"duplicate control evaluation for {tag!r}")
        payloads[tag] = payload

    missing = [tag for tag in embedders if tag not in payloads]
    if missing:
        raise ValueError(f"missing antigen-control evaluation(s) for {missing}")

    rows = []
    expected_counts = None
    for tag in embedders:
        payload = payloads[tag]
        for split in SPLITS:
            if split not in payload.get("splits", {}):
                raise ValueError(f"evaluation for {tag!r} has no {split!r} split")
            values = payload["splits"][split]
            counts = (values["n_examples"], values["n_tokens"])
            if expected_counts is None:
                expected_counts = {}
            if split in expected_counts and expected_counts[split] != counts:
                raise ValueError(
                    f"{tag!r} {split} cohort {counts} does not match "
                    f"{expected_counts[split]}"
                )
            expected_counts[split] = counts
            rows.append({
                "embedder": tag,
                "label": labels.get(tag, tag),
                "run_id": payload.get("run_id"),
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                "embedding_width": int(payload["d_ag"]),
                "split": split,
                "nll": values["nll"],
                "perplexity": values["perplexity"],
                "n_examples": values["n_examples"],
                "n_tokens": values["n_tokens"],
            })

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8.2, 1.55 * len(embedders) + 2), 4.6))
    x = np.arange(len(embedders))
    width = 0.34
    color_map = pc.colors(embedders)
    bar_colors = [color_map[tag] for tag in embedders]
    for offset, split, hatch, alpha in [
        (-width / 2, "train", None, 1.0),
        (width / 2, "test", "///", 0.58),
    ]:
        values = [
            data.loc[
                (data.embedder == tag) & (data.split == split), "nll"
            ].iloc[0]
            for tag in embedders
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=bar_colors,
            hatch=hatch,
            alpha=alpha,
            edgecolor=pc.TEXT,
            linewidth=0.8,
            label=split,
        )
        pc.label_bars(ax, bars, values)

    ax.set_xticks(x, [labels.get(tag, tag) for tag in embedders])
    ax.tick_params(axis="x", labelrotation=12)
    pc.style(
        ax,
        "Antigen conditioning controls",
        xlabel="Conditioning representation",
        ylabel="Pooled heavy/light NLL",
    )
    ax.legend(
        frameon=False,
        title="split",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    pc.save(fig, data, out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError(
            "plot_antigen_controls.py is intended to run through Snakemake"
        )
    plot_antigen_controls(
        list(smk.input.evals),
        list(smk.params.embedders),
        dict(smk.params.labels),
        smk.params.dpi,
        smk.output.figure,
        smk.output.data,
    )


if __name__ == "__main__":
    main()
