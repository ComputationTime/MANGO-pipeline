"""Plot the frozen-ESM2-size ablation using pooled train/test MANGO NLL."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


SPLITS = ("train", "test")


def _parameter_count(size):
    suffix = str(size).rsplit("_", 1)[-1].upper()
    if suffix.endswith("M"):
        return int(suffix[:-1]) * 1_000_000
    if suffix.endswith("B"):
        return int(suffix[:-1]) * 1_000_000_000
    raise ValueError(f"cannot parse ESM2 parameter count from {size!r}")


def _size_label(size):
    return str(size).rsplit("_", 1)[-1].upper()


def plot_esm2_ablation(eval_jsons, embedders, specs, dpi, out_figure, out_data):
    if len(eval_jsons) != len(embedders):
        raise ValueError("one eval.json is required for every ESM2 ablation tag")

    payloads = {}
    for path in eval_jsons:
        with open(path) as handle:
            payload = json.load(handle)
        tag = payload.get("embedder")
        if tag in payloads:
            raise ValueError(f"duplicate ESM2 ablation evaluation for {tag!r}")
        payloads[tag] = payload

    missing = [tag for tag in embedders if tag not in payloads]
    if missing:
        raise ValueError(f"missing ESM2 ablation evaluation(s) for {missing}")

    ordered = sorted(
        embedders,
        key=lambda tag: _parameter_count(specs[tag]["size"]),
    )
    rows = []
    for tag in ordered:
        spec = specs[tag]
        if spec.get("method") != "esm2":
            raise ValueError(f"ESM2 ablation tag {tag!r} uses {spec.get('method')!r}")
        payload = payloads[tag]
        for split in SPLITS:
            if split not in payload.get("splits", {}):
                raise ValueError(f"evaluation for {tag!r} has no {split!r} split")
            values = payload["splits"][split]
            rows.append({
                "embedder": tag,
                "esm2_size": spec["size"],
                "esm2_parameters": _parameter_count(spec["size"]),
                "embedding_width": int(payload["d_ag"]),
                "run_id": payload.get("run_id"),
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                "split": split,
                "nll": values["nll"],
                "perplexity": values["perplexity"],
                "n_examples": values["n_examples"],
                "n_tokens": values["n_tokens"],
            })

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(6.4, 1.45 * len(ordered) + 2), 4.4))
    x = np.arange(len(ordered))
    width = 0.34
    color = pc.EMBEDDER_COLORS["esm2"]
    for offset, split, hatch, alpha in [
        (-width / 2, "train", None, 1.0),
        (width / 2, "test", "///", 0.55),
    ]:
        values = [
            data.loc[
                (data.embedder == tag) & (data.split == split), "nll"
            ].iloc[0]
            for tag in ordered
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=color,
            hatch=hatch,
            alpha=alpha,
            edgecolor=pc.TEXT,
            linewidth=0.8,
            label=split,
        )
        if len(ordered) <= 4:
            pc.label_bars(ax, bars, values)

    ax.set_xticks(
        x,
        [_size_label(specs[tag]["size"]) for tag in ordered],
    )
    pc.style(
        ax,
        "ESM2 size ablation",
        xlabel="Frozen ESM2 parameters",
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
            "plot_esm2_ablation.py is intended to run through Snakemake"
        )
    plot_esm2_ablation(
        list(smk.input.evals),
        list(smk.params.embedders),
        {tag: dict(spec) for tag, spec in dict(smk.params.specs).items()},
        smk.params.dpi,
        smk.output.figure,
        smk.output.data,
    )


if __name__ == "__main__":
    main()
