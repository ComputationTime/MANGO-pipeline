"""Plot handbook Figure 4: LD to the nearest ANARCI heavy V/J germline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def plot_germline(paths, embedders, labels, dpi, out_figure, out_data):
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    valid = data.loc[data["germline_status"] == "ok"].copy()
    cmap = pc.colors(embedders)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    title = "Distance from nearest ANARCI heavy-chain germline"
    if valid.empty:
        pc.empty_panel(ax, title, "ANARCI assigned no generated sequence",
                       xlabel="Levenshtein distance", ylabel="density")
    else:
        edges = pc.shared_edges(valid, "ld_germline", bins=30)
        for tag in embedders:
            values = valid.loc[valid.embedder == tag, "ld_germline"].dropna()
            ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                    color=cmap[tag], label=labels.get(tag, tag))
        pc.style(ax, title, xlabel="Levenshtein distance", ylabel="density")
    pc.add_legend(fig, embedders, labels, cmap)
    columns = ["embedder", "run_id", "target_id", "design_index", "sequence",
               "germline_species", "v_gene", "v_identity", "j_gene", "j_identity",
               "germline_reference_sequence", "ld_germline",
               "ld_germline_normalized", "germline_status"]
    pc.save(fig, data[[c for c in columns if c in data]], out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_germline.py is intended to run through Snakemake")
    plot_germline(list(smk.input.metrics), list(smk.params.embedders),
                   dict(smk.params.labels), smk.params.dpi,
                   smk.output.figure, smk.output.data)


if __name__ == "__main__":
    main()
