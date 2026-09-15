"""Plot V- and J-gene family counts using the project reference design."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def _read_inputs(file_paths):
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    return pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)


def plot_species_distribution(
    file_paths,
    embedders,
    labels,
    out_figure,
    out_data,
    dpi=300,
):
    """Plot nearest-germline species counts from corrected ANARCI output."""
    data = _read_inputs(file_paths)
    required = {"embedder", "germline_species"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"ANARCI output is missing columns: {sorted(missing)}")

    valid = data.loc[data["germline_species"].fillna("").astype(str) != ""].copy()
    if valid.empty:
        raise RuntimeError("ANARCI produced no nearest-germline species assignments")
    valid["germline_species"] = valid["germline_species"].astype(str).str.lower()

    species_order = valid["germline_species"].value_counts().index.tolist()
    if "human" in species_order:
        species_order = ["human"] + [
            species for species in species_order if species != "human"
        ]

    cmap = pc.colors(embedders)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.countplot(
        data=valid,
        x="germline_species",
        hue="germline_species",
        hue_order=species_order,
        order=species_order,
        palette="colorblind",
        legend=False,
        ax=ax,
    )
    pc.style(
        ax,
        "Nearest-germline species assigned by ANARCI",
        xlabel="Species",
        ylabel="Number of sequences",
    )
    legend = ax.get_legend()
    if len(embedders) == 1 and legend is not None:
        legend.remove()
    elif legend is not None:
        legend.set_title("Antigen representation")
        for text, tag in zip(legend.get_texts(), embedders):
            text.set_text(labels.get(tag, tag))

    summary = (
        valid.groupby(["embedder", "germline_species"])
        .size()
        .rename("count")
        .reset_index()
    )
    pc.save(fig, summary, out_figure, out_data, dpi)


def plot_all_embedder_genes(
    file_paths,
    embedders,
    labels,
    out_figure,
    out_data,
    dpi=300,
    species_paths=None,
):
    df = _read_inputs(file_paths)
    species_df = _read_inputs(species_paths if species_paths is not None else file_paths)

    required_cols = ["v_gene", "j_gene", "embedder"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"The CSV must contain the column {col!r}.")

    def club_v_family(gene):
        if pd.isna(gene):
            return None
        gene_str = str(gene).upper()
        match = re.search(r"V(\d+)", gene_str)
        if match:
            return f"IGV{match.group(1)}"
        return None

    def club_j_family(gene):
        if pd.isna(gene):
            return None
        gene_str = str(gene).upper()
        match = re.search(r"J(\d+)", gene_str)
        if match:
            return f"IGJ{match.group(1)}"
        return None

    df["v_family"] = df["v_gene"].apply(club_v_family)
    df["j_family"] = df["j_gene"].apply(club_j_family)
    df_plot = df.dropna(subset=["v_family", "j_family"])

    if df_plot.empty:
        raise RuntimeError("No valid V or J gene families could be extracted.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    v_order = df_plot["v_family"].value_counts().index
    j_order = df_plot["j_family"].value_counts().index
    cmap = pc.colors(embedders)

    sns.countplot(
        data=df_plot,
        x="v_family",
        hue="embedder",
        hue_order=embedders,
        palette=cmap,
        ax=axes[0],
        order=v_order,
    )
    axes[0].set_title("")
    axes[0].set_xlabel("V Gene Family", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    sns.countplot(
        data=df_plot,
        x="j_family",
        hue="embedder",
        hue_order=embedders,
        ax=axes[1],
        order=j_order,
        palette=cmap,
    )
    axes[1].set_title("")
    axes[1].set_xlabel("J Gene Family", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    species_summary = []
    if "germline_species" not in species_df:
        pc.empty_panel(
            axes[2],
            "",
            "No ANARCI species assignments available",
            xlabel="Nearest-germline species assigned by ANARCI",
            ylabel="Number of sequences identified by ANARCI",
        )
    else:
        assigned = species_df.copy()
        assigned["germline_species"] = (
            assigned["germline_species"].fillna("").astype(str).str.lower()
        )
        assigned = assigned.loc[assigned["germline_species"] != ""].copy()
        if assigned.empty:
            pc.empty_panel(
                axes[2],
                "",
                "No ANARCI species assignments available",
                xlabel="Nearest-germline species assigned by ANARCI",
                ylabel="Number of sequences identified by ANARCI",
            )
        else:
            species_order = assigned["germline_species"].value_counts().index.tolist()
            if "human" in species_order:
                species_order = ["human"] + [
                    species for species in species_order if species != "human"
                ]
            sns.countplot(
                data=assigned,
                x="germline_species",
                hue="embedder",
                hue_order=embedders,
                order=species_order,
                palette=cmap,
                ax=axes[2],
            )
            axes[2].set_title("")
            axes[2].set_xlabel(
                "Nearest-germline species assigned by ANARCI", fontsize=12
            )
            axes[2].set_ylabel(
                "Number of sequences identified by ANARCI", fontsize=12
            )
            axes[2].tick_params(axis="x", rotation=45)
            species_legend = axes[2].get_legend()
            if species_legend is not None:
                species_legend.remove()
        species_summary = [
            {
                "plot": "species",
                "embedder": tag,
                "category": species,
                "count": int(count),
            }
            for (tag, species), count in assigned.groupby(
                ["embedder", "germline_species"]
            ).size().items()
            if species
        ]

    handles, raw_legend_labels = axes[0].get_legend_handles_labels()
    for ax in axes[:2]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        [labels.get(tag, tag) for tag in raw_legend_labels],
        title="Embedder",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(4, len(raw_legend_labels)),
        frameon=False,
    )

    gene_summary = []
    for segment, column in (("V", "v_family"), ("J", "j_family")):
        gene_summary.extend(
            {
                "plot": segment,
                "embedder": tag,
                "category": family,
                "count": int(count),
            }
            for (tag, family), count in df_plot.groupby(
                ["embedder", column]
            ).size().items()
        )
    pc.save(
        fig,
        pd.DataFrame(gene_summary + species_summary),
        out_figure,
        out_data,
        dpi,
    )


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_genes.py is intended to run through Snakemake")
    plot_all_embedder_genes(
        list(smk.input.metrics),
        list(smk.params.embedders),
        dict(smk.params.labels),
        smk.output.figure,
        smk.output.data,
        smk.params.dpi,
    )


if __name__ == "__main__":
    main()
