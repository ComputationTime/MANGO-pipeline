"""Plot FLAb average perplexity against AlphaSeq affinity, one model at a time.

The figure matches the HER2 binding scatter: white background, blue points,
and a red regression line. Pass any trained model's average-perplexity column.
This module does not import the shared MANGO plot style, which would recolor
the axes.
"""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flab_common import MODEL_LABELS, load_score_tables, model_ids, perplexity_column


def _pearson(x, y):
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 1.0)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), x, y
    statistic, _pvalue = pearsonr(x, y)
    if not np.isfinite(statistic):
        return float("nan"), x, y
    return float(statistic), x, y


def _resolve_column(frame: pd.DataFrame, model: str, perplexity_column_name: str | None) -> str:
    if perplexity_column_name:
        if perplexity_column_name not in frame.columns:
            raise ValueError(f"missing perplexity column {perplexity_column_name}")
        return perplexity_column_name
    column = perplexity_column(model)
    if column not in frame.columns:
        raise ValueError(
            f"{frame.shape[0]} rows have no {column}. "
            "Score that trained model before plotting it."
        )
    return column


def plot_average_perplexity(
    frame: pd.DataFrame,
    target: str,
    model: str,
    out_figure,
    label: str | None = None,
    perplexity_column_name: str | None = None,
    affinity_column: str = "alphaseq_affinity",
):
    """Draw one binding scatter for one already-scored model.

    ``model`` is the score prefix, such as ``iglm`` or ``esm2``. The x values
    are ``{model}_average_perplexity`` unless a column is passed explicitly.
    """
    if "target" in frame.columns:
        selected = frame.loc[frame["target"].astype(str) == str(target)].copy()
    else:
        selected = frame.copy()
    if selected.empty:
        raise ValueError(f"no rows for target {target}")
    column = _resolve_column(selected, model, perplexity_column_name)
    if affinity_column not in selected.columns:
        raise ValueError(f"missing {affinity_column}")
    display = label or MODEL_LABELS.get(model, model)
    correlation, x, y = _pearson(selected[column], selected[affinity_column])
    if len(x) < 3:
        raise ValueError(
            f"{display} has {len(x)} valid perplexities for {target}; "
            "need at least 3 values greater than 1"
        )

    # Local style only. Do not import plot_common; it paints every axes gray.
    figure, ax = plt.subplots(figsize=(8, 6))
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    try:
        import seaborn as sns
        plot_frame = pd.DataFrame({
            "average_perplexity": x,
            "alphaseq_affinity": y,
        })
        sns.regplot(
            data=plot_frame,
            x="average_perplexity",
            y="alphaseq_affinity",
            ax=ax,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"},
        )
    except ImportError:
        ax.scatter(x, y, alpha=0.6, linewidths=0, c="#1f77b4")
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 50)
        ax.plot(grid, slope * grid + intercept, color="red", linewidth=1.5)

    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    title_r = f"{correlation:.3f}" if np.isfinite(correlation) else "n/a"
    ax.set_title(f"{target} Binding Data\nPearson r = {title_r}", fontsize=21)
    ax.set_xlabel(f"{display} Average Perplexity", fontsize=18)
    ax.set_ylabel(r"AlphaSeq Affinity [$\mathrm{log}_{10}(K_D)$]", fontsize=18)
    ax.tick_params(labelsize=12)
    figure.tight_layout()
    out_path = Path(out_figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"wrote {out_path} ({display}, n={len(x)}, Pearson r={title_r})", flush=True)
    return {"target": target, "model": model, "n": int(len(x)), "pearson_r": correlation}


def plot_models(frame, target, models, out_dir, labels=None):
    """Write one HER2-style figure for each named model that has scores."""
    labels = labels or {}
    out_dir = Path(out_dir)
    written = []
    for model in models:
        if isinstance(model, tuple):
            model, label = model
        else:
            label = labels.get(model, MODEL_LABELS.get(model, model))
        figure = out_dir / f"{target}_{model}.png"
        written.append(plot_average_perplexity(
            frame, target, model, figure, label=label,
        ))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="One or more scored model names, for example iglm esm2 one_hot",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = load_score_tables(args.scores)
    missing = [model for model in args.model if model not in model_ids(frame)]
    if missing:
        raise SystemExit(
            f"no average perplexity for {missing}; available: {model_ids(frame)}"
        )
    plot_models(frame, args.target, args.model, args.out_dir)


if __name__ == "__main__":
    main()
