"""Render an atomically replaceable training/validation loss curve."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling_mean(values, window):
    means = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        means.append(running / min(index + 1, window))
    return means


def write_training_plot(history, output, run_id):
    """Atomically refresh an iteration-axis training/validation loss plot."""
    train = [point for point in history if point["phase"] == "train"]
    train_epoch = [point for point in history if point["phase"] == "train_epoch"]
    validation = [point for point in history if point["phase"] == "validation"]
    if not train:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_train = [point["iteration"] for point in train]
    y_train = [point["loss"] for point in train]
    ax.plot(
        x_train, y_train, color="#4C78A8", alpha=0.22, linewidth=0.7,
        label="Training loss (per iteration)",
    )
    if len(train) >= 10:
        window = min(100, max(10, len(train) // 20))
        ax.plot(
            x_train, _rolling_mean(y_train, window),
            color="#1F4E79", linewidth=1.8,
            label=f"Training loss ({window}-iteration mean)",
        )
    if validation:
        if train_epoch:
            ax.plot(
                [point["iteration"] for point in train_epoch],
                [point["loss"] for point in train_epoch],
                color="#54A24B", marker="o", linewidth=1.8,
                label="Training loss (epoch mean)",
            )
        ax.plot(
            [point["iteration"] for point in validation],
            [point["loss"] for point in validation],
            color="#E45756", marker="o", linewidth=1.8,
            label="Validation loss (epoch end)",
        )
    ax.set(
        title=f"Training curve: {run_id}",
        xlabel="Training iteration",
        ylabel="NLL loss",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)
