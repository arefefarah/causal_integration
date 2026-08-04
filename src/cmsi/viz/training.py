"""Training diagnostics: did it converge, and did every output get learned?"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.viz.style import COLORS


def loss_curves(history):
    """Train and validation loss, with the early-stopping point marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train", color=COLORS["network"])
    ax.plot(history["val"], label="validation", color=COLORS["analytical"])
    ax.axvline(history["best_epoch"], ls="--", lw=1, color="grey",
               label=f"best (epoch {history['best_epoch']})")
    ax.set(xlabel="epoch", ylabel="loss", yscale="log", title="training")
    ax.legend()
    fig.tight_layout()
    return fig


def per_output_loss(history):
    """Validation loss split by output -- catches one output being starved.

    A flat, high curve for a single output while the others fall is the classic
    scale-imbalance signature; check standardize_inputs and balance_loss.
    """
    per_output = np.asarray(history["val_per_output"])
    names = history["target_names"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(names):
        ax.plot(per_output[:, i], label=name)
    ax.set(xlabel="epoch", ylabel="weighted MSE", yscale="log",
           title="validation loss per output")
    ax.legend()
    fig.tight_layout()
    return fig


def all_figures(history):
    """-> results/<run>/figures/training."""
    return {
        "01_loss_curves": loss_curves(history),
        "02_per_output_loss": per_output_loss(history),
    }
