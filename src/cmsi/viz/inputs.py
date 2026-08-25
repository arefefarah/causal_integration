"""Input visualisation: what the network is actually being shown.

Worth looking at before trusting any result -- most "the model won't learn"
problems are visible here first (bumps off the edge of the visual field, a dead
push-pull population, gain that doesn't track reliability).
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.data.encoding import encode_groups, gaussian_code, push_pull_code
from cmsi.viz.style import COLORS


def tuning_curves(encoders, enc, n_show=8):
    """Unit tuning: gaussian RF bumps and push-pull lines across position."""
    x = np.linspace(*enc["visual_field"], 400)
    gain = np.ones_like(x)
    rf = gaussian_code(x, encoders["rf_centers"], enc["rf_width"], gain)
    pp = push_pull_code(x, encoders["prop_slope"], encoders["prop_intercept"], gain)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(x, rf[:, ::max(1, rf.shape[1] // n_show)], color=COLORS["visual"], alpha=0.7)
    axes[0].set(title="visual: gaussian receptive fields",
                xlabel="stimulus (deg)", ylabel="rate")
    axes[1].plot(x, pp[:, :n_show], color=COLORS["prop"], alpha=0.7)
    axes[1].set(title="proprioceptive: push-pull units",
                xlabel="stimulus (deg)", ylabel="rate")
    fig.tight_layout()
    return fig


def population_heatmap(d, enc, n=300, seed=0):
    """Trials sorted by measurement: the population bump should track it."""
    order = np.argsort(d["x_vis"])[::max(1, len(d["x_vis"]) // n)]
    trials = {k: v[order] for k, v in d.items()
              if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(d["x_vis"])}
    groups = encode_groups(trials, d["encoders"], enc, np.random.default_rng(seed))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, g) in zip(axes, groups.items()):
        ax.imshow(g, aspect="auto", origin="lower", cmap="viridis")
        ax.grid(False)
        ax.set(title=name, xlabel="unit", ylabel="trials sorted by x_vis")
    fig.tight_layout()
    return fig


def reliability_gain(d, enc, seed=1):
    """Total activity per group should scale with 1 / variance."""
    groups = encode_groups(d, d["encoders"], enc, np.random.default_rng(seed))
    pairs = [("visual_hand", "sig2_vis", "visual"),
             ("prop_hand", "sig2_prop", "prop"),
             ("prop_eye", "sig2_eye", "eye")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (group, sig, color) in zip(axes, pairs):
        ax.scatter(1 / d[sig], groups[group].sum(1), s=3, alpha=0.15,
                   color=COLORS[color])
        ax.set(title=group, xlabel=f"1 / {sig}", ylabel="total activity")
    fig.suptitle("gain scales with reliability")
    fig.tight_layout()
    return fig


def latent_distributions(d):
    """Sanity check on the generative model: disparity, p(C=1), reliabilities."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(d["disparity"], bins=60, color=COLORS["network"])
    axes[0].set(title="body-frame disparity", xlabel="deg")
    axes[1].hist(d["post_c1"], bins=60, color=COLORS["network"])
    axes[1].set(title="analytical p(C=1)", xlabel="probability")
    for sig, color in [("sig2_vis", "visual"), ("sig2_prop", "prop"), ("sig2_eye", "eye")]:
        axes[2].hist(d[sig], bins=40, alpha=0.5, label=sig, color=COLORS[color])
    axes[2].set(title="per-trial noise variances", xlabel="deg^2")
    axes[2].legend()
    fig.tight_layout()
    return fig


def all_figures(d, cfg):
    """Every input figure at once -> {name: Figure}, for results/<run>/figures/inputs."""
    enc = cfg["encoding"]
    return {
        "01_tuning_curves": tuning_curves(d["encoders"], enc),
        "02_population_heatmap": population_heatmap(d, enc),
        "03_reliability_gain": reliability_gain(d, enc),
        "04_latent_distributions": latent_distributions(d),
    }
