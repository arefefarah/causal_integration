"""Plot helpers in the prior paper's figure style.

Each helper takes pre-computed arrays and an optional matplotlib ``Axes`` and
returns the ``Axes`` so figures can be composed. None of these compute science --
they only render arrays produced elsewhere.

Units: positions in degrees; variances in deg^2.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ensure_ax(ax: Axes | None) -> Axes:
    """Return ``ax`` or create a fresh one."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    return ax


def tuning_curves(
    stimulus: FloatArray,
    responses: FloatArray,
    ax: Axes | None = None,
    title: str = "",
    max_units: int | None = None,
) -> Axes:
    """Plot population tuning curves: response vs swept stimulus, one line per unit.

    Parameters
    ----------
    stimulus
        Swept stimulus values, shape ``(n_stim,)`` (deg).
    responses
        Responses, shape ``(n_stim, n_units)``.
    ax
        Optional axes.
    title
        Subplot title.
    max_units
        If set, plot only the first ``max_units`` units (keeps dense codes legible).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    n_units = responses.shape[1]
    if max_units is None or max_units >= n_units:
        unit_idx = np.arange(n_units)
    else:
        # Sample units EVENLY across the population so the displayed subset spans
        # the full range of preferred values (not just the first max_units units).
        unit_idx = np.unique(np.linspace(0, n_units - 1, max_units).round().astype(int))
    for j in unit_idx:
        ax.plot(stimulus, responses[:, j], lw=0.9, alpha=0.6)
    ax.set_xlabel("stimulus (deg)")
    ax.set_ylabel("mean response")
    ax.set_title(title)
    return ax


def population_vectors(
    stimulus_values: list[float],
    activity_by_stimulus: list[FloatArray],
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """Overlay single-trial population vectors for a few stimulus values.

    Parameters
    ----------
    stimulus_values
        The stimulus value behind each vector (for the legend), length ``k``.
    activity_by_stimulus
        List of ``k`` activity vectors, each shape ``(n_units,)``.
    ax
        Optional axes.
    title
        Subplot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    units = np.arange(activity_by_stimulus[0].shape[0])
    for val, act in zip(stimulus_values, activity_by_stimulus, strict=True):
        ax.plot(units, act, marker="o", ms=2.5, lw=0.9, label=f"stim={val:.0f} deg")
    ax.set_xlabel("unit index")
    ax.set_ylabel("activity")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return ax


def input_heatmap(
    matrix: FloatArray,
    ax: Axes | None = None,
    title: str = "",
    xlabel: str = "unit index",
    ylabel: str = "trial (sorted by stimulus)",
) -> Axes:
    """Heatmap of a population code over trials (rows) and units (columns).

    Parameters
    ----------
    matrix
        Activity matrix, shape ``(n_trials, n_units)`` (rows sorted by the relevant
        scalar by the caller).
    ax
        Optional axes.
    title, xlabel, ylabel
        Labels.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def loss_curves(
    train_total: FloatArray,
    val_total: FloatArray,
    ax: Axes | None = None,
    val_components: dict[str, FloatArray] | None = None,
) -> Axes:
    """Plot training/validation loss over epochs (log scale).

    Parameters
    ----------
    train_total
        Per-epoch training total loss, shape ``(n_epochs,)``.
    val_total
        Per-epoch validation total loss, shape ``(n_epochs,)``.
    ax
        Optional axes.
    val_components
        Optional mapping of component name -> per-epoch validation loss, drawn as
        dashed lines (e.g. ``est``, ``var``, ``pc``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    epochs = np.arange(1, len(train_total) + 1)
    ax.plot(epochs, train_total, label="train total", lw=1.5)
    ax.plot(epochs, val_total, label="val total", lw=1.5)
    if val_components:
        for name, vals in val_components.items():
            ax.plot(epochs, vals, "--", lw=1.0, alpha=0.7, label=f"val {name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return ax


def bar_metrics(
    names: list[str],
    values: list[float],
    ax: Axes | None = None,
    ylabel: str = "R^2",
    title: str = "",
    ylim: tuple[float, float] | None = None,
) -> Axes:
    """Bar chart of per-output metrics (e.g. R^2 per output).

    Parameters
    ----------
    names
        Bar labels.
    values
        Bar heights.
    ax
        Optional axes.
    ylabel, title
        Labels.
    ylim
        Optional y-axis limits.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    bars = ax.bar(names, values, color="steelblue")
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax


def binned_curve(
    x_centres: FloatArray,
    series: dict[str, FloatArray],
    ax: Axes | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> Axes:
    """Plot one or more binned curves sharing an x-axis.

    Parameters
    ----------
    x_centres
        Bin centres, shape ``(n_bins,)``.
    series
        Mapping label -> y-values of shape ``(n_bins,)``.
    ax
        Optional axes.
    xlabel, ylabel, title
        Labels.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    for label, ys in series.items():
        ax.plot(x_centres, ys, marker="o", ms=4, lw=1.3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def regression_scatter(
    analytical: FloatArray,
    decoded: FloatArray,
    ax: Axes | None = None,
    label: str = "",
    identity: bool = True,
) -> Axes:
    """Scatter decoded-vs-analytical values with an identity reference line.

    Parameters
    ----------
    analytical
        Ground-truth analytical values, shape ``(N,)``.
    decoded
        Network/decoded values, shape ``(N,)``.
    ax
        Optional axes.
    label
        Point-series label.
    identity
        Whether to draw the ``y = x`` line.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    ax.scatter(analytical, decoded, s=6, alpha=0.3, label=label or None)
    if identity:
        lo = float(min(analytical.min(), decoded.min()))
        hi = float(max(analytical.max(), decoded.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("analytical")
    ax.set_ylabel("decoded")
    ax.set_aspect("equal", adjustable="box")
    return ax


def error_histogram(errors: FloatArray, ax: Axes | None = None, bins: int = 50) -> Axes:
    """Histogram of estimation errors (decoded - analytical).

    Parameters
    ----------
    errors
        Error values, shape ``(N,)`` (deg).
    ax
        Optional axes.
    bins
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    ax.hist(errors, bins=bins, alpha=0.8)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("error (deg)")
    ax.set_ylabel("count")
    return ax


def calibration_curve(
    predicted_prob: FloatArray,
    target_prob: FloatArray,
    ax: Axes | None = None,
    n_bins: int = 10,
) -> Axes:
    """Reliability/calibration curve for the common-cause probability.

    Parameters
    ----------
    predicted_prob
        Network ``p(C=1)``, shape ``(N,)``.
    target_prob
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    ax
        Optional axes.
    n_bins
        Number of probability bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(predicted_prob, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            xs.append(float(predicted_prob[mask].mean()))
            ys.append(float(target_prob[mask].mean()))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("predicted p(C=1)")
    ax.set_ylabel("analytical p(C=1)")
    ax.set_aspect("equal", adjustable="box")
    return ax


def fusion_weight_curve(
    p_common: FloatArray,
    fusion_weight: FloatArray,
    ax: Axes | None = None,
) -> Axes:
    """Empirical fusion weight vs analytical ``p(C=1)`` (identity = optimal).

    Parameters
    ----------
    p_common
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    fusion_weight
        Empirical weight on the fused solution, shape ``(N,)`` in ``[0, 1]``.
    ax
        Optional axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    order = np.argsort(p_common)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="optimal")
    ax.plot(p_common[order], fusion_weight[order], ".", alpha=0.4, label="empirical")
    ax.set_xlabel("analytical p(C=1)")
    ax.set_ylabel("fusion weight")
    ax.legend()
    return ax


def gain_index_histogram(gain_index: FloatArray, ax: Axes | None = None, bins: int = 40) -> Axes:
    """Histogram of per-unit gain indices.

    Parameters
    ----------
    gain_index
        Per-unit gain-index values, shape ``(n_units,)``.
    ax
        Optional axes.
    bins
        Number of bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    ax.hist(gain_index, bins=bins, alpha=0.8)
    ax.set_xlabel("gain index")
    ax.set_ylabel("count")
    return ax


def rf_map(
    centers: FloatArray,
    responses: FloatArray,
    ax: Axes | None = None,
) -> Axes:
    """Plot receptive-field tuning curves (response vs stimulus per unit).

    Parameters
    ----------
    centers
        Stimulus values, shape ``(n_stimuli,)`` (deg).
    responses
        Responses, shape ``(n_stimuli, n_units)``.
    ax
        Optional axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _ensure_ax(ax)
    for j in range(responses.shape[1]):
        ax.plot(centers, responses[:, j], lw=0.8, alpha=0.6)
    ax.set_xlabel("stimulus (deg)")
    ax.set_ylabel("response")
    return ax
