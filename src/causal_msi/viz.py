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
