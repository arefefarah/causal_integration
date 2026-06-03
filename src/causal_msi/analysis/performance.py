r"""Test-gate performance metrics: network outputs vs the analytical observer.

Reports per-output regression (slope, R^2), error distributions, the common-cause
calibration curve, the Kording-style proportion-common-cause vs disparity (and vs
reliability), and generalization to out-of-range disparities/reliabilities.

These functions consume already-computed network outputs and analytical targets
(both ``(N, 5)`` arrays in output order). Pure metric plumbing is implemented;
``TODO(science)`` marks the curve summaries whose exact functional forms the
author defines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score

from causal_msi.generative import ObserverTargets

FloatArray = NDArray[np.float64]

OUTPUT_NAMES = ("mu_vis", "var_vis", "mu_prop", "var_prop", "p_common")


@dataclass(frozen=True)
class OutputRegression:
    """Per-output regression summary (decoded vs analytical)."""

    name: str
    slope: float
    intercept: float
    r2: float


def per_output_regression(pred: FloatArray, target: FloatArray) -> list[OutputRegression]:
    """Regress each network output on its analytical target.

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 5)`` in output order.
    target
        Analytical targets, shape ``(N, 5)``.

    Returns
    -------
    list of OutputRegression
        Slope, intercept, and R^2 per output (5 entries). A well-trained network
        gives slope ~ 1 and high R^2 for every output.
    """
    out: list[OutputRegression] = []
    for i, name in enumerate(OUTPUT_NAMES):
        y, yhat = target[:, i], pred[:, i]
        slope, intercept = np.polyfit(y, yhat, 1)
        out.append(OutputRegression(name, float(slope), float(intercept), float(r2_score(y, yhat))))
    return out


def error_distributions(pred: FloatArray, target: FloatArray) -> dict[str, FloatArray]:
    """Per-output error arrays (decoded - analytical).

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 5)``.
    target
        Analytical targets, shape ``(N, 5)``.

    Returns
    -------
    dict
        Mapping output name -> error array of shape ``(N,)``.
    """
    return {name: pred[:, i] - target[:, i] for i, name in enumerate(OUTPUT_NAMES)}


def pc_calibration(
    pred_pc: FloatArray, target_pc: FloatArray, n_bins: int = 10
) -> tuple[FloatArray, FloatArray]:
    """Binned calibration of the common-cause probability.

    Parameters
    ----------
    pred_pc
        Network ``p(C=1)``, shape ``(N,)``.
    target_pc
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    n_bins
        Number of probability bins.

    Returns
    -------
    tuple of numpy.ndarray
        ``(bin_pred_mean, bin_target_mean)`` per occupied bin. A calibrated network
        lies on the identity.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred_pc, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            xs.append(float(pred_pc[mask].mean()))
            ys.append(float(target_pc[mask].mean()))
    return np.asarray(xs), np.asarray(ys)


def body_frame_disparity(
    targets: ObserverTargets, measurements_x_prop: FloatArray, x_vis_body: FloatArray
) -> FloatArray:
    """Body-frame disparity ``d = x_vis_body - x_prop`` per trial.

    Parameters
    ----------
    targets
        Analytical observer outputs (unused except for shape/context).
    measurements_x_prop
        Proprioceptive measurement, shape ``(N,)`` (deg).
    x_vis_body
        Body-frame visual measurement, shape ``(N,)`` (deg).

    Returns
    -------
    numpy.ndarray
        Disparity per trial, shape ``(N,)`` (deg).
    """
    return x_vis_body - measurements_x_prop


def proportion_common_vs_disparity(
    disparity: FloatArray, p_common: FloatArray, grid: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Average ``p(C=1)`` as a function of binned body-frame disparity.

    Parameters
    ----------
    disparity
        Per-trial body-frame disparity, shape ``(N,)`` (deg).
    p_common
        Per-trial probability (network or analytical), shape ``(N,)``.
    grid
        Bin edges/centres over disparity (deg).

    Returns
    -------
    tuple of numpy.ndarray
        ``(grid_centres, mean_p_common)``. The Kording-style curve: high near zero
        disparity, decreasing with ``|disparity|``.

    Notes
    -----
    TODO(science): define the exact binning/curve summary (e.g. midpoint and
    sharpness of the fused->segregated transition) to compare against human data.
    Tested in the integration/disparity-sweep tests.
    """
    raise NotImplementedError("TODO(science): proportion-common-cause vs disparity summary")


def proportion_common_vs_reliability(
    reliability: FloatArray, p_common: FloatArray, levels: FloatArray
) -> dict[float, float]:
    """Mean ``p(C=1)`` grouped by reliability level.

    Parameters
    ----------
    reliability
        Per-trial reliability proxy (e.g. ``sigma2_vis``), shape ``(N,)`` (deg^2).
    p_common
        Per-trial probability, shape ``(N,)``.
    levels
        Representative reliability levels (deg^2).

    Returns
    -------
    dict
        Mapping reliability level -> mean ``p(C=1)``.

    Notes
    -----
    TODO(science): define how reliability sharpens the common-cause curve (more
    reliable cues should drive a sharper transition).
    """
    raise NotImplementedError("TODO(science): proportion-common-cause vs reliability")


def generalization_report(
    pred: FloatArray, target: FloatArray, in_range_mask: NDArray[np.bool_]
) -> dict[str, list[OutputRegression]]:
    """Compare per-output regression inside vs outside the training range.

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 5)``.
    target
        Analytical targets, shape ``(N, 5)``.
    in_range_mask
        Boolean mask, shape ``(N,)``, True for trials within the training
        disparity/reliability range.

    Returns
    -------
    dict
        ``{"in_range": [...], "out_of_range": [...]}`` regression summaries.
    """
    return {
        "in_range": per_output_regression(pred[in_range_mask], target[in_range_mask]),
        "out_of_range": per_output_regression(pred[~in_range_mask], target[~in_range_mask]),
    }
