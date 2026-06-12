r"""Recover the common-cause posterior implicitly coded by the network.

Although ``p(C=1 | x)`` is **not** a network output under the Kording (2007) design,
the network's optimal estimates are a ``p``-weighted blend of the common-cause
(fused, Eq. 12) and separate-cause (segregated, Eq. 11) endpoints:

    mu_i_net = p * fused_mu + (1 - p) * seg_i_mu        (Eqs. 9/10)

Given the analytical endpoints (which the observer computes per trial) we can
*invert* the network's two mean outputs to the single mixing weight ``p`` it
effectively applied -- the **implied common-cause posterior**. We then ask:

1. Does the implied ``p`` match the analytical ``p(C=1 | x)`` (is it implicitly
   coded)?  -> :func:`compare_common_cause`.
2. Does it discriminate trials that truly share a cause (``C == 1``) from
   independent-source trials (``C == 2``)?  -> :func:`discrimination_auc`,
   :func:`pc_by_true_cause`.

Units: degrees / deg^2; ``p`` is a probability in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score, roc_auc_score

from causal_msi.generative import ObserverTargets

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def inferred_common_cause_posterior(
    outputs: FloatArray, targets: ObserverTargets, eps: float = 1e-6
) -> FloatArray:
    """Infer the common-cause weight ``p`` implied by the network's mean outputs.

    Jointly inverts the two model-averaging equations (Eqs. 9 and 10) for the
    single weight ``p`` that best explains the network's visual and proprioceptive
    mean estimates, given the analytical fused (Eq. 12) and segregated (Eq. 11)
    endpoints. The least-squares solution over both modalities is

        ``p = (a_vis*d_vis + a_prop*d_prop) / (a_vis**2 + a_prop**2)``

    with ``a_i = fused_mu - seg_i_mu`` and ``d_i = mu_i_net - seg_i_mu``. Using both
    modalities stabilises the estimate; ``p`` is clipped to ``[0, 1]``. Trials with
    a near-degenerate denominator (fused coincides with both segregated estimates,
    i.e. negligible disparity, so ``p`` carries no leverage) are returned as NaN.

    Parameters
    ----------
    outputs
        Network outputs, shape ``(N, 4)`` = ``[mu_vis, var_vis, mu_prop, var_prop]``.
    targets
        Analytical observer outputs, providing the per-trial endpoints
        ``fused_mu`` (Eq. 12) and ``seg_vis_mu`` / ``seg_prop_mu`` (Eq. 11).
    eps
        Degenerate-denominator threshold.

    Returns
    -------
    numpy.ndarray
        Implied common-cause posterior per trial, shape ``(N,)``, in ``[0, 1]``
        (NaN where undefined).
    """
    fused = targets.fused_mu
    a_vis = fused - targets.seg_vis_mu
    a_prop = fused - targets.seg_prop_mu
    d_vis = outputs[:, 0] - targets.seg_vis_mu
    d_prop = outputs[:, 2] - targets.seg_prop_mu

    denom = a_vis**2 + a_prop**2
    safe = denom > eps
    p = np.full(outputs.shape[0], np.nan, dtype=float)
    p[safe] = (a_vis[safe] * d_vis[safe] + a_prop[safe] * d_prop[safe]) / denom[safe]
    p[safe] = np.clip(p[safe], 0.0, 1.0)
    return p


def compare_common_cause(network_pc: FloatArray, analytical_pc: FloatArray) -> dict[str, float]:
    """Compare the network-implied ``p(C=1)`` to the analytical ``p(C=1 | x)``.

    Parameters
    ----------
    network_pc
        Implied common-cause posterior from :func:`inferred_common_cause_posterior`,
        shape ``(N,)`` (may contain NaN).
    analytical_pc
        Analytical ``p(C=1 | x)`` (Eq. 2), shape ``(N,)``.

    Returns
    -------
    dict
        ``{"r2", "slope", "rmse", "pearson_r", "n"}`` computed on the non-NaN
        trials. ``r2``/``slope``/``pearson_r`` near 1 mean the network implicitly
        codes the analytical posterior.
    """
    m = np.isfinite(network_pc) & np.isfinite(analytical_pc)
    x, y = analytical_pc[m], network_pc[m]
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return {
        "r2": float(r2_score(x, y)),
        "slope": slope,
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "n": int(m.sum()),
    }


def discrimination_auc(score: FloatArray, true_c: IntArray) -> float:
    """ROC-AUC for discriminating true common-cause (``C==1``) trials from ``C==2``.

    Parameters
    ----------
    score
        Per-trial score (e.g. implied or analytical ``p(C=1)``), shape ``(N,)``
        (may contain NaN).
    true_c
        True causal structure per trial, shape ``(N,)`` with values in ``{1, 2}``.

    Returns
    -------
    float
        Area under the ROC curve for predicting ``C == 1`` from ``score``. 0.5 =
        chance, 1.0 = perfect separation. NaN if a class is absent.
    """
    m = np.isfinite(score)
    y_true = (true_c[m] == 1).astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, score[m]))


def pc_by_true_cause(pc: FloatArray, true_c: IntArray) -> dict[str, float]:
    """Mean implied/analytical ``p(C=1)`` split by the true causal structure.

    Parameters
    ----------
    pc
        Per-trial common-cause probability, shape ``(N,)`` (may contain NaN).
    true_c
        True causal structure per trial, shape ``(N,)`` in ``{1, 2}``.

    Returns
    -------
    dict
        ``{"mean_pc_true_common", "mean_pc_true_separate", "separation"}`` where
        ``separation`` is the difference (positive = correctly higher on common
        cause trials).
    """
    m = np.isfinite(pc)
    pc_m, c_m = pc[m], true_c[m]
    mean_common = float(pc_m[c_m == 1].mean()) if (c_m == 1).any() else float("nan")
    mean_separate = float(pc_m[c_m == 2].mean()) if (c_m == 2).any() else float("nan")
    return {
        "mean_pc_true_common": mean_common,
        "mean_pc_true_separate": mean_separate,
        "separation": mean_common - mean_separate,
    }
