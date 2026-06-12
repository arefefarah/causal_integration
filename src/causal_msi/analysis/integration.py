r"""The integration analysis (explicit multisensory integration).

Under the Kording (2007) output design the network is trained to output the
model-averaged *optimal* estimates directly (Eqs. 9/10), so integration is now
EXPLICIT in the read-out. Three estimates of the multisensory location are
compared:

1. ANALYTICAL optimal estimate (reference) -- Eq. 9, recomputed from the
   analytical ``p(C=1)``, fused (Eq. 12), and segregated (Eq. 11) values.
2. NETWORK estimate -- read directly from the network's output column.
3. POPULATION-DECODED estimate -- a separate linear decoder from MSL to the
   analytical optimal estimate (what the population carries, independent of the
   read-out head).

Comparisons ask whether the OUTPUT matches the optimal estimate and whether the
POPULATION already carries it internally. The ``fusion_weight_curve`` recovers the
*implied* common-cause weight behind the network's estimate (decomposing it into
the C=2 segregated vs C=1 fused references) and compares it to the analytical
``p(C=1)`` -- the network never sees ``p(C=1)`` directly.

All estimator formulae are implemented; decoder fitting and curve-binning plumbing
is implemented. Units: deg / deg^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causal_msi.analysis._decoders import DecodeResult, fit_linear_decoder
from causal_msi.config import DecoderConfig
from causal_msi.generative import ObserverTargets

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# 1. Analytical reference
# --------------------------------------------------------------------------- #
def analytical_model_averaged_estimate(targets: ObserverTargets) -> FloatArray:
    """Ground-truth model-averaged location estimate from analytical quantities.

    Parameters
    ----------
    targets
        Analytical observer outputs (segregated means/vars, fused mean, p(C=1)).

    Returns
    -------
    numpy.ndarray
        Model-averaged visual location estimate per trial, shape ``(N,)`` (deg).

    Notes
    -----
    Implements Bayesian model averaging of the visual estimate (Kording Eq. 9)
        ``s_hat = p(C=1) * fused_mu + (1 - p(C=1)) * seg_vis_mu``.
    This equals ``targets.mu_vis`` (now a direct network target); it is recomputed
    here from the stored components as the analytical reference. Tested in
    ``tests/test_integration.py::test_model_averaging_endpoints``.
    """
    pc = targets.p_common
    return pc * targets.fused_mu + (1.0 - pc) * targets.seg_vis_mu


# --------------------------------------------------------------------------- #
# 2. Network-reconstructed estimate
# --------------------------------------------------------------------------- #
def network_optimal_estimate(outputs: FloatArray, modality: str = "vis") -> FloatArray:
    """Read the network's optimal position estimate for one modality.

    Under the Kording (2007) design the network is trained to output the
    model-averaged optimal estimate **directly** (Eqs. 9/10), so no reconstruction
    is needed -- the estimate is simply the corresponding output column. (This
    replaces the older 5-output design, where integration was implicit and the
    estimate had to be reconstructed from segregated posteriors + ``p_common``.)

    Parameters
    ----------
    outputs
        Network outputs, shape ``(N, 4)`` =
        ``[mu_vis, var_vis, mu_prop, var_prop]``.
    modality
        ``"vis"`` (column 0) or ``"prop"`` (column 2).

    Returns
    -------
    numpy.ndarray
        The network's optimal estimate per trial, shape ``(N,)`` (deg).
    """
    if modality == "vis":
        return outputs[:, 0]
    if modality == "prop":
        return outputs[:, 2]
    raise ValueError(f"modality must be 'vis' or 'prop', got {modality!r}")


# --------------------------------------------------------------------------- #
# 3. Population-decoded estimate
# --------------------------------------------------------------------------- #
def population_decoded_estimate(
    msl_activations: FloatArray,
    analytical_estimate: FloatArray,
    cfg: DecoderConfig,
    seed: int = 0,
) -> DecodeResult:
    """Decode the analytical model-averaged estimate from MSL activations.

    A SEPARATE linear decoder (independent of the read-out head) reveals what the
    population represents internally.

    Parameters
    ----------
    msl_activations
        MSL activations, shape ``(N, h2)``.
    analytical_estimate
        Analytical model-averaged estimate, shape ``(N,)`` (deg).
    cfg
        Decoder configuration.
    seed
        Split seed.

    Returns
    -------
    DecodeResult
        Held-out decoding of the estimate from the population.
    """
    return fit_linear_decoder(msl_activations, analytical_estimate, cfg, seed)


# --------------------------------------------------------------------------- #
# 4. Comparisons
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegrationComparison:
    """Bias/scatter summary comparing two location estimates."""

    slope: float
    r2: float
    rmse: float


def compare_estimates(a: FloatArray, b: FloatArray) -> IntegrationComparison:
    """Regress estimate ``b`` on estimate ``a`` and report slope/R^2/RMSE.

    Parameters
    ----------
    a
        Reference estimate, shape ``(N,)`` (deg).
    b
        Comparison estimate, shape ``(N,)`` (deg).

    Returns
    -------
    IntegrationComparison
        Slope, R^2, and RMSE of ``b`` against ``a``.
    """
    from sklearn.metrics import r2_score

    slope = float(np.polyfit(a, b, 1)[0])
    r2 = float(r2_score(a, b))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return IntegrationComparison(slope=slope, r2=r2, rmse=rmse)


# --------------------------------------------------------------------------- #
# 5. Fusion-weight curve
# --------------------------------------------------------------------------- #
def fusion_weight_curve(
    segregated: FloatArray,
    fused: FloatArray,
    estimate: FloatArray,
) -> FloatArray:
    """Empirical weight on the fused solution implied by an estimate.

    Parameters
    ----------
    segregated
        Segregated (single-cue) estimate per trial, shape ``(N,)`` (deg).
    fused
        Forced-fusion estimate per trial, shape ``(N,)`` (deg).
    estimate
        The estimate to decompose (reconstructed or decoded), shape ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Empirical fusion weight ``w`` per trial in ``[0, 1]``, shape ``(N,)``.

    Notes
    -----
    Solves ``estimate = w * fused + (1 - w) * segregated`` for ``w``
        ``w = (estimate - segregated) / (fused - segregated)``.
    The degenerate ``fused == segregated`` case (no leverage to estimate ``w``) is
    returned as NaN. Optimal averaging predicts ``w == p(C=1)`` (identity line vs
    analytical ``p(C=1)``); deviations (over-fusion at intermediate disparity,
    saturation, hysteresis) are read off the curve. Tested in
    ``tests/test_integration.py::test_fusion_weight_in_unit_interval``.
    """
    eps = 1e-9
    denom = fused - segregated
    safe = np.abs(denom) > eps
    w = np.full(estimate.shape, np.nan, dtype=float)
    w[safe] = (estimate[safe] - segregated[safe]) / denom[safe]
    return w


# --------------------------------------------------------------------------- #
# 6. Disparity sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepCurve:
    """A fused->segregated transition curve."""

    disparity: FloatArray
    fusion_weight: FloatArray
    midpoint: float
    sharpness: float


def _bin_means(x: FloatArray, y: FloatArray, grid: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean of ``y`` grouped by nearest ``grid`` centre of ``x`` (occupied bins)."""
    grid = np.asarray(grid, dtype=float)
    valid = ~np.isnan(y)
    x, y = x[valid], y[valid]
    idx = np.abs(x[:, None] - grid[None, :]).argmin(axis=1)
    centres, means = [], []
    for b in range(len(grid)):
        m = idx == b
        if m.any():
            centres.append(float(grid[b]))
            means.append(float(y[m].mean()))
    return np.asarray(centres), np.asarray(means)


def _fit_transition(abs_disparity: FloatArray, fusion_weight: FloatArray) -> tuple[float, float]:
    """Fit ``w = 1 / (1 + exp(k * (|d| - d0)))`` -> return ``(midpoint d0, sharpness k)``.

    The fusion weight falls from ~1 at zero disparity to ~0 at large ``|disparity|``;
    ``d0`` is the disparity at which it crosses 0.5 and ``k`` its steepness. Returns
    ``(nan, nan)`` if the non-linear fit does not converge.
    """
    from scipy.optimize import curve_fit

    valid = ~np.isnan(fusion_weight)
    d, w = abs_disparity[valid], fusion_weight[valid]
    if d.size < 4 or np.ptp(d) == 0:
        return float("nan"), float("nan")

    def logistic(dd: FloatArray, d0: float, k: float) -> FloatArray:
        return 1.0 / (1.0 + np.exp(k * (dd - d0)))

    try:
        popt, _ = curve_fit(logistic, d, w, p0=[float(np.median(d)), 1.0], maxfev=10000)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")


def disparity_sweep(
    disparity: FloatArray,
    fusion_weight: FloatArray,
    grid: FloatArray,
) -> SweepCurve:
    """Trace the fused->segregated transition across a disparity sweep.

    Parameters
    ----------
    disparity
        Per-trial body-frame disparity, shape ``(N,)`` (deg). Reliabilities should
        be held fixed by the caller.
    fusion_weight
        Per-trial empirical fusion weight, shape ``(N,)``.
    grid
        Disparity bin centres (deg).

    Returns
    -------
    SweepCurve
        Binned (signed) fusion-weight curve plus the midpoint and sharpness of the
        transition fitted over ``|disparity|``.

    Notes
    -----
    Bins ``fusion_weight`` over signed ``disparity`` for the curve, then fits a
    logistic of the weight against ``|disparity|`` to extract the midpoint
    (disparity at ``w = 0.5``) and sharpness (slope) -- the Kording-style
    psychometric transition.
    """
    centres, mean_w = _bin_means(disparity, fusion_weight, grid)
    midpoint, sharpness = _fit_transition(np.abs(disparity), fusion_weight)
    return SweepCurve(
        disparity=centres, fusion_weight=mean_w, midpoint=midpoint, sharpness=sharpness
    )


# --------------------------------------------------------------------------- #
# 7. Reliability dependence
# --------------------------------------------------------------------------- #
def reliability_dependence(
    disparity: FloatArray,
    fusion_weight: FloatArray,
    reliability: FloatArray,
    levels: FloatArray,
    grid: FloatArray,
) -> dict[float, SweepCurve]:
    """Repeat the disparity sweep at several reliability levels.

    Parameters
    ----------
    disparity
        Per-trial body-frame disparity, shape ``(N,)`` (deg).
    fusion_weight
        Per-trial empirical fusion weight, shape ``(N,)``.
    reliability
        Per-trial reliability proxy (e.g. ``sigma2_vis``), shape ``(N,)`` (deg^2).
    levels
        Reliability levels to group by (deg^2).
    grid
        Disparity bin centres (deg).

    Returns
    -------
    dict
        Mapping reliability level -> :class:`SweepCurve`.

    Notes
    -----
    Assigns each trial to the nearest reliability level and runs
    :func:`disparity_sweep` per group. The hypothesis: the fusion->segregation
    crossover shifts with reliability -- less reliable cues tolerate larger
    disparity before segregating (midpoint increases as reliability drops).
    """
    levels = np.asarray(levels, dtype=float)
    nearest = np.abs(reliability[:, None] - levels[None, :]).argmin(axis=1)
    out: dict[float, SweepCurve] = {}
    for b, level in enumerate(levels):
        mask = nearest == b
        if mask.any():
            out[float(level)] = disparity_sweep(disparity[mask], fusion_weight[mask], grid)
    return out


# --------------------------------------------------------------------------- #
# 8. Decision-strategy fit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyFit:
    """Goodness-of-fit of candidate decision strategies."""

    model_averaging: float
    model_selection: float
    probability_matching: float

    def best(self) -> str:
        """Return the name of the best-fitting strategy (lowest error)."""
        scores = {
            "model_averaging": self.model_averaging,
            "model_selection": self.model_selection,
            "probability_matching": self.probability_matching,
        }
        return min(scores, key=scores.get)  # type: ignore[arg-type]


def decision_strategy_fit(
    population_estimate: FloatArray,
    p_common: FloatArray,
    fused: FloatArray,
    segregated: FloatArray,
    rng: np.random.Generator | None = None,
) -> StrategyFit:
    """Fit model averaging vs selection vs probability matching to the population.

    Parameters
    ----------
    population_estimate
        Population-decoded estimate, shape ``(N,)`` (deg).
    p_common
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    fused, segregated
        Fused and segregated estimates, each shape ``(N,)`` (deg).
    rng
        Generator for the stochastic probability-matching prediction.

    Returns
    -------
    StrategyFit
        Per-strategy fit error; ``.best()`` names the implemented strategy.

    Notes
    -----
    Builds the three candidate predictions and scores each by RMSE against the
    population estimate (lower = better; ``.best()`` names the winner):
        - MODEL AVERAGING:   ``p_common * fused + (1 - p_common) * segregated``
        - MODEL SELECTION:   ``fused if p_common > 0.5 else segregated`` (hard)
        - PROBABILITY MATCHING: sample ``fused`` w.p. ``p_common`` else ``segregated``
    To test whether the loss (BCE vs MSE on p(C=1)) steers the strategy, fit on
    population estimates decoded from networks trained under each loss and compare.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def _rmse(prediction: FloatArray) -> float:
        return float(np.sqrt(np.mean((prediction - population_estimate) ** 2)))

    averaging = p_common * fused + (1.0 - p_common) * segregated
    selection = np.where(p_common > 0.5, fused, segregated)
    matched = np.where(rng.random(p_common.shape) < p_common, fused, segregated)

    return StrategyFit(
        model_averaging=_rmse(averaging),
        model_selection=_rmse(selection),
        probability_matching=_rmse(matched),
    )
