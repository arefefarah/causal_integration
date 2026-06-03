r"""The sharper integration analysis (implicit multisensory integration).

Goal: characterize the network's implicit integration WITHOUT the circular
shortcut of checking the network's own reconstructed estimate against its own
``p(C=1)``. Three independent estimates of the multisensory location are compared:

1. ANALYTICAL model-averaged estimate (reference) -- from the analytical
   ``p(C=1)``, fused, and segregated values.
2. NETWORK-RECONSTRUCTED estimate -- backed out of the network's 5 outputs.
3. POPULATION-DECODED estimate -- a separate linear decoder from MSL to the
   analytical model-averaged estimate (what the population carries, independent of
   the read-out head).

Comparisons then ask whether the OUTPUT implements optimal averaging and whether
the POPULATION already carries the fused/averaged estimate internally.

All estimator formulae are ``TODO(science)`` (stated in the docstrings); decoder
fitting and curve-binning plumbing is implemented. Units: deg / deg^2.
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
    TODO(science): Bayesian model averaging of the visual estimate
        ``s_hat = p(C=1) * fused_mu + (1 - p(C=1)) * mu_vis_segregated``.
    (Analogously for proprioception with ``mu_prop``.) This is the reference the
    network is implicitly compared against. Tested in
    ``tests/test_integration.py::test_model_averaging_endpoints``.
    """
    raise NotImplementedError("TODO(science): analytical model-averaged estimate")


# --------------------------------------------------------------------------- #
# 2. Network-reconstructed estimate
# --------------------------------------------------------------------------- #
def network_reconstructed_estimate(outputs: FloatArray, mu0: float, sigma0_sq: float) -> FloatArray:
    """Reconstruct the model-averaged estimate from the network's 5 outputs.

    Parameters
    ----------
    outputs
        Network outputs, shape ``(N, 5)`` =
        ``[mu_vis, var_vis, mu_prop, var_prop, p_common]``.
    mu0, sigma0_sq
        Prior mean (deg) and variance (deg^2) used to back out cue likelihoods.

    Returns
    -------
    numpy.ndarray
        Reconstructed model-averaged estimate per trial, shape ``(N,)`` (deg).

    Notes
    -----
    TODO(science): (a) invert each segregated posterior to recover the cue
    likelihood by removing the prior
        ``1/var_like = 1/var_seg - 1/sigma0_sq``;
        ``x_like = var_like * (mu_seg/var_seg - mu0/sigma0_sq)``;
    (b) recombine the two likelihoods + prior into the fused estimate;
    (c) model-average fused vs segregated using the network's ``p_common``. The
    network is NOT given the fused output, so this reconstruction is the only way
    to read its implicit integration. Tested in ``tests/test_integration.py``.
    """
    raise NotImplementedError("TODO(science): reconstruct fused / model-averaged from outputs")


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
    TODO(science): solve ``estimate = w * fused + (1 - w) * segregated`` for ``w``
        ``w = (estimate - segregated) / (fused - segregated)``
    (guard the degenerate ``fused == segregated`` case). Optimal averaging predicts
    ``w == p(C=1)`` (identity line vs analytical ``p(C=1)``); report deviations
    (over-fusion at intermediate disparity, saturation, hysteresis). Tested in
    ``tests/test_integration.py::test_fusion_weight_in_unit_interval``.
    """
    raise NotImplementedError("TODO(science): empirical fusion weight")


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
        Binned fusion-weight curve plus its midpoint and sharpness.

    Notes
    -----
    TODO(science): bin ``fusion_weight`` over ``disparity`` and fit the transition
    (e.g. logistic) to extract the midpoint (disparity at ``w = 0.5``) and the
    sharpness (slope). Compare to Kording-style human psychometric curves.
    """
    raise NotImplementedError("TODO(science): disparity-sweep transition fit")


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
    TODO(science): test whether the fusion->segregation crossover shifts with
    reliability -- less reliable cues should tolerate larger disparity before
    segregating (midpoint increases as reliability drops).
    """
    raise NotImplementedError("TODO(science): reliability-dependent crossover")


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
    TODO(science): build the three predictions and score each against the
    population estimate:
        - MODEL AVERAGING:   ``p_common * fused + (1 - p_common) * segregated``
        - MODEL SELECTION:   ``fused if p_common > 0.5 else segregated`` (hard)
        - PROBABILITY MATCHING: sample ``fused`` w.p. ``p_common`` else ``segregated``
    Report which the trained network implements and expose a hook to test whether
    the loss (BCE vs MSE on p(C=1)) steers the strategy.
    """
    raise NotImplementedError("TODO(science): decision-strategy model comparison")
