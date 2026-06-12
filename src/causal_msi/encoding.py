r"""Population-code encoders: scalar measurements -> network input vectors.

Three input groups are produced and concatenated to form the network input:

- ``visual_hand``: Gaussian receptive-field code of the (retinal) visual
  measurement. RF centres are spread uniformly over the visual field; a fixed RF
  width is shared across units.
- ``prop_hand``: linear monotonic push-pull code of the proprioceptive hand
  measurement (random slopes of both signs, random intercepts).
- ``prop_eye``: linear monotonic push-pull code of the eye-position measurement.

Each group's activity is gain-modulated by the cue reliability (activity scales
with ``1 / variance``, constant ``gain_K``) and corrupted by Poisson trial-to-trial
noise. Encoders are vectorised over trials and unit centres.

Units: positions in degrees; activities are spike-count-like (non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causal_msi.config import EncodingConfig
from causal_msi.generative import LatentBatch, Measurements

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushPullParams:
    """Fixed per-unit parameters for a linear push-pull population.

    Attributes
    ----------
    slopes
        Per-unit slope (1/deg-like), shape ``(n_units,)``; mixed signs realise the
        push-pull (oppositely tuned) structure.
    intercepts
        Per-unit baseline activity, shape ``(n_units,)``.
    """

    slopes: FloatArray
    intercepts: FloatArray


def make_rf_centers(n_units: int, visual_field: tuple[float, float]) -> FloatArray:
    """Place Gaussian receptive-field centres uniformly over the visual field.

    Parameters
    ----------
    n_units
        Number of RF units.
    visual_field
        ``(lo, hi)`` extent of the field (deg).

    Returns
    -------
    numpy.ndarray
        RF centre positions, shape ``(n_units,)`` (deg), evenly spaced (inclusive).
    """
    lo, hi = visual_field
    return np.linspace(lo, hi, n_units)


def make_push_pull_params(
    rng: np.random.Generator,
    n_units: int,
    slope_range: tuple[float, float],
    intercept_range: tuple[float, float],
) -> PushPullParams:
    """Draw random slopes (both signs) and intercepts for a push-pull population.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    n_units
        Number of units.
    slope_range
        ``(lo, hi)`` slope range; spanning negative-to-positive yields push-pull.
    intercept_range
        ``(lo, hi)`` baseline range.

    Returns
    -------
    PushPullParams
        Per-unit slopes and intercepts, each shape ``(n_units,)``.
    """
    slopes = rng.uniform(*slope_range, size=n_units)
    intercepts = rng.uniform(*intercept_range, size=n_units)
    return PushPullParams(slopes=slopes, intercepts=intercepts)


def reliability_gain(variance: FloatArray, gain_K: float) -> FloatArray:
    """Reliability-based gain: activity scales with ``1 / variance``.

    Parameters
    ----------
    variance
        Per-trial cue variance, shape ``(n_trials,)`` (deg^2).
    gain_K
        Gain constant.

    Returns
    -------
    numpy.ndarray
        Multiplicative gain ``gain_K / variance``, shape ``(n_trials,)``.
    """
    return gain_K / variance


def gaussian_rf_code(
    x: FloatArray, centers: FloatArray, width: float, gain: FloatArray
) -> FloatArray:
    """Gaussian receptive-field population code of a scalar measurement.

    Each unit ``j`` responds with a Gaussian bump centred on ``centers[j]``:
    ``r_ij = gain_i * exp(-(x_i - c_j)^2 / (2 * width^2))``.

    Parameters
    ----------
    x
        Scalar measurement per trial, shape ``(n_trials,)`` (deg).
    centers
        RF centre per unit, shape ``(n_units,)`` (deg).
    width
        RF standard deviation (deg).
    gain
        Per-trial gain, shape ``(n_trials,)``.

    Returns
    -------
    numpy.ndarray
        Mean activity (rate), shape ``(n_trials, n_units)``, non-negative.
    """
    z = (x[:, None] - centers[None, :]) / width
    bumps = np.exp(-0.5 * z**2)
    return gain[:, None] * bumps


def push_pull_code(x: FloatArray, params: PushPullParams, gain: FloatArray) -> FloatArray:
    """Linear monotonic push-pull population code of a scalar measurement.

    Each unit is an affine-then-rectified function of the measurement:
    ``r_ij = gain_i * relu(slope_j * x_i + intercept_j)``. Mixed-sign slopes give
    oppositely tuned (push vs pull) units; rectification keeps rates non-negative.

    Parameters
    ----------
    x
        Scalar measurement per trial, shape ``(n_trials,)`` (deg).
    params
        Per-unit slopes and intercepts.
    gain
        Per-trial gain, shape ``(n_trials,)``.

    Returns
    -------
    numpy.ndarray
        Mean activity (rate), shape ``(n_trials, n_units)``, non-negative.
    """
    drive = x[:, None] * params.slopes[None, :] + params.intercepts[None, :]
    rectified = np.clip(drive, 0.0, None)
    return gain[:, None] * rectified


def poisson_noise(rng: np.random.Generator, rates: FloatArray) -> FloatArray:
    """Apply Poisson trial-to-trial noise to non-negative mean rates.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    rates
        Mean activity (Poisson lambda), shape ``(n_trials, n_units)``,
        non-negative.

    Returns
    -------
    numpy.ndarray
        Poisson spike counts, same shape, returned as float for downstream tensors.
        The sample mean over many trials converges to ``rates``.
    """
    return rng.poisson(np.clip(rates, 0.0, None)).astype(np.float64)


@dataclass(frozen=True)
class Encoders:
    """Fixed (seeded) encoder parameters shared across a dataset.

    Holding these constant across train/val/test (and across the dataset) keeps the
    population's tuning fixed while only the inputs vary trial-to-trial.
    """

    rf_centers: FloatArray
    prop_hand: PushPullParams
    prop_eye: PushPullParams

    @classmethod
    def build(cls, rng: np.random.Generator, cfg: EncodingConfig) -> Encoders:
        """Construct encoder parameters from the encoding config.

        Parameters
        ----------
        rng
            Seeded numpy generator.
        cfg
            Encoding configuration.

        Returns
        -------
        Encoders
            Fixed RF centres and push-pull parameters for both linear groups.
        """
        return cls(
            rf_centers=make_rf_centers(cfg.n_units_visual_hand, cfg.visual_field),
            prop_hand=make_push_pull_params(
                rng, cfg.n_units_prop_hand, cfg.slope_range, cfg.intercept_range
            ),
            prop_eye=make_push_pull_params(
                rng, cfg.n_units_prop_eye, cfg.slope_range, cfg.intercept_range
            ),
        )


def encode_groups(
    rng: np.random.Generator,
    meas: Measurements,
    latents: LatentBatch,
    cfg: EncodingConfig,
    encoders: Encoders,
) -> dict[str, FloatArray]:
    """Encode the three input groups as population-code matrices.

    Parameters
    ----------
    rng
        Seeded numpy generator (used for Poisson noise).
    meas
        Noisy scalar measurements.
    latents
        Latent batch (used only for the per-trial reliabilities).
    cfg
        Encoding configuration.
    encoders
        Fixed encoder parameters.

    Returns
    -------
    dict
        Mapping ``{"visual_hand": (N, n_vis), "prop_hand": (N, n_prop),
        "prop_eye": (N, n_eye)}`` of population activities.
    """
    g_vis = reliability_gain(latents.sigma2_vis, cfg.gain_K)
    g_prop = reliability_gain(latents.sigma2_prop, cfg.gain_K)
    g_eye = reliability_gain(latents.sigma2_eye, cfg.gain_K)

    visual_hand = gaussian_rf_code(meas.x_vis, encoders.rf_centers, cfg.rf_width, g_vis)
    prop_hand = push_pull_code(meas.x_prop, encoders.prop_hand, g_prop)
    prop_eye = push_pull_code(meas.x_eye, encoders.prop_eye, g_eye)

    if cfg.poisson_noise:
        visual_hand = poisson_noise(rng, visual_hand)
        prop_hand = poisson_noise(rng, prop_hand)
        prop_eye = poisson_noise(rng, prop_eye)

    return {"visual_hand": visual_hand, "prop_hand": prop_hand, "prop_eye": prop_eye}


def assemble_inputs(
    rng: np.random.Generator,
    meas: Measurements,
    latents: LatentBatch,
    cfg: EncodingConfig,
    encoders: Encoders | None = None,
) -> FloatArray:
    """Encode and concatenate the three groups into the network input matrix.

    The network receives ONLY the three population-coded sensory groups -- never
    ``p_common``, disparity, the true sources, eye position, or ``C``.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    meas
        Noisy scalar measurements.
    latents
        Latent batch (per-trial reliabilities).
    cfg
        Encoding configuration.
    encoders
        Optional pre-built encoders; if None, fresh ones are built from ``rng``.

    Returns
    -------
    numpy.ndarray
        Network input ``X``, shape ``(N, input_dim)`` in group order
        ``[visual_hand, prop_hand, prop_eye]``.
    """
    if encoders is None:
        encoders = Encoders.build(rng, cfg)
    groups = encode_groups(rng, meas, latents, cfg, encoders)
    parts = [groups["visual_hand"], groups["prop_hand"], groups["prop_eye"]]
    return np.concatenate(parts, axis=1)
