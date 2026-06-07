r"""Generative model and analytical (Bayesian) observer.

Task: visuo-proprioceptive realignment across eye positions.

Generative model
----------------
- Spatial prior:        ``source ~ Normal(mu0, sigma0_sq)``.
- Causal latent:        ``C ~ Bernoulli(p_common)``. If ``C == 1`` there is one
  shared source ``s`` driving both cues; if ``C == 2`` there are two independent
  sources ``s1`` (vision) and ``s2`` (proprioception), each drawn from the prior.
- Eye position:         ``e`` drawn from its own Normal distribution.
- Per-trial reliabilities (noise variances) ``sigma2_vis``, ``sigma2_prop``,
  ``sigma2_eye`` drawn from configured ranges.
- Reference-frame transform: vision is encoded in RETINAL coordinates, so
  ``retinal_source = (s or s1) - e``. Proprioception is in BODY coordinates.
  Body-frame visual position is ``retinal + eye``.
- Noisy scalar measurements:
    ``x_vis  ~ N(retinal_source, sigma2_vis)``   (retinal frame)
    ``x_eye  ~ N(e,             sigma2_eye)``
    ``x_prop ~ N(s or s2,       sigma2_prop)``    (body frame)

Analytical observer (label pipeline)
-------------------------------------
Operates ONLY on the noisy scalar measurements (never the true sources) and
produces the five training targets:

    out[0] = mu_vis    (body frame)
    out[1] = var_vis   (body frame, uses sigma2_vis + sigma2_eye)
    out[2] = mu_prop
    out[3] = var_prop
    out[4] = p(C=1 | x)   (graded common-cause posterior)

All spatial quantities are in DEGREES; all variances are in DEGREES^2 (deg^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from causal_msi.config import Config, GenerativeConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatentBatch:
    """Latent (hidden) variables for a batch of ``N`` trials.

    Attributes
    ----------
    C
        Causal structure per trial, shape ``(N,)``: 1 = common cause, 2 = separate.
    s_vis
        World-frame source driving vision, shape ``(N,)`` (deg). Equals ``s_prop``
        when ``C == 1``.
    s_prop
        World-frame source driving proprioception, shape ``(N,)`` (deg).
    eye
        Eye position, shape ``(N,)`` (deg).
    retinal_source
        Visual source in retinal coordinates ``s_vis - eye``, shape ``(N,)`` (deg).
    sigma2_vis, sigma2_prop, sigma2_eye
        Per-trial measurement-noise variances, each shape ``(N,)`` (deg^2).
    """

    C: IntArray
    s_vis: FloatArray
    s_prop: FloatArray
    eye: FloatArray
    retinal_source: FloatArray
    sigma2_vis: FloatArray
    sigma2_prop: FloatArray
    sigma2_eye: FloatArray

    @property
    def n(self) -> int:
        """Number of trials in the batch."""
        return int(self.C.shape[0])


@dataclass(frozen=True)
class Measurements:
    """Noisy scalar measurements available to the observer and the encoders.

    Attributes
    ----------
    x_vis
        Visual measurement in RETINAL coordinates, shape ``(N,)`` (deg).
    x_eye
        Eye-position measurement, shape ``(N,)`` (deg).
    x_prop
        Proprioceptive measurement in BODY coordinates, shape ``(N,)`` (deg).
    """

    x_vis: FloatArray
    x_eye: FloatArray
    x_prop: FloatArray


@dataclass(frozen=True)
class ObserverTargets:
    """Analytical-observer outputs: the five training targets plus references.

    The first five attributes are the supervised targets (``to_array`` packs them
    in output order). ``fused_*`` and ``log_bf`` are intermediate quantities kept
    for analysis (e.g. the integration and Bayes-factor modules); they are NOT
    network outputs.
    """

    mu_vis_body: FloatArray
    var_vis_body: FloatArray
    mu_prop: FloatArray
    var_prop: FloatArray
    p_common: FloatArray

    fused_mu: FloatArray
    fused_var: FloatArray
    log_bf: FloatArray

    def to_array(self) -> FloatArray:
        """Pack the five training targets into an ``(N, 5)`` array (output order)."""
        return np.stack(
            [self.mu_vis_body, self.var_vis_body, self.mu_prop, self.var_prop, self.p_common],
            axis=1,
        )


@dataclass(frozen=True)
class Dataset:
    """A materialised dataset: network inputs ``X`` and targets ``Y`` plus metadata.

    Attributes
    ----------
    X
        Population-coded inputs, shape ``(N, input_dim)``.
    Y
        Targets, shape ``(N, 5)`` (or ``(N, 2)`` for integration-only labelling).
    latents
        The :class:`LatentBatch` used to generate the trials (for analysis).
    measurements
        The :class:`Measurements` fed to the encoders / observer (for analysis).
    targets
        The full :class:`ObserverTargets` (including non-output references).
    """

    X: FloatArray
    Y: FloatArray
    latents: LatentBatch
    measurements: Measurements
    targets: ObserverTargets


# --------------------------------------------------------------------------- #
# Sampling (implemented -- standard generative sampling, not core derivation)
# --------------------------------------------------------------------------- #
def sample_causal_structure(rng: np.random.Generator, n: int, p_common: float) -> IntArray:
    """Sample the causal latent ``C`` for ``n`` trials.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    n
        Number of trials.
    p_common
        Prior probability ``P(C=1)`` of a single shared source.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n,)`` with values in ``{1, 2}``.
    """
    common = rng.random(n) < p_common
    return np.where(common, 1, 2).astype(np.int64)


def sample_sources(
    rng: np.random.Generator, c: IntArray, mu0: float, sigma0_sq: float
) -> tuple[FloatArray, FloatArray]:
    """Sample world-frame sources for vision and proprioception.

    For trials with ``C == 1`` a single shared source is used for both cues; for
    ``C == 2`` two independent sources are drawn from the prior.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    c
        Causal structure per trial, shape ``(n,)``.
    mu0, sigma0_sq
        Prior mean (deg) and variance (deg^2).

    Returns
    -------
    tuple of numpy.ndarray
        ``(s_vis, s_prop)``, each shape ``(n,)`` (deg). Equal where ``C == 1``.
    """
    n = c.shape[0]
    sigma0 = float(np.sqrt(sigma0_sq))
    shared = rng.normal(mu0, sigma0, size=n)
    s_vis_indep = rng.normal(mu0, sigma0, size=n)
    s_prop_indep = rng.normal(mu0, sigma0, size=n)
    is_common = c == 1
    s_vis = np.where(is_common, shared, s_vis_indep)
    s_prop = np.where(is_common, shared, s_prop_indep)
    return s_vis, s_prop


def sample_eye_position(
    rng: np.random.Generator, n: int, eye_mu: float, eye_sigma_sq: float
) -> FloatArray:
    """Sample eye position ``e ~ Normal(eye_mu, eye_sigma_sq)``.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    n
        Number of trials.
    eye_mu, eye_sigma_sq
        Eye-position distribution mean (deg) and variance (deg^2).

    Returns
    -------
    numpy.ndarray
        Eye positions, shape ``(n,)`` (deg).
    """
    return rng.normal(eye_mu, float(np.sqrt(eye_sigma_sq)), size=n)


def sample_reliabilities(
    rng: np.random.Generator, n: int, gen: GenerativeConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sample per-trial measurement-noise variances from configured ranges.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    n
        Number of trials.
    gen
        Generative configuration holding the ``sigma2_*_range`` bounds (deg^2).

    Returns
    -------
    tuple of numpy.ndarray
        ``(sigma2_vis, sigma2_prop, sigma2_eye)``, each shape ``(n,)`` (deg^2),
        drawn uniformly from their respective ranges.
    """
    sv = rng.uniform(*gen.sigma2_vis_range, size=n)
    sp = rng.uniform(*gen.sigma2_prop_range, size=n)
    se = rng.uniform(*gen.sigma2_eye_range, size=n)
    return sv, sp, se


def to_retinal(s_vis: FloatArray, eye: FloatArray) -> FloatArray:
    """Map a world-frame visual source to retinal coordinates.

    Parameters
    ----------
    s_vis
        World-frame visual source, shape ``(n,)`` (deg).
    eye
        Eye position, shape ``(n,)`` (deg).

    Returns
    -------
    numpy.ndarray
        Retinal-frame visual source ``s_vis - eye``, shape ``(n,)`` (deg).
    """
    return s_vis - eye


def render_measurements(rng: np.random.Generator, latents: LatentBatch) -> Measurements:
    """Render noisy scalar measurements from the latent variables.

    Vision is measured in the retinal frame, proprioception in the body frame,
    and eye position by its own channel:

        ``x_vis  ~ N(retinal_source, sigma2_vis)``
        ``x_eye  ~ N(eye,            sigma2_eye)``
        ``x_prop ~ N(s_prop,         sigma2_prop)``

    Parameters
    ----------
    rng
        Seeded numpy generator.
    latents
        Latent variables for the batch.

    Returns
    -------
    Measurements
        Noisy scalar measurements, each shape ``(n,)`` (deg).
    """
    x_vis = rng.normal(latents.retinal_source, np.sqrt(latents.sigma2_vis))
    x_eye = rng.normal(latents.eye, np.sqrt(latents.sigma2_eye))
    x_prop = rng.normal(latents.s_prop, np.sqrt(latents.sigma2_prop))
    return Measurements(x_vis=x_vis, x_eye=x_eye, x_prop=x_prop)


def sample_latents(rng: np.random.Generator, n: int, gen: GenerativeConfig) -> LatentBatch:
    """Sample a full batch of latent variables.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    n
        Number of trials.
    gen
        Generative configuration.

    Returns
    -------
    LatentBatch
        All hidden variables for the batch.
    """
    c = sample_causal_structure(rng, n, gen.p_common)
    s_vis, s_prop = sample_sources(rng, c, gen.mu0, gen.sigma0_sq)
    eye = sample_eye_position(rng, n, gen.eye_mu, gen.eye_sigma_sq)
    sv, sp, se = sample_reliabilities(rng, n, gen)
    retinal = to_retinal(s_vis, eye)
    return LatentBatch(
        C=c,
        s_vis=s_vis,
        s_prop=s_prop,
        eye=eye,
        retinal_source=retinal,
        sigma2_vis=sv,
        sigma2_prop=sp,
        sigma2_eye=se,
    )


# --------------------------------------------------------------------------- #
# Analytical observer -- CORE SCIENCE (left for the author to implement)
# --------------------------------------------------------------------------- #
def transform_visual_to_body(
    x_vis: FloatArray, x_eye: FloatArray, sigma2_vis: FloatArray, sigma2_eye: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Transform the retinal visual measurement into the body frame.

    Parameters
    ----------
    x_vis
        Retinal-frame visual measurement, shape ``(n,)`` (deg).
    x_eye
        Eye-position measurement, shape ``(n,)`` (deg).
    sigma2_vis, sigma2_eye
        Visual and eye measurement-noise variances, each shape ``(n,)`` (deg^2).

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_vis_body, var_vis_body)``, each shape ``(n,)``. The transform carries
        the eye-position uncertainty into the visual estimate.
    """
    x_vis_body = x_vis + x_eye
    var_vis_body = sigma2_vis + sigma2_eye
    return x_vis_body, var_vis_body


def segregated_estimate(
    x: FloatArray, var: FloatArray, mu0: float, sigma0_sq: float
) -> tuple[FloatArray, FloatArray]:
    """Single-cue posterior: combine one measurement with the spatial prior.

    Parameters
    ----------
    x
        Cue measurement in the body frame, shape ``(n,)`` (deg).
    var
        Measurement variance of the cue, shape ``(n,)`` (deg^2). For vision this
        is the transformed ``sigma2_vis + sigma2_eye``.
    mu0, sigma0_sq
        Prior mean (deg) and variance (deg^2).

    Returns
    -------
    tuple of numpy.ndarray
        ``(mu, var_post)``, each shape ``(n,)`` -- the segregated posterior mean
        (deg) and variance (deg^2).
    """
    mu = (x / var + mu0 / sigma0_sq) / (1 / var + 1 / sigma0_sq)
    var_post = 1 / (1 / var + 1 / sigma0_sq)

    return mu, var_post


def fused_estimate(
    x_vis_body: FloatArray,
    var_vis_body: FloatArray,
    x_prop: FloatArray,
    var_prop: FloatArray,
    mu0: float,
    sigma0_sq: float,
) -> tuple[FloatArray, FloatArray]:
    """Forced-fusion posterior assuming a common cause (``C == 1``).

    Parameters
    ----------
    x_vis_body, var_vis_body
        Body-frame visual measurement and variance, each shape ``(n,)``.
    x_prop, var_prop
        Proprioceptive measurement and variance, each shape ``(n,)``.
    mu0, sigma0_sq
        Prior mean (deg) and variance (deg^2).

    Returns
    -------
    tuple of numpy.ndarray
        ``(mu_fused, var_fused)``, each shape ``(n,)`` -- the optimal combination
        of both cues with the prior, under the assumption of a single source.
    """
    P = 1 / var_vis_body + 1 / var_prop + 1 / sigma0_sq
    mu_fused = (x_vis_body / var_vis_body + x_prop / var_prop + mu0 / sigma0_sq) / P
    var_fused = 1 / P
    # See Kording et al. (2007).
    return mu_fused, var_fused


def log_bayes_factor(
    x_vis_body: FloatArray,
    var_vis_body: FloatArray,
    x_prop: FloatArray,
    var_prop: FloatArray,
    mu0: float,
    sigma0_sq: float,
) -> FloatArray:
    """Log Bayes factor ``log p(x | C=1) - log p(x | C=2)``.

    Parameters
    ----------
    x_vis_body, var_vis_body
        Body-frame visual measurement and variance, each shape ``(n,)``.
    x_prop, var_prop
        Proprioceptive measurement and variance, each shape ``(n,)``.
    mu0, sigma0_sq
        Prior mean (deg) and variance (deg^2).

    Returns
    -------
    numpy.ndarray
        Log Bayes factor per trial, shape ``(n,)`` (nats). Large positive values
        favour a common cause; large negative values favour separate causes.

    Notes
    -----
    Gaussian closed form (Kording et al., 2007). The two evidence
    terms are
        ``p(x | C=1) = N(x_vis_body - x_prop ; 0, var_vis_body + var_prop + ...)``
        marginalising the shared source over the prior, versus
        ``p(x | C=2)`` with each cue marginalised over its own prior.
    The Bayes factor is driven by the body-frame disparity
    ``d = x_vis_body - x_prop`` relative to the combined noise: ``BF -> large`` as
    ``d -> 0`` and ``BF -> 0`` as ``|d| -> inf``. Tested in
    ``tests/test_generative.py::test_bf_decreases_with_disparity``.

    Implements the Kording et al. (2007) Gaussian closed form. With
    ``Sigma_c = var_vis_body*var_prop + var_vis_body*sigma0_sq + var_prop*sigma0_sq``,

        log p(x | C=1) = -0.5*log(Sigma_c) - 0.5*E1   (dropping the -log(2*pi))
        E1 = [ (x_vis_body - x_prop)^2 * sigma0_sq
               + (x_vis_body - mu0)^2 * var_prop
               + (x_prop     - mu0)^2 * var_vis_body ] / Sigma_c

        log p(x | C=2) = -0.5*log(v_vis * v_prop) - 0.5*E2   (-log(2*pi) dropped)
        v_vis = var_vis_body + sigma0_sq,  v_prop = var_prop + sigma0_sq
        E2 = (x_vis_body - mu0)^2 / v_vis + (x_prop - mu0)^2 / v_prop

    The shared ``-log(2*pi)`` normaliser cancels in the difference.
    """
    sv = var_vis_body
    sp = var_prop
    s0 = sigma0_sq

    # --- Evidence for a common cause (C=1): marginalise the shared source. ---
    sigma_c = sv * sp + sv * s0 + sp * s0
    e1 = (
        (x_vis_body - x_prop) ** 2 * s0 + (x_vis_body - mu0) ** 2 * sp + (x_prop - mu0) ** 2 * sv
    ) / sigma_c
    log_p_c1 = -0.5 * np.log(sigma_c) - 0.5 * e1

    # --- Evidence for separate causes (C=2): independent, prior-marginalised. ---
    v_vis = sv + s0
    v_prop = sp + s0
    e2 = (x_vis_body - mu0) ** 2 / v_vis + (x_prop - mu0) ** 2 / v_prop
    log_p_c2 = -0.5 * np.log(v_vis * v_prop) - 0.5 * e2

    return log_p_c1 - log_p_c2


def common_cause_posterior(log_bf: FloatArray, p_common: float) -> FloatArray:
    """Graded common-cause posterior ``p(C=1 | x)`` from the log Bayes factor.

    Parameters
    ----------
    log_bf
        Log Bayes factor per trial, shape ``(n,)`` (nats).
    p_common
        Prior ``P(C=1)``.

    Returns
    -------
    numpy.ndarray
        Posterior probability of a common cause, shape ``(n,)``, in ``[0, 1]``.

    Notes
    -----
    with ``BF = exp(log_bf)``
        ``p(C=1 | x) = BF * p_common / (BF * p_common + (1 - p_common))``.
    Equivalently a logistic of ``log_bf + logit(p_common)``. The posterior must be
    monotone increasing in ``log_bf``. Tested in
    ``tests/test_generative.py::test_posterior_monotone_in_bf`` and the disparity
    limit tests.

    Implemented as ``p = expit(log_bf + logit(p_common))`` -- algebraically
    identical to ``BF*p_common / (BF*p_common + (1 - p_common))`` but stable for
    large ``|log_bf|`` (no ``exp(log_bf)`` overflow). ``p_common`` is clipped away
    from 0/1 so the prior logit stays finite.
    """
    pc = float(np.clip(p_common, 1e-12, 1.0 - 1e-12))
    prior_logit = np.log(pc) - np.log1p(-pc)
    posterior: FloatArray = expit(log_bf + prior_logit)
    return posterior


def analytical_observer(
    meas: Measurements, latents: LatentBatch, gen: GenerativeConfig
) -> ObserverTargets:
    """Run the full analytical observer on the noisy measurements.

    Composes the body-frame transform, the segregated single-cue posteriors, the
    forced-fusion posterior, the Bayes factor, and the common-cause posterior into
    the five training targets (plus references kept for analysis).

    Parameters
    ----------
    meas
        Noisy scalar measurements.
    latents
        Latent batch -- used ONLY for the per-trial noise variances
        (``sigma2_*``), never the true sources or ``C``.
    gen
        Generative configuration (prior + ``p_common``).

    Returns
    -------
    ObserverTargets
        The five targets and the intermediate fused/BF references.
    """
    x_vis_body, var_vis_body = transform_visual_to_body(
        meas.x_vis, meas.x_eye, latents.sigma2_vis, latents.sigma2_eye
    )
    mu_vis_body, post_var_vis = segregated_estimate(
        x_vis_body, var_vis_body, gen.mu0, gen.sigma0_sq
    )
    mu_prop, post_var_prop = segregated_estimate(
        meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    fused_mu, fused_var = fused_estimate(
        x_vis_body, var_vis_body, meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    log_bf = log_bayes_factor(
        x_vis_body, var_vis_body, meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    p_common = common_cause_posterior(log_bf, gen.p_common)
    return ObserverTargets(
        mu_vis_body=mu_vis_body,
        var_vis_body=post_var_vis,
        mu_prop=mu_prop,
        var_prop=post_var_prop,
        p_common=p_common,
        fused_mu=fused_mu,
        fused_var=fused_var,
        log_bf=log_bf,
    )


# --------------------------------------------------------------------------- #
# Dataset assembly (implemented plumbing)
# --------------------------------------------------------------------------- #
def build_dataset(rng: np.random.Generator, config: Config, n_trials: int | None = None) -> Dataset:
    """Generate a full ``(X, Y)`` dataset from a config.

    Samples latents, renders measurements, encodes the three population-code input
    groups, and runs the analytical observer to produce targets.

    Parameters
    ----------
    rng
        Seeded numpy generator.
    config
        Full validated configuration.
    n_trials
        Override for the number of trials; defaults to ``config.training.n_trials``.

    Returns
    -------
    Dataset
        Materialised inputs, targets, latents, measurements, and observer outputs.

    """
    from causal_msi.encoding import assemble_inputs

    n = int(n_trials if n_trials is not None else config.training.n_trials)
    latents = sample_latents(rng, n, config.generative)
    meas = render_measurements(rng, latents)
    targets = analytical_observer(meas, latents, config.generative)

    x = assemble_inputs(rng, meas, latents, config.encoding, p_common=config.generative.p_common)

    if config.model.head_type == "integration_only":
        y = np.stack([targets.fused_mu, targets.fused_var], axis=1)
    else:
        y = targets.to_array()

    return Dataset(X=x, Y=y, latents=latents, measurements=meas, targets=targets)


def train_val_test_split(
    n: int, split: tuple[float, float, float], rng: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle indices and partition them into train/val/test sets.

    Parameters
    ----------
    n
        Number of trials.
    split
        ``(train, val, test)`` fractions; must sum to 1.
    rng
        Seeded numpy generator.

    Returns
    -------
    tuple of numpy.ndarray
        ``(train_idx, val_idx, test_idx)`` integer index arrays.
    """
    idx = rng.permutation(n)
    n_train = int(round(split[0] * n))
    n_val = int(round(split[1] * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx
