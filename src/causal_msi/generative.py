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
produces the four training targets -- the Bayesian causal-inference *optimal*
position estimates of Kording et al. (2007):

    out[0] = mu_vis    (Eq. 9:  model-averaged optimal visual estimate, body frame)
    out[1] = var_vis   (posterior/mixture variance of the visual estimate)
    out[2] = mu_prop   (Eq. 10: model-averaged optimal proprioceptive estimate)
    out[3] = var_prop  (posterior/mixture variance of the prop estimate)

The common-cause posterior ``p(C=1 | x)`` (Eq. 2) is computed internally to form
these estimates but is NEITHER an input NOR an output.

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
    """Analytical-observer outputs: the four training targets plus references.

    The four supervised targets are the Bayesian causal-inference *optimal*
    position estimates (Kording et al., 2007): for each modality the
    model-averaged estimate (Eqs. 9/10) and its posterior (mixture) variance.
    ``p(C=1)`` is computed internally (Eq. 2) but is **not** a training target.

    ``to_array`` packs the four targets in output order
    ``[mu_vis, var_vis, mu_prop, var_prop]``. The remaining attributes are
    intermediate references kept for analysis (segregated/fused estimates, the
    common-cause posterior, the log Bayes factor); they are NOT network outputs.
    """

    # --- the four supervised training targets (Kording Eqs. 9/10 + mixture var) ---
    mu_vis: FloatArray  # Eq. 9: optimal visual estimate (model-averaged)
    var_vis: FloatArray  # posterior (mixture) variance of the visual estimate
    mu_prop: FloatArray  # Eq. 10: optimal proprioceptive estimate (model-averaged)
    var_prop: FloatArray  # posterior (mixture) variance of the prop estimate

    # --- internal references for analysis (NOT outputs) ---
    seg_vis_mu: FloatArray  # Eq. 11: visual estimate given C=2 (separate causes)
    seg_vis_var: FloatArray  # variance of the C=2 visual posterior
    seg_prop_mu: FloatArray  # Eq. 11: prop estimate given C=2
    seg_prop_var: FloatArray  # variance of the C=2 prop posterior
    fused_mu: FloatArray  # Eq. 12: shared estimate given C=1 (common cause)
    fused_var: FloatArray  # variance of the C=1 fused posterior
    p_common: FloatArray  # Eq. 2: posterior probability of a common cause
    log_bf: FloatArray  # log p(x|C=1) - log p(x|C=2)

    def to_array(self) -> FloatArray:
        """Pack the four training targets into an ``(N, 4)`` array (output order)."""
        return np.stack(
            [self.mu_vis, self.var_vis, self.mu_prop, self.var_prop],
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
        Targets, shape ``(N, 4)`` (or ``(N, 2)`` for integration-only labelling).
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


def model_averaged_estimate(
    p_common: FloatArray,
    fused_mu: FloatArray,
    fused_var: FloatArray,
    seg_mu: FloatArray,
    seg_var: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Bayesian model-averaged optimal estimate for one modality (Kording Eqs. 9/10).

    Combines the common-cause (C=1, Eq. 12) and separate-cause (C=2, Eq. 11)
    estimates by the posterior probability of a common cause. The mean is the
    cost-minimising estimate (mean of the 2-component Gaussian mixture posterior);
    the variance is the variance of that same mixture, via the law of total
    variance.

    Parameters
    ----------
    p_common
        Posterior ``p(C=1 | x)`` per trial, shape ``(n,)`` (Eq. 2).
    fused_mu, fused_var
        The C=1 (common-cause) estimate and its variance, each shape ``(n,)``
        (Eq. 12).
    seg_mu, seg_var
        The C=2 (separate-cause) single-cue estimate and its variance, each shape
        ``(n,)`` (Eq. 11).

    Returns
    -------
    tuple of numpy.ndarray
        ``(mu, var)`` -- the model-averaged optimal estimate (Eq. 9 or 10) and its
        posterior (mixture) variance, each shape ``(n,)``.

    Notes
    -----
    Mean (Eq. 9/10):  ``mu = p_common * fused_mu + (1 - p_common) * seg_mu``.
    Mixture variance (law of total variance):
        ``var = p_common*(fused_var + fused_mu**2)
                + (1 - p_common)*(seg_var + seg_mu**2) - mu**2``.
    """
    w = p_common
    mu = w * fused_mu + (1.0 - w) * seg_mu
    second_moment = w * (fused_var + fused_mu**2) + (1.0 - w) * (seg_var + seg_mu**2)
    var = second_moment - mu**2
    return mu, var


def analytical_observer(
    meas: Measurements, latents: LatentBatch, gen: GenerativeConfig
) -> ObserverTargets:
    """Run the full Kording (2007) causal-inference observer on the measurements.

    Pipeline:
    1. Reference-frame transform: map the retinal visual measurement to the body
       frame (``x_vis_body = x_vis + x_eye``, variance ``sigma2_vis + sigma2_eye``)
       so both cues live in the body frame before applying Kording's equations.
    2. Separate-cause (C=2) single-cue estimates per modality (Eq. 11).
    3. Common-cause (C=1) fused estimate, shared by both modalities (Eq. 12).
    4. Common-cause posterior ``p(C=1 | x)`` from the Gaussian evidences (Eqs. 2/4).
    5. Model-averaged optimal estimates + posterior variances (Eqs. 9/10).

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
        The four optimal-estimate targets (Eqs. 9/10) plus internal references
        (segregated/fused estimates, ``p(C=1)``, log Bayes factor).
    """
    x_vis_body, var_vis_body = transform_visual_to_body(
        meas.x_vis, meas.x_eye, latents.sigma2_vis, latents.sigma2_eye
    )
    # C=2 separate-cause single-cue estimates (Eq. 11).
    seg_vis_mu, seg_vis_var = segregated_estimate(x_vis_body, var_vis_body, gen.mu0, gen.sigma0_sq)
    seg_prop_mu, seg_prop_var = segregated_estimate(
        meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    # C=1 common-cause fused estimate, shared by both modalities (Eq. 12).
    fused_mu, fused_var = fused_estimate(
        x_vis_body, var_vis_body, meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    # Common-cause posterior p(C=1 | x) from the Gaussian evidences (Eqs. 2/4/6).
    log_bf = log_bayes_factor(
        x_vis_body, var_vis_body, meas.x_prop, latents.sigma2_prop, gen.mu0, gen.sigma0_sq
    )
    p_common = common_cause_posterior(log_bf, gen.p_common)
    # Model-averaged optimal estimates + posterior variances (Eqs. 9/10).
    mu_vis, var_vis = model_averaged_estimate(
        p_common, fused_mu, fused_var, seg_vis_mu, seg_vis_var
    )
    mu_prop, var_prop = model_averaged_estimate(
        p_common, fused_mu, fused_var, seg_prop_mu, seg_prop_var
    )
    return ObserverTargets(
        mu_vis=mu_vis,
        var_vis=var_vis,
        mu_prop=mu_prop,
        var_prop=var_prop,
        seg_vis_mu=seg_vis_mu,
        seg_vis_var=seg_vis_var,
        seg_prop_mu=seg_prop_mu,
        seg_prop_var=seg_prop_var,
        fused_mu=fused_mu,
        fused_var=fused_var,
        p_common=p_common,
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
    from causal_msi.encoding import Encoders, assemble_inputs

    n = int(n_trials if n_trials is not None else config.training.n_trials)
    latents = sample_latents(rng, n, config.generative)
    meas = render_measurements(rng, latents)
    targets = analytical_observer(meas, latents, config.generative)

    # Build the population encoders from a DEDICATED, deterministic RNG seeded by
    # config.seed (NOT the data RNG). This fixes the random push-pull tuning so that
    # any two datasets built from the same config share identical encoders -- the
    # model trained on one can be evaluated on another. Otherwise each build re-rolls
    # the tuning and a trained model sees a different input basis (garbage R^2).
    encoders = Encoders.build(np.random.default_rng(config.seed), config.encoding)
    x = assemble_inputs(rng, meas, latents, config.encoding, encoders=encoders)

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
