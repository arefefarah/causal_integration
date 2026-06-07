"""Property tests for the analytical observer (CORE SCIENCE).

Every test here is marked ``science``/``xfail`` against ``NotImplementedError``:
the functions in ``causal_msi.generative`` are intentionally left unimplemented
(``TODO(science)``). The assertions encode the properties the implementation must
satisfy. As each closed form is filled in, its test should flip from xfail to pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_msi.generative import (
    common_cause_posterior,
    log_bayes_factor,
    segregated_estimate,
    transform_visual_to_body,
)

science = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(science): not yet implemented", strict=False
)


@science
def test_transformed_visual_variance() -> None:
    """Body-frame visual variance equals sigma2_vis + sigma2_eye."""
    n = 100
    x_vis = np.zeros(n)
    x_eye = np.zeros(n)
    sigma2_vis = np.full(n, 4.0)
    sigma2_eye = np.full(n, 9.0)
    _, var_body = transform_visual_to_body(x_vis, x_eye, sigma2_vis, sigma2_eye)
    np.testing.assert_allclose(var_body, 13.0)


@science
def test_transformed_visual_mean() -> None:
    """Body-frame visual mean equals x_vis + x_eye."""
    x_vis = np.array([1.0, -2.0, 5.0])
    x_eye = np.array([0.5, 0.5, -1.0])
    sigma2 = np.ones(3)
    mean_body, _ = transform_visual_to_body(x_vis, x_eye, sigma2, sigma2)
    np.testing.assert_allclose(mean_body, x_vis + x_eye)


@science
def test_segregated_reduces_to_prior() -> None:
    """As a cue becomes uninformative (var -> inf), estimate collapses to the prior."""
    mu0, sigma0_sq = 3.0, 10.0
    x = np.array([100.0])  # far from prior, but ...
    var = np.array([1e12])  # ... essentially no information
    mu, var_post = segregated_estimate(x, var, mu0, sigma0_sq)
    np.testing.assert_allclose(mu, mu0, rtol=1e-3)
    np.testing.assert_allclose(var_post, sigma0_sq, rtol=1e-3)


@science
def test_segregated_reduces_to_single_cue() -> None:
    """With a very weak prior, the estimate reduces to the measurement itself."""
    mu0, sigma0_sq = 0.0, 1e12
    x = np.array([7.0])
    var = np.array([2.0])
    mu, var_post = segregated_estimate(x, var, mu0, sigma0_sq)
    np.testing.assert_allclose(mu, x, rtol=1e-3)
    np.testing.assert_allclose(var_post, var, rtol=1e-3)


@science
def test_segregated_variance_shrinks() -> None:
    """The posterior variance is smaller than both the cue and the prior variance."""
    mu0, sigma0_sq = 0.0, 10.0
    x = np.array([1.0])
    var = np.array([4.0])
    _, var_post = segregated_estimate(x, var, mu0, sigma0_sq)
    assert var_post[0] < min(4.0, 10.0)


@science
def test_bf_decreases_with_disparity() -> None:
    """The log Bayes factor decreases as body-frame disparity grows."""
    mu0, sigma0_sq = 0.0, 100.0
    var = np.full(5, 4.0)
    x_prop = np.zeros(5)
    x_vis = np.array([0.0, 2.0, 5.0, 10.0, 20.0])  # increasing disparity
    log_bf = log_bayes_factor(x_vis, var, x_prop, var, mu0, sigma0_sq)
    assert np.all(np.diff(log_bf) < 0)


@science
def test_posterior_to_one_at_zero_disparity() -> None:
    """p(C=1) -> 1 at zero disparity when the cues are reliable.

    At exactly zero body-frame disparity the posterior is bounded by the cue
    reliabilities and the prior; it approaches 1 only as the measurement
    variances shrink relative to the prior variance. Reliable cues (var << sigma0)
    are used here so the limit is exercised.
    """
    mu0, sigma0_sq, p_common = 0.0, 100.0, 0.5
    var = np.array([0.1])
    log_bf = log_bayes_factor(np.array([0.0]), var, np.array([0.0]), var, mu0, sigma0_sq)
    p = common_cause_posterior(log_bf, p_common)
    assert p[0] > 0.9


@science
def test_posterior_to_zero_at_large_disparity() -> None:
    """p(C=1) -> 0 as body-frame disparity -> inf (cues far apart)."""
    mu0, sigma0_sq, p_common = 0.0, 100.0, 0.5
    var = np.array([1.0])
    log_bf = log_bayes_factor(np.array([0.0]), var, np.array([1000.0]), var, mu0, sigma0_sq)
    p = common_cause_posterior(log_bf, p_common)
    assert p[0] < 0.1


@science
def test_posterior_monotone_in_bf() -> None:
    """p(C=1|x) is monotone increasing in the log Bayes factor."""
    log_bf = np.linspace(-10.0, 10.0, 50)
    p = common_cause_posterior(log_bf, p_common=0.5)
    assert np.all(np.diff(p) > 0)


@science
def test_posterior_in_unit_interval() -> None:
    """The common-cause posterior stays within [0, 1]."""
    log_bf = np.linspace(-50.0, 50.0, 200)
    p = common_cause_posterior(log_bf, p_common=0.3)
    assert np.all((p >= 0.0) & (p <= 1.0))
