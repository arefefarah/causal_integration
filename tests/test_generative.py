"""Does the observer obey the theory?

These are property tests, not regression tests: each one states something that
must hold for any correct implementation of Kording et al. (2007), so they stay
valid if you change the parameters or rewrite the internals.
"""

import numpy as np
import pytest

from cmsi.data.generative import (
    common_cause_posterior,
    fused_posterior,
    log_bayes_factor,
    model_average,
    observer,
    sample_trials,
    single_cue_posterior,
    to_body_frame,
)

PRIOR = (0.0, 100.0)   # mu0, sigma0_sq


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_common_cause_trials_share_one_source(cfg, rng):
    d = sample_trials(5000, cfg["generative"], rng)
    common = d["C"] == 1
    assert common.any() and (~common).any(), "both causal structures should occur"
    assert np.allclose(d["s_vis"][common], d["s_prop"][common])
    assert not np.allclose(d["s_vis"][~common], d["s_prop"][~common])


def test_vision_is_retinal(cfg, rng):
    d = sample_trials(1000, cfg["generative"], rng)
    assert np.allclose(d["retinal"], d["s_vis"] - d["eye"])


def test_sampling_is_reproducible(cfg):
    a = sample_trials(500, cfg["generative"], np.random.default_rng(3))
    b = sample_trials(500, cfg["generative"], np.random.default_rng(3))
    assert all(np.array_equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# posteriors
# --------------------------------------------------------------------------- #
def test_single_cue_posterior_sits_between_cue_and_prior():
    x = np.array([20.0])
    mu, var = single_cue_posterior(x, np.array([4.0]), *PRIOR)
    assert 0.0 < mu[0] < x[0], "posterior mean is pulled toward the prior"
    assert var[0] < 4.0, "adding the prior can only reduce uncertainty"


def test_fusion_is_more_certain_than_either_cue():
    args = (np.array([5.0]), np.array([4.0]), np.array([1.0]), np.array([9.0]))
    _, fused_var = fused_posterior(*args, *PRIOR)
    _, vis_var = single_cue_posterior(args[0], args[1], *PRIOR)
    _, prop_var = single_cue_posterior(args[2], args[3], *PRIOR)
    assert fused_var[0] < min(vis_var[0], prop_var[0])


def test_fused_estimate_leans_toward_the_reliable_cue():
    # vision far more reliable than proprioception -> fused sits nearer vision
    mu, _ = fused_posterior(np.array([10.0]), np.array([0.5]),
                            np.array([-10.0]), np.array([50.0]), *PRIOR)
    assert mu[0] > 0


def test_body_frame_transform_adds_eye_uncertainty():
    """Variances ADD here -- this is the frame transformation, not cue
    combination, so the two uncertainties accumulate rather than shrink."""
    x, var = to_body_frame(np.array([3.0]), np.array([2.0]), np.array([1.0]),
                           np.array([4.0]), eye_mu=0.0, eye_sigma_sq=np.inf)
    assert x[0] == 5.0
    assert var[0] == 5.0


def test_eye_prior_shrinks_the_eye_contribution():
    """With a proper prior on eye position the measurement is pulled toward it,
    and the transformed estimate is correspondingly more certain."""
    args = (np.array([3.0]), np.array([2.0]), np.array([1.0]), np.array([4.0]))
    flat_x, flat_var = to_body_frame(*args, eye_mu=0.0, eye_sigma_sq=np.inf)
    x, var = to_body_frame(*args, eye_mu=0.0, eye_sigma_sq=25.0)

    k = 25.0 / (25.0 + 4.0)
    assert x[0] == pytest.approx(3.0 + k * 2.0)
    assert var[0] == pytest.approx(1.0 + 1 / (1 / 4.0 + 1 / 25.0))
    assert abs(x[0] - 3.0) < abs(flat_x[0] - 3.0)
    assert var[0] < flat_var[0]


def test_eye_prior_gives_a_better_body_frame_estimate(cfg, rng):
    """The raw sum x_vis + x_eye is unbiased but not efficient: the two share e,
    so they are correlated and the sum is not the sufficient statistic. Ignoring
    the eye prior therefore costs accuracy against the true source."""
    d = sample_trials(50000, cfg["generative"], rng)
    gen = cfg["generative"]
    shared = (d["x_vis"], d["x_eye"], d["sig2_vis"], d["sig2_eye"])

    raw, _ = to_body_frame(*shared, eye_mu=gen["eye_mu"], eye_sigma_sq=np.inf)
    optimal, _ = to_body_frame(*shared, eye_mu=gen["eye_mu"],
                               eye_sigma_sq=gen["eye_sigma_sq"])

    assert np.mean((optimal - d["s_vis"]) ** 2) < np.mean((raw - d["s_vis"]) ** 2)


# --------------------------------------------------------------------------- #
# causal inference
# --------------------------------------------------------------------------- #
def test_bayes_factor_falls_with_disparity():
    disparity = np.linspace(0, 40, 25)
    var = np.full_like(disparity, 4.0)
    log_bf = log_bayes_factor(disparity, var, np.zeros_like(disparity), var, *PRIOR)
    assert np.all(np.diff(log_bf) < 0), "agreement is evidence for a common cause"


def test_posterior_is_monotone_in_the_bayes_factor():
    log_bf = np.linspace(-20, 20, 100)
    p = common_cause_posterior(log_bf, 0.5)
    assert np.all(np.diff(p) > 0)
    assert np.all((p >= 0) & (p <= 1))


def test_posterior_saturates_at_the_extremes():
    p = common_cause_posterior(np.array([-500.0, 0.0, 500.0]), 0.5)
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[1] == pytest.approx(0.5)
    assert p[2] == pytest.approx(1.0, abs=1e-9)


def test_prior_shifts_the_posterior_the_right_way():
    log_bf = np.zeros(1)
    assert common_cause_posterior(log_bf, 0.9)[0] > common_cause_posterior(log_bf, 0.1)[0]


# --------------------------------------------------------------------------- #
# model averaging
# --------------------------------------------------------------------------- #
def test_model_averaging_hits_both_endpoints():
    fused_mu, fused_var = np.array([2.0]), np.array([1.0])
    seg_mu, seg_var = np.array([8.0]), np.array([3.0])

    mu1, var1 = model_average(np.array([1.0]), fused_mu, fused_var, seg_mu, seg_var)
    assert mu1[0] == pytest.approx(2.0) and var1[0] == pytest.approx(1.0)

    mu0, var0 = model_average(np.array([0.0]), fused_mu, fused_var, seg_mu, seg_var)
    assert mu0[0] == pytest.approx(8.0) and var0[0] == pytest.approx(3.0)


def test_mixture_variance_exceeds_its_components_when_they_disagree():
    """The law of total variance: disagreement between the two hypotheses is
    itself uncertainty, which is why var_vis stays large at mid disparities."""
    _, var = model_average(np.array([0.5]), np.array([-10.0]), np.array([1.0]),
                           np.array([10.0]), np.array([1.0]))
    assert var[0] > 1.0


# --------------------------------------------------------------------------- #
# the whole observer
# --------------------------------------------------------------------------- #
def test_observer_outputs_are_finite_and_variances_positive(cfg, rng):
    d = sample_trials(2000, cfg["generative"], rng)
    out = observer(d, cfg["generative"])
    for key, value in out.items():
        assert np.all(np.isfinite(value)), f"{key} contains non-finite values"
    for key in ("var_vis", "var_prop", "fused_var", "seg_vis_var", "seg_prop_var"):
        assert np.all(out[key] > 0), f"{key} must be positive"


def test_observer_never_touches_the_true_sources(cfg, rng):
    """Same measurements, different hidden truth -> identical targets.

    This is the one that matters: if it fails, the labels leak information the
    network could not possibly have, and every result is inflated.
    """
    d = sample_trials(500, cfg["generative"], rng)
    baseline = observer(d, cfg["generative"])

    tampered = dict(d)
    tampered["s_vis"] = d["s_vis"] + 100.0
    tampered["s_prop"] = d["s_prop"] - 100.0
    tampered["C"] = 3 - d["C"]
    after = observer(tampered, cfg["generative"])

    assert all(np.array_equal(baseline[k], after[k]) for k in baseline)


def test_low_disparity_trials_are_judged_more_common(cfg, rng):
    d = sample_trials(5000, cfg["generative"], rng)
    out = observer(d, cfg["generative"])
    near = np.abs(out["disparity"]) < 2
    far = np.abs(out["disparity"]) > 20
    assert out["post_c1"][near].mean() > out["post_c1"][far].mean()
