"""Do the analyses recover a known answer?

Each test plants a ground truth and checks the analysis finds it. If these pass,
a surprising result is more likely to be a finding than a bug.
"""

import numpy as np
import pytest

from cmsi import analysis
from cmsi.data.generative import common_cause_posterior


def test_accuracy_is_perfect_on_a_perfect_prediction():
    target = np.random.randn(500, 2) * 5
    rows = analysis.accuracy(target.copy(), target, ["a", "b"])
    for row in rows:
        assert row["r2"] == pytest.approx(1.0)
        assert row["slope"] == pytest.approx(1.0)
        assert row["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_binned_weight_and_weight_by_posterior_average_every_trial():
    """The two binnings of a per-trial weight: by disparity on a grid (mean
    and standard error, sparse or noisy bins dropped) and by posterior."""
    rng = np.random.default_rng(0)
    disp = rng.uniform(-30, 30, 6000)
    post = 1 / (1 + np.exp(0.4 * (np.abs(disp) - 8)))
    w = post + rng.normal(0, 0.1, disp.size)
    grid = np.array([-20, -10, 0, 10, 20], float)
    c, m, n, se = analysis.binned_weight(disp, w, grid, min_count=25)
    assert n.sum() == disp.size and np.all(n >= 25)
    nearest = np.abs(disp[:, None] - grid[None, :]).argmin(1)
    for b, (cc, mm, k, s) in enumerate(zip(c, m, n, se, strict=True)):
        member = nearest == b
        assert cc == grid[b] and k == member.sum()
        assert mm == pytest.approx(w[member].mean(), abs=1e-9)
        assert s == pytest.approx(w[member].std(ddof=1) / np.sqrt(k), abs=1e-9)
    # a bin too sparse, or too uncertain, is dropped rather than drawn
    c2, *_ = analysis.binned_weight(disp, w, [-20, 0, 20, 200], min_count=25)
    assert 200 not in c2
    c3, *_ = analysis.binned_weight(disp, w, [-20, 0, 20], min_count=25, max_se=1e-6)
    assert c3.size == 0
    bp = analysis.weight_by_posterior(w, post, n_bins=10, min_count=25)
    assert np.all(np.diff(bp["centres"]) > 0) and bp["n"].sum() == disp.size
    assert np.allclose(bp["w"], bp["centres"], atol=0.03)


def test_implied_log_bf_inverts_the_posterior():
    log_bf = np.linspace(-15, 15, 200)
    p = common_cause_posterior(log_bf, 0.3)
    assert np.allclose(analysis.implied_log_bf(p, 0.3), log_bf, atol=1e-6)


def test_mean_by_bin_averages_within_bins():
    x = np.array([-10.0, -9.9, 0.0, 0.1, 10.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    centres, means, counts = analysis.mean_by_bin(x, y, [-10, 0, 10])
    assert np.array_equal(centres, [-10, 0, 10])
    assert np.allclose(means, [2.0, 6.0, 9.0])
    assert np.array_equal(counts, [2, 2, 1])


def test_mean_by_bin_ignores_nans():
    centres, means, counts = analysis.mean_by_bin(
        np.array([0.0, 0.0]), np.array([np.nan, 4.0]), [0])
    assert means[0] == 4.0 and counts[0] == 1


def test_transition_fit_recovers_a_planted_midpoint():
    d = np.linspace(0, 30, 300)
    w = 1 / (1 + np.exp(0.4 * (d - 12.0)))
    midpoint, sharpness = analysis.transition_fit(d, w)
    assert midpoint == pytest.approx(12.0, abs=0.5)
    assert sharpness == pytest.approx(0.4, abs=0.05)


def test_strategy_fit_identifies_the_strategy_that_made_the_data():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 3000)
    fused, seg = rng.normal(0, 3, 3000), rng.normal(0, 3, 3000) + 15

    assert analysis.strategy_fit(
        p * fused + (1 - p) * seg, p, fused, seg)["best"] == "averaging"
    assert analysis.strategy_fit(
        np.where(p > 0.5, fused, seg), p, fused, seg)["best"] == "selection"


def test_decoding_finds_a_linear_signal_and_rejects_noise():
    rng = np.random.default_rng(2)
    acts = rng.normal(size=(1500, 20))
    signal = acts @ rng.normal(size=20) + rng.normal(0, 0.05, 1500)

    assert analysis.decode(acts, signal)["r2"] > 0.95
    assert analysis.decode(acts, rng.normal(size=1500))["r2"] < 0.1


def test_decode_by_layer_reports_every_hidden_layer():
    rng = np.random.default_rng(3)
    acts = {"layer0": rng.normal(size=(400, 8)), "layer1": rng.normal(size=(400, 8)),
            "sil": rng.normal(size=(400, 8))}   # aliases must not be double-counted
    out = analysis.decode_by_layer(acts, rng.normal(size=400))
    assert set(out) == {"layer0", "layer1"}


def _mixture_observer(n=4000, pull=1.0, seed=0):
    """An observer that mixes both outputs with the weight pull * posterior."""
    rng = np.random.default_rng(seed)
    disp = rng.normal(0, 12, n)
    post = 1 / (1 + np.exp(0.4 * (np.abs(disp) - 8)))
    fused_var = rng.uniform(2.0, 3.0, n)
    seg_var = fused_var + rng.uniform(3, 6, n)
    delta = rng.normal(0, 4, n)
    w = pull * post
    var_out = analysis.mixture_variance(w, fused_var, seg_var, delta)
    return post, fused_var, seg_var, delta, w, var_out


def test_variance_weight_inverts_the_mixture_variance_where_the_root_is_unique():
    """Eq. 10 solved for w: exact wherever Delta^2 <= c -- including Delta = 0,
    where a position ratio is undefined -- and flagged where two roots exist
    (there the hybrid read turns to the position output instead)."""
    post, fused_var, seg_var, delta, w, var_out = _mixture_observer()
    delta[:500] = 0.0
    var_out = analysis.mixture_variance(w, fused_var, seg_var, delta)
    w_hat, sens, flags = analysis.variance_weight(var_out, fused_var, seg_var, delta)
    two = delta ** 2 > seg_var - fused_var
    assert np.allclose(w_hat[~two], w[~two], atol=1e-6)
    assert np.all(flags[:500] == 0)
    assert np.all(flags[two] == 1) and np.all(flags[~two & (delta != 0)] == 0)
    # on the two-root trials the larger root is returned; it is the true one
    # only when the weight lies on the falling side of the parabola
    with np.errstate(divide="ignore"):
        peak = 0.5 - (seg_var - fused_var) / (2 * delta ** 2)
    falling = two & (w > peak)
    assert np.allclose(w_hat[falling], w[falling], atol=1e-6)
    # the sensitivity at Delta = 0 is c = seg_var - fused_var, never zero
    assert np.allclose(sens[:500], (seg_var - fused_var)[:500])
    # a variance above any mixture is flagged and returns the parabola's maximum
    w_b, _, f_b = analysis.variance_weight(np.full(3, 50.0), np.full(3, 2.0), np.full(3, 5.0),
                                           np.full(3, 4.0))
    assert np.all(f_b == 2) and np.allclose(w_b, (16 - 3) / 32)


def test_hybrid_weight_takes_the_root_where_unique_and_the_ratio_where_not():
    """The hybrid of one channel: the variance root on Delta^2 <= c trials
    (Delta = 0 included), the channel's own position ratio on Delta^2 > c
    trials -- no hint anywhere -- and the nominal sd of each part."""
    post, fused_var, seg_var, delta, w, var_out = _mixture_observer()
    delta[:500] = 0.0
    var_out = analysis.mixture_variance(w, fused_var, seg_var, delta)
    seg_mu = np.zeros_like(delta)
    mu_out = w * delta                      # mu = w fused + (1 - w) seg with seg = 0
    w_hat, sigma, flags = analysis.hybrid_weight(var_out, fused_var, seg_var, delta, mu_out,
                                                 seg_mu, sig_out_mu=0.3, sig_out_var=0.02)
    assert np.allclose(w_hat, w, atol=1e-6)         # exact on an observer without noise
    two = delta ** 2 > seg_var - fused_var
    assert np.all(flags[two] == 1) and np.all(flags[~two] == 0)
    assert np.all(flags[:500] == 0)                 # Delta = 0 is a root trial
    # each part's nominal sd: sigma_out / |Delta| on the ratio trials, sigma_out / |dv/dw|
    # on the root trials (c at Delta = 0)
    assert np.allclose(sigma[two], 0.3 / np.abs(delta[two]))
    assert np.allclose(sigma[:500], 0.02 / (seg_var - fused_var)[:500])
    # a ratio-regime trial is read from the position output alone: a noisy
    # variance output there changes nothing, a noisy position output does
    noisy_var = var_out.copy()
    noisy_var[two] += 5.0
    w_nv, _, _ = analysis.hybrid_weight(noisy_var, fused_var, seg_var, delta, mu_out, seg_mu)
    assert np.allclose(w_nv[two], w[two], atol=1e-6)
    w_nm, _, _ = analysis.hybrid_weight(var_out, fused_var, seg_var, delta, mu_out + 0.5, seg_mu)
    assert np.allclose(w_nm[two], w[two] + 0.5 / delta[two], atol=1e-6)
    assert np.allclose(w_nm[~two], w[~two], atol=1e-6)
    # without the readout noises the sd is not given
    _, s_none, _ = analysis.hybrid_weight(var_out, fused_var, seg_var, delta, mu_out, seg_mu)
    assert np.all(np.isnan(s_none))


def test_variance_regressions_find_the_pull():
    """The variance-domain regression reads slope 1 off an optimal observer and
    below 1 off one that under-fuses; the weight regression reads the pull."""
    post, fused_var, seg_var, delta, w, var_out = _mixture_observer(pull=1.0)
    reg = analysis.variance_regression(var_out + np.random.default_rng(1).normal(0, 0.05, w.size),
                                       fused_var, seg_var, delta, post)
    assert reg["slope"] == pytest.approx(1.0, abs=0.02)
    assert reg["intercept"] == pytest.approx(0.0, abs=0.05)
    post, fused_var, seg_var, delta, w, var_out = _mixture_observer(pull=0.8)
    reg = analysis.variance_regression(var_out, fused_var, seg_var, delta, post)
    assert reg["slope"] < 0.98            # under-fusion shows as a slope below 1
    mu_out = w * delta                    # the position output with seg = 0
    w_hat, _, _ = analysis.hybrid_weight(var_out, fused_var, seg_var, delta, mu_out,
                                         np.zeros_like(delta))
    wreg = analysis.weight_regression(w_hat, post)
    assert wreg["slope"] == pytest.approx(0.8, abs=1e-6)
    assert wreg["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert wreg["n"] == w.size
