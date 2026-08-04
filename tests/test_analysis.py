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


def test_fusion_weight_recovers_the_weight_that_generated_the_estimate():
    rng = np.random.default_rng(0)
    fused, seg = rng.normal(0, 5, 2000), rng.normal(0, 5, 2000) + 20
    w_true = rng.uniform(0, 1, 2000)
    estimate = w_true * fused + (1 - w_true) * seg

    w = analysis.fusion_weight(estimate, seg, fused, min_separation=1.0)
    ok = np.isfinite(w)
    assert np.allclose(w[ok], w_true[ok], atol=1e-8)


def test_fusion_weight_drops_the_unstable_trials():
    """Where fused ~= segregated the ratio has no information; those trials must
    come back NaN rather than as huge numbers that dominate every summary."""
    fused = np.array([10.0, 0.5])
    seg = np.array([0.0, 0.0])
    w = analysis.fusion_weight(np.array([5.0, 0.4]), seg, fused, min_separation=1.0)
    assert w[0] == pytest.approx(0.5)
    assert np.isnan(w[1])


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
