"""The implied-weight experiment: the pieces that do arithmetic."""

import numpy as np
import pytest

from cmsi.experiments import implied_weight as iw


def test_parse_hidden_accepts_square_and_asymmetric_specs():
    assert iw.parse_hidden("64") == [64, 64]
    assert iw.parse_hidden("32x128") == [32, 128]
    assert iw.parse_hidden("16,16") == [16, 16]
    assert iw.variant_name([32, 128]) == "h32x128"
    with pytest.raises(ValueError):
        iw.parse_hidden("1x2x3")


def test_parse_overrides_reads_yaml_values():
    out = iw.parse_overrides(["rf_width=4", "n_vis=100", "sigma2_vis_range=[1,4]"])
    assert out == {"rf_width": 4, "n_vis": 100, "sigma2_vis_range": [1, 4]}
    with pytest.raises(ValueError):
        iw.parse_overrides(["rf_width"])


def test_quantile_levels_hold_equal_counts_and_centre_levels_take_nearest():
    rng = np.random.default_rng(0)
    v = rng.uniform(1, 7, 1000)
    idx, info = iw.assign_levels(v, 5)
    assert len(info) == 5
    assert np.ptp(np.bincount(idx, minlength=5)) <= 1
    idx, info = iw.assign_levels(v, [1.5, 4.0, 6.5])
    assert [c for _, c in info] == [1.5, 4.0, 6.5]
    assert np.all(idx[v < 2.75] == 0) and np.all(idx[v > 5.25] == 2)


def test_on_grid_leaves_gaps_where_bins_were_dropped():
    grid = [-3, -1, 1, 3]
    g, w = iw._on_grid(grid, [-3, 1], [0.1, 0.9])
    assert np.array_equal(g, grid)
    assert w[0] == 0.1 and w[2] == 0.9 and np.isnan(w[1]) and np.isnan(w[3])


def _synthetic(n=4000, pull=1.0, seed=0):
    """A Bayes-optimal observer's outputs (pull = 1) with read-out noise."""
    rng = np.random.default_rng(seed)
    disp = rng.normal(0, 12, n)
    post = 1 / (1 + np.exp(0.4 * (np.abs(disp) - 8)))
    seg_v = rng.normal(0, 10, n)
    fused = seg_v + 0.6 * disp
    seg_p = fused - 0.4 * disp
    names = ["mu_vis", "var_vis", "mu_prop", "var_prop"]
    d = {"disparity": disp, "post_c1": post, "fused_mu": fused,
         "seg_vis_mu": seg_v, "seg_prop_mu": seg_p,
         "mu_vis": seg_v + post * (fused - seg_v),
         "mu_prop": seg_p + post * (fused - seg_p),
         "var_vis": 5.0 + 0.1 * post, "var_prop": 3.0 + 0.1 * post,
         "sig2_vis": rng.uniform(1, 6, n), "sig2_prop": rng.uniform(2, 9, n),
         "sig2_eye": rng.uniform(3, 13, n)}
    pred = np.stack([seg_v + pull * post * (fused - seg_v), d["var_vis"],
                     seg_p + pull * post * (fused - seg_p), d["var_prop"]], 1)
    pred[:, 0] += rng.normal(0, 0.5, n)
    pred[:, 2] += rng.normal(0, 0.5, n)
    return pred, d, names


def test_weight_analysis_recovers_the_pull_and_the_readability():
    pred, d, names = _synthetic(pull=0.85)
    sig_out = np.array([0.5, 0.05, 0.5, 0.05])
    acfg = {"disparity_grid": iw.GRID, "sigma_w_criterion": 0.1, "min_separation": 1.0}
    m, arrays = iw.weight_analysis(pred, d, names, sig_out, acfg)
    for key in ("vis", "prop"):
        r = m["reads"][key]
        assert abs(r["position_regression"]["slope"] - 0.85) < 0.03
        assert r["delta_threshold_deg"] == pytest.approx(5.0)
        # sigma_w < 0.1 exactly when |Delta| > 5
        assert r["frac_readable"] == pytest.approx(
            np.mean(np.abs(arrays[key]["delta"]) > 5.0))
        assert np.all(np.diff(r["by_posterior"]["centres"]) > 0)
        # the unfiltered ratio exists on every trial with Delta != 0
        assert r["w_raw"]["n"] == int(np.sum(arrays[key]["delta"] != 0))
        assert abs(r["w_raw"]["median"] - np.nanmedian(arrays[key]["w_raw"])) < 1e-9
    assert np.isfinite(m["analytical"]["midpoint_deg"])
    assert abs(m["analytical"]["midpoint_deg"] - 8.0) < 0.5


def test_reliability_split_returns_a_curve_and_a_midpoint_per_level():
    pred, d, names = _synthetic(n=8000)
    sig_out = np.array([0.5, 0.05, 0.5, 0.05])
    acfg = {"disparity_grid": iw.GRID}
    curves = iw.weight_by_reliability(pred, d, names, sig_out, acfg, "vis", 4)
    assert len(curves) == 4
    for e in curves.values():
        assert e["n"] >= 1900
        assert set(e["midpoints"]) == {"analytical", "vis", "prop"}
        assert len(e["vis"][0]) == len(e["vis"][1]) == len(e["vis"][3])
