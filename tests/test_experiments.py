"""The implied-weight experiment: the pieces that do arithmetic."""

import numpy as np
import pytest

from cmsi.experiments import implied_weight as iw


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


def _synthetic(n=6000, pull=1.0, noise=0.3, seed=0):
    """A model-averaging observer that mixes both channels with the weight
    pull * posterior, means and variances included, plus read-out noise on
    every output; the analytical targets alongside."""
    rng = np.random.default_rng(seed)
    disp = rng.normal(0, 12, n)
    post = 1 / (1 + np.exp(0.4 * (np.abs(disp) - 8)))
    seg_v = rng.normal(0, 10, n)
    fused = seg_v + 0.6 * disp
    seg_p = fused - 0.4 * disp
    fused_var = rng.uniform(2.0, 3.0, n)
    seg_v_var, seg_p_var = fused_var + rng.uniform(3, 6, n), fused_var + rng.uniform(1.5, 3, n)
    w_net = pull * post
    dv, dp = fused - seg_v, fused - seg_p
    names = ["mu_vis", "var_vis", "mu_prop", "var_prop"]
    d = {"disparity": disp, "post_c1": post, "fused_mu": fused, "fused_var": fused_var,
         "seg_vis_mu": seg_v, "seg_prop_mu": seg_p, "seg_vis_var": seg_v_var,
         "seg_prop_var": seg_p_var,
         "mu_vis": seg_v + post * dv, "mu_prop": seg_p + post * dp,
         "var_vis": post * fused_var + (1 - post) * seg_v_var + post * (1 - post) * dv ** 2,
         "var_prop": post * fused_var + (1 - post) * seg_p_var + post * (1 - post) * dp ** 2,
         "sig2_vis": rng.uniform(1, 6, n), "sig2_prop": rng.uniform(2, 9, n),
         "sig2_eye": rng.uniform(3, 13, n)}
    pred = np.stack([
        seg_v + w_net * dv + rng.normal(0, noise, n),
        w_net * fused_var + (1 - w_net) * seg_v_var + w_net * (1 - w_net) * dv ** 2
        + rng.normal(0, 0.05, n),
        seg_p + w_net * dp + rng.normal(0, noise, n),
        w_net * fused_var + (1 - w_net) * seg_p_var + w_net * (1 - w_net) * dp ** 2
        + rng.normal(0, 0.05, n)], 1)
    return pred, d, names, w_net


SIG_OUT = np.array([0.3, 0.05, 0.3, 0.05])
ACFG = {"disparity_grid": iw.GRID, "reliability_levels": [1.5, 3.5, 6.0]}


def test_weight_analysis_reads_the_pull_on_every_trial():
    """The hybrid read of each channel exists on every trial, is the variance
    root where Delta^2 <= c and the channel's ratio where Delta^2 > c, and
    every summary sees the pull (0.85 of the posterior), not the posterior."""
    pred, d, names, w_net = _synthetic(pull=0.85)
    m, arrays = iw.weight_analysis(pred, d, names, SIG_OUT, ACFG)
    for key, mu, var, _seg_mu, seg_var in iw.CHANNELS:
        r, a = m["reads"][key], arrays[key]
        assert r["n"] == len(pred) and np.all(np.isfinite(a["w"]))
        two = a["delta"] ** 2 > d[seg_var] - d["fused_var"]
        assert np.all(a["flags"][two] == 1) and np.all(a["flags"][~two] != 1)
        assert r["frac_position_ratio"] == pytest.approx(two.mean())
        assert r["sd_vs_posterior_root"] < r["sd_vs_posterior_ratio"] < 0.2
        assert abs(r["weight_regression"]["slope"] - 0.85) < 0.03
        assert abs(r["position_regression"]["slope"] - 0.85) < 0.03
        assert 0.7 < r["variance_regression"]["slope"] < 1.0
        assert r["outside"] < 0.01
        assert r["output"] == f"{var} root / {mu} ratio"
        assert np.all(np.diff(r["by_posterior"]["centres"]) > 0)
        top = [w for c, w in zip(r["by_posterior"]["centres"], r["by_posterior"]["w"],
                                 strict=False) if c > 0.8]
        assert top and all(0.6 < t < 0.8 for t in top)
        assert len(r["curve"]["centres"]) == len(r["curve"]["se"]) > 10
        assert np.isfinite(r["midpoint_deg"])
    assert m["consistency"]["n"] == len(pred) and m["consistency"]["corr"] > 0.9
    assert abs(m["analytical"]["midpoint_deg"] - 8.0) < 0.5
    assert set(m["sigma_out"]) == set(names)


def test_reliability_split_returns_a_curve_and_a_midpoint_per_level():
    pred, d, names, _ = _synthetic(n=8000)
    curves = iw.weight_by_reliability(pred, d, names, SIG_OUT, ACFG, "vis", 4)
    assert len(curves) == 4
    for e in curves.values():
        assert e["n"] >= 1900
        assert set(e["midpoints"]) == {"analytical", "vis", "prop"}
        assert len(e["vis"][0]) == len(e["vis"][1]) == len(e["vis"][3])
        assert np.all(np.isfinite(e["vis"][1]))


def test_baseline_figures_are_the_pipelines_weight_figures():
    pred, d, names, _ = _synthetic()
    m, arrays = iw.weight_analysis(pred, d, names, SIG_OUT, ACFG)
    figs = iw.baseline_figures(pred, d, names, SIG_OUT, ACFG, arrays, m)
    assert set(figs) == {"04_fusion_weight", "05_fusion_weight_by_reliability",
                         "08v_variance_regression", "15_weight_vs_posterior",
                         "16_weight_distribution"}
    assert len(figs["08v_variance_regression"].axes) == 2
    lines = figs["04_fusion_weight"].axes[0].get_lines()
    assert sum("implied" in line.get_label() for line in lines) == 2
    rfigs, table = iw.reliability_figures(pred, d, names, SIG_OUT, ACFG, ["vis"], 3)
    assert set(rfigs) == {"01a_weight_by_sig2_vis", "01b_midpoint_by_sig2_vis"}
    assert set(table) == {"sig2_vis"} and len(table["sig2_vis"]) == 3
    import matplotlib.pyplot as plt
    plt.close("all")
