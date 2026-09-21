"""The implied-weight experiment: the pieces that do arithmetic."""

import numpy as np
import pytest

from cmsi.experiments import architecture as arch
from cmsi.experiments import fixed_variance as fv
from cmsi.experiments import implied_weight as iw


def test_parse_hidden_accepts_square_and_asymmetric_specs():
    assert arch.parse_hidden("64") == [64, 64]
    assert arch.parse_hidden("32x128") == [32, 128]
    assert arch.parse_hidden("16,16") == [16, 16]
    assert arch.variant_name([32, 128]) == "h32x128"
    with pytest.raises(ValueError):
        arch.parse_hidden("1x2x3")


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


def test_fixed_variance_pins_every_range_and_labels_the_variant():
    cfg = {"seed": 0,
           "generative": {"p_common": 0.5, "sigma2_vis_range": [1.2, 6.6],
                          "sigma2_prop_range": [1.8, 9.6], "sigma2_eye_range": [3.3, 13.2],
                          "sigma0_sq": 425.0, "eye_sigma_sq": 325.0},
           "encoding": {}, "model": {"head": "causal", "hidden": [64, 64]},
           "training": {"split": [0.7, 0.15, 0.15], "optimizer": "adam", "batch_size": 256,
                        "n_trials": 100},
           "analysis": {"reliability_levels": [1.5, 3.5, 6.0]}}
    mid = fv.range_midpoints(cfg)
    assert mid == {"vis": 3.9, "prop": 5.7, "eye": 8.25}
    grid = fv.variant_grid(cfg, vis=[1.5, 6.0], prop=None, eye=[8.0])
    assert [g["vis"] for g in grid] == [1.5, 6.0]
    assert all(g["prop"] == 5.7 and g["eye"] == 8.0 for g in grid)
    pinned = fv.fixed_config(cfg, grid[0])
    assert pinned["generative"]["sigma2_vis_range"] == [1.5, 1.5]
    assert pinned["generative"]["sigma2_prop_range"] == [5.7, 5.7]
    assert pinned["generative"]["sigma2_eye_range"] == [8.0, 8.0]
    assert pinned["analysis"]["reliability_levels"] == [1.5]
    assert cfg["generative"]["sigma2_vis_range"] == [1.2, 6.6]      # the base is untouched
    assert fv.variant_label(grid[0]) == "v1.5_p5.7_e8"


# --------------------------------------------------------------------------- #
# readability, design, sweep
# --------------------------------------------------------------------------- #
def test_predicted_outside_matches_a_simulation_of_p_plus_noise_over_delta():
    rng = np.random.default_rng(1)
    n = 200000
    delta = rng.normal(0, 3, n)
    post = rng.uniform(0, 1, n)
    for sd in (0.3, 1.0):
        w = post + rng.normal(0, sd, n) / delta
        observed = np.mean((w < -1) | (w > 2))
        assert abs(iw.predicted_outside(delta, post, sd) - observed) < 0.004
    assert iw.predicted_outside(delta, post, 0.0) == 0.0
    # importance weights that pick out the large-|Delta| half must lower it
    heavy = (np.abs(delta) > 3).astype(float)
    assert iw.predicted_outside(delta, post, 1.0, weights=heavy) \
        < iw.predicted_outside(delta, post, 1.0)


def test_flat_posterior_weights_flatten_the_histogram():
    rng = np.random.default_rng(2)
    post = np.clip(rng.beta(0.4, 0.4, 5000), 0, 1)        # bimodal, like the task
    w = iw.flat_posterior_weights(post, n_bins=10)
    assert w.sum() == pytest.approx(1.0)
    weighted, _ = np.histogram(post, bins=10, range=(0, 1), weights=w)
    assert np.allclose(weighted, 0.1, atol=1e-9)


def test_weight_analysis_carries_a_readability_block_that_matches_its_arrays():
    pred, d, names = _synthetic(pull=1.0)
    sig_out = np.array([0.5, 0.05, 0.5, 0.05])
    acfg = {"disparity_grid": iw.GRID, "sigma_w_criterion": 0.1, "min_separation": 1.0}
    m, arrays = iw.weight_analysis(pred, d, names, sig_out, acfg)
    for key in ("vis", "prop"):
        rb = m["reads"][key]["readability"]
        w = arrays[key]["w_raw"]
        ok = np.isfinite(w)
        assert rb["outside"] == pytest.approx(np.mean(ok & ((w < -1) | (w > 2))))
        assert rb["frac_absdelta_lt1"] == pytest.approx(np.mean(np.abs(arrays[key]["delta"]) < 1))
        # an optimal network with Gaussian noise: the prediction from its own
        # sd is close to what it shows (the synthetic error is Gaussian and
        # independent of Delta, so the two agree to within sampling error)
        assert abs(rb["outside_predicted_own_sd"] - rb["outside"]) < 0.02
        assert rb["own_sd"] == pytest.approx(0.5, abs=0.03)
        assert 0 <= rb["outside_if_flat_posterior"] <= 1


def test_balance_posterior_keeps_the_requested_fraction_and_flattens():
    from cmsi.experiments import design

    rng = np.random.default_rng(3)
    n = 20000
    post = np.clip(rng.beta(0.4, 0.4, n), 0, 1)
    d = {"X": rng.normal(size=(n, 3)), "post_c1": post, "C": rng.integers(1, 3, n),
         "rejection_rate": np.array([0.0]), "encoders": {"a": np.ones(3)}}
    out, info = design.balance_posterior(d, 0.5, seed=0)
    assert abs(info["kept"] - 0.5) < 0.03
    assert len(out["post_c1"]) == info["n_after"] == len(out["X"])
    assert max(info["posterior_hist_after"]) < max(info["posterior_hist_before"])
    assert out["encoders"] is d["encoders"] and out["rejection_rate"].size == 1
    # kept trials are a subset of the original ones, untouched
    rows = {row.tobytes() for row in d["X"]}
    assert all(row.tobytes() in rows for row in out["X"])
    full, _ = design.balance_posterior(d, 1.0, seed=0)
    assert len(full["post_c1"]) == n


def test_balance_posterior_accepts_by_posterior_bin_and_not_by_cause(cfg):
    """The acceptance must be blind to C given the posterior bin. A generator
    seeded like the sampler's would reproduce the uniforms that decided C,
    and accept trials by their true cause -- this is the check for that."""
    from cmsi.data import make_dataset
    from cmsi.experiments import design

    d = make_dataset(cfg, n=6000)                      # drawn from default_rng(cfg["seed"])
    out, info = design.balance_posterior(d, 0.5, seed=cfg["seed"])
    post, C = d["post_c1"], d["C"]
    idx = np.clip((post * 10).astype(int), 0, 9)
    kept = np.zeros(len(post), bool)
    rows = {row.tobytes() for row in out["X"]}
    kept[[i for i, row in enumerate(d["X"]) if row.tobytes() in rows]] = True
    for b in range(10):
        m = idx == b
        if m.sum() < 100:
            continue
        assert abs(kept[m].mean() - info["acceptance_by_bin"][b]) < 0.08
        for c in (1, 2):
            if (m & (C == c)).sum() >= 50:
                assert abs(kept[m & (C == c)].mean() - kept[m].mean()) < 0.12


def test_design_analysis_screens_a_config_without_training(cfg):
    from cmsi.experiments import design

    stats, arrays = design.design_analysis(cfg, n=3000, floor=True)
    assert stats["n"] == 3000
    p = stats["posterior"]
    assert 0 <= p["mass_intermediate"] <= 1 and len(p["hist"]) == 10
    assert 0 < p["at_zero_disparity"]["low_noise"] < 1
    for key in ("vis", "prop"):
        a = arrays[key]
        assert np.all(np.diff(a["outside"]) >= -1e-12)        # more error, more spikes
        assert np.all(np.diff(a["readable"]) <= 1e-12)        # more error, fewer readable
        assert a["outside_flat"].shape == a["sd"].shape
        assert 0 <= stats["reads"][key]["frac_absdelta_lt1"] <= 1
    fl = stats["floor"]
    assert 0 < fl["fused_mu"] < 5 and fl["n"] == 3000
    assert all(v > 0 for v in fl["decode_sd"].values())
    balanced, _ = design.design_analysis(cfg, n=2000, balance=0.5, floor=False)
    assert "balance" in balanced and abs(balanced["balance"]["kept"] - 0.5) < 0.05
    assert "max_auc_deviation" in balanced["balance"]["anticonfound"]


def test_sweep_labels_and_configs():
    from cmsi.experiments import sweep
    from cmsi.utils import load_config

    assert sweep.value_label(100) == "100" and sweep.value_label(1.5) == "1.5"
    assert sweep.value_label([2, 2.5]) == "2-2.5" and sweep.value_label([128, 128]) == "128-128"
    assert sweep.variant_label("sigma0_sq", 100) == "sigma0_sq=100"
    assert sweep.sort_key([2, 4]) == 3.0 and sweep.sort_key(7) == 7.0
    cfg = load_config()
    v = sweep.variant_config(cfg, "sigma0_sq", 169)
    assert v["generative"]["sigma0_sq"] == 169 and cfg["generative"]["sigma0_sq"] != 169
    with pytest.raises(KeyError):
        sweep.variant_config(cfg, "no_such_key", 1)


# --------------------------------------------------------------------------- #
# per-trial weights from both position outputs and from the variance outputs
# --------------------------------------------------------------------------- #
def test_variance_weight_inverts_the_mixture_variance_exactly():
    rng = np.random.default_rng(4)
    n = 5000
    w = rng.uniform(0, 1, n)
    fused_var = rng.uniform(1.5, 3.0, n)
    seg_var = fused_var + rng.uniform(1.0, 6.0, n)            # c = seg - fused > 0
    delta = rng.normal(0, 4, n)
    delta[:500] = 0.0                                          # the Delta = 0 limit
    v = w * fused_var + (1 - w) * seg_var + w * (1 - w) * delta ** 2
    w_hat, sens, flags = iw.variance_weight(v, fused_var, seg_var, delta, hint=w)
    assert np.allclose(w_hat, w, atol=1e-6)
    assert np.all(flags[:500] == 0)
    two = delta ** 2 > seg_var - fused_var
    assert np.all(flags[two] == 1) and np.all(flags[~two & (delta != 0)] == 0)
    assert np.allclose(sens, np.abs((delta ** 2 - (seg_var - fused_var)) - 2 * delta ** 2 * w))
    # without a hint the unique-root trials are still exact
    w_nohint, _, _ = iw.variance_weight(v, fused_var, seg_var, delta)
    assert np.allclose(w_nohint[~two], w[~two], atol=1e-6)
    # a variance above any mixture is flagged and returns the parabola's maximum
    big = np.full(3, 50.0)
    w_b, _, f_b = iw.variance_weight(big, np.full(3, 2.0), np.full(3, 5.0), np.full(3, 4.0))
    assert np.all(f_b == 2) and np.allclose(w_b, (16 - 3) / 32)


def _synthetic_full(n=6000, pull=0.85, noise=0.3, seed=0):
    """An observer that mixes BOTH outputs with the weight pull * posterior,
    variances included, plus read-out noise on every output."""
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


def test_per_trial_weights_read_the_network_weight_from_the_variance_outputs():
    pred, d, names, w_net = _synthetic_full()
    sig_out = np.array([0.3, 0.05, 0.3, 0.05])
    m, arrays = iw.per_trial_weights(pred, d, names, sig_out, crit=0.1)
    est = m["estimators"]
    # the variance read recovers the weight the network used on (nearly) every
    # trial, the ratio does not
    w_var = arrays["variance"]["w"]
    ok = np.isfinite(w_var)
    assert ok.mean() > 0.99
    assert np.std((w_var - w_net)[ok]) < 0.03
    assert est["variance"]["outside"] < 0.002
    assert est["ratio_vis"]["outside"] > max(0.005, 5 * est["variance"]["outside"])
    assert est["variance"]["frac_sigma_w_lt_crit"] > 0.95
    # the joint position read is tighter than either single ratio
    assert est["joint"]["sd_vs_posterior"] < est["ratio_vis"]["sd_vs_posterior"]
    # coherent observer: the two reads agree where the position read is precise
    co = m["coherence"]
    assert abs(co["mean_diff_variance_minus_joint"]) < 0.02
    assert co["sd_diff_variance_minus_joint"] < 0.06
    # and both see the pull (0.85 of the posterior), not the posterior itself
    bp = est["variance"]["by_posterior"]
    top = [w for c, w in zip(bp["centres"], bp["mean"], strict=False) if c > 0.8]
    assert top and all(0.6 < t < 0.8 for t in top)
    # the whole thing is also produced by weight_analysis, with the two
    # no-division regressions: in the variance domain (slope 1 = optimal
    # mixture variance) and of the variance read on the posterior (slope =
    # the pull, 0.85 here)
    acfg = {"disparity_grid": iw.GRID, "sigma_w_criterion": 0.1, "min_separation": 1.0,
            "reliability_levels": [1.5, 3.5, 6.0]}
    mm, aa = iw.weight_analysis(pred, d, names, sig_out, acfg)
    assert "per_trial" in mm and "per_trial" in aa
    assert abs(mm["per_trial"]["weight_regression"]["variance"]["slope"] - 0.85) < 0.03
    assert abs(mm["per_trial"]["weight_regression"]["variance"]["intercept"]) < 0.02
    for key in ("vis", "prop"):
        assert 0.7 < mm["per_trial"]["variance_regression"][key]["slope"] < 1.0
    # and the pipeline's figures 04 and 08 exist in their variance-read form
    figs = iw.baseline_figures(pred, d, names, sig_out, acfg, aa, mm)
    assert {"04v_fusion_weight_variance_read", "08v_variance_regression"} <= set(figs)
    import matplotlib.pyplot as plt
    plt.close("all")
