"""Tests added by the design audit (2026-08-23).

The one that certifies SS5 is `test_targets_match_brute_force_integration`:
the closed-form observer against a numerical integration of the full posterior
over (C, source, eye). The rest guard the new mechanisms: range containment,
the stratified split, the SS7.1 statistics, the SS7.3 variance signature, the
SS7.4 model comparison, and the SS7.6 behavior curves.
"""

import numpy as np
import pytest

from cmsi import analysis
from cmsi.data.dataset import make_dataset, split_indices
from cmsi.data.generative import containment_bounds, observer, sample_trials


# --------------------------------------------------------------------------- #
# SS5 -- the mandatory brute-force certification
# --------------------------------------------------------------------------- #
def test_targets_match_brute_force_integration(cfg, rng):
    """Closed-form w, means, and variances == numerical posterior over
    (C, s, e). Certifies the whole of SS5 (design: 'mandatory unit test')."""
    gen = cfg["generative"]
    n = 20
    d = sample_trials(n, gen, rng)
    code = observer(d, gen)

    mu0, s0 = gen["mu0"], gen["sigma0_sq"]
    mu_e, se0 = gen["eye_mu"], gen["eye_sigma_sq"]
    gs = np.linspace(mu0 - 6 * np.sqrt(s0), mu0 + 6 * np.sqrt(s0), 601)
    ge = np.linspace(mu_e - 6 * np.sqrt(se0), mu_e + 6 * np.sqrt(se0), 601)
    ds_, de_ = gs[1] - gs[0], ge[1] - ge[0]

    def norm(x, m, v):
        return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)

    prior_s, prior_e = norm(gs, mu0, s0), norm(ge, mu_e, se0)
    S, E = np.meshgrid(gs, ge, indexing="ij")
    for i in range(n):
        sv, sp, sE = d["sig2_vis"][i], d["sig2_prop"][i], d["sig2_eye"][i]
        xv, xp, xE = d["x_vis"][i], d["x_prop"][i], d["x_eye"][i]

        vis_eye = norm(xv, S - E, sv) * norm(xE, E, sE) * prior_e[None, :]
        J1 = vis_eye * (prior_s * norm(xp, gs, sp))[:, None]
        L1 = J1.sum() * ds_ * de_
        ps1 = J1.sum(1) * de_
        m1 = (gs * ps1).sum() / ps1.sum()
        v1 = (gs ** 2 * ps1).sum() / ps1.sum() - m1 ** 2

        ps_p = norm(xp, gs, sp) * prior_s
        mp = (gs * ps_p).sum() / ps_p.sum()
        vp = (gs ** 2 * ps_p).sum() / ps_p.sum() - mp ** 2
        ps_v = (vis_eye * prior_s[:, None]).sum(1) * de_
        mv = (gs * ps_v).sum() / ps_v.sum()
        vv = (gs ** 2 * ps_v).sum() / ps_v.sum() - mv ** 2
        L2 = (ps_p.sum() * ds_) * (ps_v.sum() * ds_)

        pc = gen["p_common"]
        w = pc * L1 / (pc * L1 + (1 - pc) * L2)
        mu_prop = w * m1 + (1 - w) * mp
        var_prop = w * (v1 + m1 ** 2) + (1 - w) * (vp + mp ** 2) - mu_prop ** 2
        mu_vis = w * m1 + (1 - w) * mv
        var_vis = w * (v1 + m1 ** 2) + (1 - w) * (vv + mv ** 2) - mu_vis ** 2

        assert code["post_c1"][i] == pytest.approx(w, abs=1e-6)
        assert code["mu_prop"][i] == pytest.approx(mu_prop, abs=1e-5)
        assert code["var_prop"][i] == pytest.approx(var_prop, abs=1e-4)
        assert code["mu_vis"][i] == pytest.approx(mu_vis, abs=1e-5)
        assert code["var_vis"][i] == pytest.approx(var_vis, abs=1e-4)


# --------------------------------------------------------------------------- #
# SS3 -- range containment
# --------------------------------------------------------------------------- #
def test_containment_keeps_x_vis_inside_the_bounds(cfg, rng):
    lo, hi = -15.0, 15.0
    d = sample_trials(4000, cfg["generative"], rng, contain=(lo, hi))
    assert np.all((d["x_vis"] >= lo) & (d["x_vis"] <= hi))
    assert 0 < d["rejection_rate"][0] < 1


def test_containment_rate_is_c_symmetric(cfg):
    """Rejection must not become a C cue: the surviving x_vis distribution has
    to match across C (SS3 / failure mode #3)."""
    d = sample_trials(30000, cfg["generative"], np.random.default_rng(1),
                      contain=(-20.0, 20.0))
    c1 = d["C"] == 1
    assert abs(d["x_vis"][c1].std() - d["x_vis"][~c1].std()) < 0.3
    assert abs(d["x_vis"][c1].mean() - d["x_vis"][~c1].mean()) < 0.3


def test_dataset_records_the_rejection_rate(cfg):
    d = make_dataset(cfg, n=500)
    assert "rejection_rate" in d and d["rejection_rate"].shape == (1,)
    assert "X_clean" in d and d["X_clean"].shape == d["X"].shape


def test_containment_bounds_come_from_the_encoding(cfg):
    lo, hi = containment_bounds(cfg["encoding"])
    flo, fhi = cfg["encoding"]["visual_field"]
    m = 2 * cfg["encoding"]["rf_width"]
    assert lo == flo + m and hi == fhi - m


# --------------------------------------------------------------------------- #
# SS8.4 -- stratified split
# --------------------------------------------------------------------------- #
def test_stratified_split_is_disjoint_complete_and_covers_deciles(rng):
    post = np.random.default_rng(5).uniform(0, 1, 5000)
    splits = split_indices(5000, [0.7, 0.15, 0.15], rng, stratify=post)
    joined = np.concatenate(list(splits.values()))
    assert len(joined) == 5000 and len(np.unique(joined)) == 5000
    # every posterior decile is represented in the test split at ~its share
    edges = np.quantile(post, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, post[splits["test"]], "right") - 1, 0, 9)
    counts = np.bincount(bins, minlength=10)
    assert counts.min() > 0.5 * counts.mean()


# --------------------------------------------------------------------------- #
# SS7.1 -- position-domain regression and friends
# --------------------------------------------------------------------------- #
def test_position_regression_recovers_optimality():
    rng = np.random.default_rng(0)
    fused, seg = rng.normal(0, 5, 4000), rng.normal(0, 5, 4000)
    w = rng.uniform(0, 1, 4000)
    est = w * fused + (1 - w) * seg + rng.normal(0, 0.05, 4000)
    reg = analysis.position_regression(est, seg, fused, w)
    assert reg["slope"] == pytest.approx(1.0, abs=0.02)
    assert reg["intercept"] == pytest.approx(0.0, abs=0.02)
    assert reg["slope_ci95"][0] < 1.0 < reg["slope_ci95"][1]


def test_position_regression_detects_a_non_bayesian_weight():
    rng = np.random.default_rng(1)
    fused, seg = rng.normal(0, 5, 4000), rng.normal(0, 5, 4000)
    w = rng.uniform(0, 1, 4000)
    est = 0.5 * w * fused + (1 - 0.5 * w) * seg      # half the optimal pull
    reg = analysis.position_regression(est, seg, fused, w)
    assert reg["slope"] < 0.7


def test_joint_weight_pools_both_outputs():
    rng = np.random.default_rng(2)
    fused = rng.normal(0, 5, 2000)
    seg_v, seg_p = fused + rng.normal(0, 8, 2000), fused - rng.normal(0, 8, 2000)
    w = rng.uniform(0, 1, 2000)
    est_v = w * fused + (1 - w) * seg_v
    est_p = w * fused + (1 - w) * seg_p
    wj = analysis.joint_fusion_weight(est_v, seg_v, est_p, seg_p, fused)
    ok = np.isfinite(wj)
    assert np.allclose(wj[ok], w[ok], atol=1e-8)


def test_sigma_w_flags_the_unreadable_trials():
    sw = analysis.sigma_w(0.5, np.array([10.0, 0.0, 1.0]))
    assert sw[0] == pytest.approx(0.05)
    assert np.isinf(sw[1])
    assert sw[2] == pytest.approx(0.5)


def test_weight_consistency_on_agreeing_readings():
    w = np.random.default_rng(3).uniform(0, 1, 500)
    out = analysis.weight_consistency(w, w + 0.01, np.full(500, 0.01),
                                      np.full(500, 0.01))
    assert out["corr"] > 0.99 and out["mean_abs_diff"] < 0.02


# --------------------------------------------------------------------------- #
# SS7.2 -- reliability within disparity
# --------------------------------------------------------------------------- #
def test_reliability_test_separates_bayes_from_heuristic():
    rng = np.random.default_rng(4)
    disp = rng.uniform(0, 30, 6000)
    w_opt = np.clip(1 / (1 + np.exp(0.3 * (disp - 10)))
                    + rng.normal(0, 0.15, 6000), 0, 1)  # reliability jitter
    bayes = w_opt + rng.normal(0, 0.05, 6000)
    heuristic = 1 / (1 + np.exp(0.3 * (disp - 10)))     # disparity only
    r_bayes = analysis.reliability_within_disparity(bayes, w_opt, disp)
    r_heur = analysis.reliability_within_disparity(heuristic, w_opt, disp)
    assert r_bayes["combined_slope"] > 0.8
    assert abs(r_heur["combined_slope"]) < 0.2


# --------------------------------------------------------------------------- #
# SS7.3 -- variance signature
# --------------------------------------------------------------------------- #
def test_variance_signature_finds_the_hump():
    rng = np.random.default_rng(5)
    n = 20000
    post = rng.uniform(0, 1, n)
    fused_mu, seg_mu = rng.normal(0, 3, n), rng.normal(0, 3, n) + 10
    fused_var, seg_var = np.full(n, 2.0), np.full(n, 5.0)
    mix = post * (fused_var + fused_mu ** 2) + (1 - post) * (seg_var + seg_mu ** 2) \
        - (post * fused_mu + (1 - post) * seg_mu) ** 2
    sig = analysis.variance_signature(mix, post, fused_mu, fused_var,
                                      seg_mu, seg_var)
    assert sig["hump_net"] > 0, "mixture variance must be elevated mid-ambiguity"
    assert sig["hump_analytical"] == pytest.approx(sig["hump_net"], rel=0.05)
    # the fixed-weight comparator has no hump
    c = sig["centres"]
    mid = (c > 0.2) & (c < 0.8)
    assert sig["fixed"][mid].mean() - sig["fixed"][~mid].mean() < 0.5


# --------------------------------------------------------------------------- #
# SS7.4 -- model comparison
# --------------------------------------------------------------------------- #
def test_model_comparison_identifies_each_planted_strategy():
    rng = np.random.default_rng(6)
    n = 8000
    post = rng.uniform(0, 1, n)
    fused, seg = rng.normal(0, 3, n), rng.normal(0, 3, n) + 12
    planted = {
        "averaging": post * fused + (1 - post) * seg,
        "integration": fused,
        "segregation": seg,
        "selection": np.where(post > 0.5, fused, seg),
    }
    for name, est in planted.items():
        mc = analysis.model_comparison(est + rng.normal(0, 0.05, n),
                                       post, fused, seg)
        assert mc["best"] == name, f"planted {name}, got {mc['best']}"


def test_per_decile_weight_is_sigmoid_for_averaging_and_a_step_for_selection():
    """The SS7.4 discrimination: averaging tracks the posterior smoothly across
    deciles, selection jumps at p = 0.5."""
    rng = np.random.default_rng(7)
    n = 40000
    post = rng.uniform(0, 1, n)
    fused, seg = rng.normal(0, 3, n), rng.normal(0, 3, n) + 12

    avg = analysis.model_comparison(post * fused + (1 - post) * seg,
                                    post, fused, seg)
    w_avg = np.asarray(avg["bin_weight_net"])
    c = np.asarray(avg["bin_centres"])
    assert np.allclose(w_avg, c, atol=0.05), "averaging must track the posterior"
    assert np.abs(np.diff(w_avg)).max() < 0.3, "averaging must be smooth"

    sel = analysis.model_comparison(np.where(post > 0.5, fused, seg),
                                    post, fused, seg)
    w_sel = np.asarray(sel["bin_weight_net"])
    assert np.abs(np.diff(w_sel)).max() > 0.5, "selection must show a step"


# --------------------------------------------------------------------------- #
# SS7.6 -- behavior curves
# --------------------------------------------------------------------------- #
def test_bias_curve_matches_the_optimal_prediction_when_optimal():
    rng = np.random.default_rng(8)
    n = 20000
    disp = rng.normal(0, 10, n)
    post = 1 / (1 + np.exp(0.3 * (np.abs(disp) - 10)))
    seg = rng.normal(0, 3, n)
    fused = seg + 0.6 * disp
    est = post * fused + (1 - post) * seg
    out = analysis.bias_vs_disparity(est, seg, disp,
                                     [-20, -10, -5, 0, 5, 10, 20],
                                     prediction=post * (fused - seg))
    assert np.allclose(out["bias_net"], out["bias_opt"], atol=1e-8)


def test_conditioned_bias_shows_truncation_for_inferred_separate():
    """Conditioning on 'inferred two causes' selects noise-exaggerated
    disparities -- the separate branch must sit below the common branch."""
    rng = np.random.default_rng(9)
    n = 40000
    disp = rng.normal(0, 8, n)
    post = 1 / (1 + np.exp(0.5 * (np.abs(disp) - 6)))
    seg = np.zeros(n)
    fused = 0.5 * disp
    est = post * fused + (1 - post) * seg
    out = analysis.conditioned_bias(est, seg, disp, post > 0.5,
                                    [0, 2, 4, 6, 8, 12])
    common = dict(zip(out["common"]["centres"], out["common"]["bias"], strict=True))
    sep = dict(zip(out["separate"]["centres"], out["separate"]["bias"], strict=True))
    shared = sorted(set(common) & set(sep))
    assert shared, "need overlapping bins to compare"
    assert np.mean([common[c] - sep[c] for c in shared]) > 0
