"""The transformation -> causal inference statistics, one network at a time.

`analyse(b)` takes a bundle from inv_common.load_run (a causal network) and
returns one nested dict: the channels of the investigation plan
(claude/investigation_plan_transformation_to_causality_2026-10-13.md):

    R0        the network against its own observer (the repo's statistics)
    R1        the transformation-free observer (true eye) on the same trials
    terr      the transformation's error t_err = eye_hat - eye and what it does
              to the causal decision (channel C, natural noise; channel F by |eye|)
    frame     the frame the decision is made in (channel D)
    partials  effective eye gain, leakage into the hand estimate, shared error
              (channels D, E; noiseless partial derivatives)
    eps       injected eye-signal errors (channel C)
    beta      eye-signal gain errors (channel C)
    silence   the eye signal replaced by its prior mean (channel C)
    eyerel    the eye signal's reliability as a trial-by-trial cue (channel B)
    units     congruent / opposite / mixed classes and a light displacement
              index by class (channel G)

Every number is computed on the network's own stored test split.
"""

import numpy as np
from inv_common import (
    FAR,
    NEAR,
    TRUE_NEAR,
    binned_mean,
    crossing,
    eye_gain_k,
    hybrid_reads,
    lookup_r2,
    lstsq_coefs,
    model_comparison,
    observer_on,
    peak_location,
    position_regression,
    predict_on,
    transformation_error,
    transition_fit,
    trial_dict,
    variance_signature,
)

from cmsi.analysis.causal import compare, reliability_within_disparity  # noqa: E402

EPS_LEVELS = (-15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0)
BETA_LEVELS = (0.5, 0.75, 1.0, 1.25)
TERR_EDGES = np.array([0, 1, 2, 3, 6, 30], float)
EYE_EDGES = np.array([0, 3, 6, 9, 12, 60], float)
TRUE_GRID = np.arange(-31, 32, 2.0)


def _acc(w, C):
    ok = np.isfinite(w)
    return float(np.mean((w[ok] > 0.5) == (C[ok] == 1)))


def _fs(d_abs, w):
    ok = np.isfinite(w)
    return (float(np.nanmean(w[ok & (d_abs <= NEAR)])), float(1 - np.nanmean(w[ok & (d_abs >= FAR)])))


def _basic(b):
    gen = b["cfg"]["generative"]
    s2e = float(np.mean(gen["sigma2_eye_range"]))
    var_eye = 1.0 / (1.0 / s2e + 1.0 / gen["eye_sigma_sq"])
    hist = b["ckpt"].get("history", {})
    return {"name": b["name"], "head": b["head"], "n_test": b["n_test"],
            "eye_sigma_sq": float(gen["eye_sigma_sq"]), "eye_sd": float(np.sqrt(gen["eye_sigma_sq"])),
            "eye_mu": float(gen["eye_mu"]), "sigma2_eye_range": [float(v) for v in gen["sigma2_eye_range"]],
            "eye_noise_sd": float(np.sqrt(s2e)), "k": float(var_eye / s2e), "var_eye": float(var_eye),
            "var_vis_body": float(np.mean(gen["sigma2_vis_range"]) + var_eye),
            "p_common": float(gen["p_common"]),
            "best_epoch": int(hist.get("best_epoch", -1)) if isinstance(hist, dict) else -1,
            "best_val": float(hist.get("best_val", np.nan)) if isinstance(hist, dict) else np.nan,
            "rejection_rate": float(np.asarray(b["d_full"].get("rejection_rate", [0.0])).ravel()[0])}


# --------------------------------------------------------------------------- #
def r0_block(b, pred, w, wp, obs):
    d, n = b["d"], b["names"]
    iv, ip, ivv = n.index("mu_vis"), n.index("mu_prop"), n.index("var_vis")
    post = obs["post_c1"]
    ad = np.abs(obs["disparity"])
    cmp_v = compare(post, w)
    reg_v = position_regression(pred[:, iv], obs["seg_vis_mu"], obs["fused_mu"], post)
    reg_p = position_regression(pred[:, ip], obs["seg_prop_mu"], obs["fused_mu"], post)
    mid_n, k_n = transition_fit(ad, w)
    mid_o, k_o = transition_fit(ad, post)
    vs = variance_signature(pred[:, ivv], post, obs["fused_mu"], obs["fused_var"], obs["seg_vis_mu"], obs["seg_vis_var"])
    mc = model_comparison(pred[:, iv], post, obs["fused_mu"], obs["seg_vis_mu"])
    f_n, s_n = _fs(ad, w)
    f_o, s_o = _fs(ad, post)
    target = np.stack([d[k] for k in n], axis=1)
    C = d["C"]
    return {"weight_vs_post_slope": cmp_v["slope"], "weight_vs_post_r2": cmp_v["r2"],
            "posreg_slope_vis": reg_v["slope"], "posreg_slope_prop": reg_p["slope"],
            "midpoint_net": mid_n, "sharpness_net": k_n, "midpoint_R0": mid_o, "sharpness_R0": k_o,
            "crossing_net": crossing(obs["disparity"], w), "crossing_R0": crossing(obs["disparity"], post),
            "fusion_net": f_n, "segregation_net": s_n, "fusion_R0": f_o, "segregation_R0": s_o,
            "hump_net": float(vs.get("hump_net", np.nan)), "hump_R0": float(vs.get("hump_analytical", np.nan)),
            "best_strategy": mc["best"], "rmse_averaging": mc["overall"]["averaging"],
            "rmse_selection": mc["overall"]["selection"], "rmse_fixed": mc["overall"]["fixed"],
            "rmse_mu_vis": float(np.sqrt(np.mean((pred[:, iv] - target[:, iv]) ** 2))),
            "rmse_mu_prop": float(np.sqrt(np.mean((pred[:, ip] - target[:, ip]) ** 2))),
            "accuracy_net": _acc(w, C), "accuracy_R0": _acc(post, C),
            "accuracy_net_hand": _acc(wp, C),
            "agreement_net_R0": float(np.mean((w > 0.5) == (post > 0.5))),
            "mean_post": float(post.mean()), "realized_c1": float(np.mean(C == 1))}


def r1_block(b, w, obs):
    """The transformation-free observer R1 on the same trials."""
    d, gen = b["d"], b["cfg"]["generative"]
    t = trial_dict(d)
    free = observer_on(t, gen, eye_known=True)
    post, pf, C = obs["post_c1"], free["post_c1"], d["C"]
    flip = (post > 0.5) != (pf > 0.5)
    ok = np.isfinite(w)
    near = np.abs(d["s_vis"] - d["s_prop"]) <= TRUE_NEAR
    mid_f, k_f = transition_fit(np.abs(free["disparity"]), pf)
    return {"accuracy_R1": _acc(pf, C), "midpoint_R1": mid_f, "sharpness_R1": k_f,
            "crossing_R1": crossing(free["disparity"], pf),
            "flip_rate_R0_vs_R1": float(flip.mean()),
            "net_sides_with_R0_on_flips": float(np.mean((w[flip & ok] > 0.5) == (post[flip & ok] > 0.5))),
            "net_sides_with_R1_on_flips": float(np.mean((w[flip & ok] > 0.5) == (pf[flip & ok] > 0.5))),
            "post_true_common_R0": float(post[near].mean()), "post_true_common_R1": float(pf[near].mean()),
            "w_true_common_net": float(np.nanmean(w[near])),
            "fused_visual_weight_R0": float(np.mean((1 / obs["var_vis_body"]) / (1 / obs["var_vis_body"] + 1 / d["sig2_prop"] + 1 / gen["sigma0_sq"]))),
            "fused_visual_weight_R1": float(np.mean((1 / d["sig2_vis"]) / (1 / d["sig2_vis"] + 1 / d["sig2_prop"] + 1 / gen["sigma0_sq"]))),
            "_post_free": pf}


def terr_block(b, pred, w, obs, post_free):
    d, gen, n = b["d"], b["cfg"]["generative"], b["names"]
    iv, ip = n.index("mu_vis"), n.index("mu_prop")
    t_err, k = transformation_error(d, gen)
    post, C = obs["post_c1"], d["C"]
    near = np.abs(d["s_vis"] - d["s_prop"]) <= TRUE_NEAR
    ok = np.isfinite(w)
    r = w - post
    rows = []
    for lo, hi in zip(TERR_EDGES[:-1], TERR_EDGES[1:]):
        m = near & ok & (np.abs(t_err) >= lo) & (np.abs(t_err) < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "fusion_net": float(w[m].mean()) if m.sum() >= 20 else np.nan,
                     "fusion_R0": float(post[m].mean()) if m.sum() >= 20 else np.nan,
                     "fusion_R1": float(post_free[m].mean()) if m.sum() >= 20 else np.nan,
                     "accuracy_net": _acc(w[m], C[m]) if m.sum() >= 20 else np.nan,
                     "accuracy_R0": _acc(post[m], C[m]) if m.sum() >= 20 else np.nan})
    m = near & ok
    sl_net = lstsq_coefs(np.column_stack([np.ones(m.sum()), np.abs(t_err[m])]), w[m])[1]
    sl_r0 = lstsq_coefs(np.column_stack([np.ones(m.sum()), np.abs(t_err[m])]), post[m])[1]
    coef = lstsq_coefs(np.column_stack([np.ones(ok.sum()), d["x_eye"][ok], d["eye"][ok]]), r[ok])
    coef_t = lstsq_coefs(np.column_stack([np.ones(ok.sum()), t_err[ok]]), r[ok])
    ecc = []
    for lo, hi in zip(EYE_EDGES[:-1], EYE_EDGES[1:]):
        m = ok & (np.abs(d["eye"]) >= lo) & (np.abs(d["eye"]) < hi)
        ecc.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "accuracy_net": _acc(w[m], C[m]), "accuracy_R0": _acc(post[m], C[m]),
                    "accuracy_R1": _acc(post_free[m], C[m]),
                    "residual_w": float(r[m].mean()),
                    "abs_err_mu_vis": float(np.abs(pred[m, iv] - d["mu_vis"][m]).mean()),
                    "abs_err_mu_prop": float(np.abs(pred[m, ip] - d["mu_prop"][m]).mean()),
                    "fusion_true_net": float(np.nanmean(w[m & near])) if (m & near).sum() >= 20 else np.nan,
                    "fusion_true_R0": float(post[m & near].mean()) if (m & near).sum() >= 20 else np.nan})
    return {"t_err_sd": float(t_err.std()),
            "t_err_sd_analytical": float(np.sqrt(np.mean(k ** 2 * d["sig2_eye"] + (1 - k) ** 2 * gen["eye_sigma_sq"]))),
            "sensory_noise_diff_sd": float(np.std((d["x_vis"] - d["retinal"]) - (d["x_prop"] - d["s_prop"]))),
            "fusion_true_by_abs_terr": rows,
            "fusion_true_slope_on_abs_terr_net": float(sl_net), "fusion_true_slope_on_abs_terr_R0": float(sl_r0),
            "residual_on_1_xeye_eye": [float(v) for v in coef], "residual_on_1_terr": [float(v) for v in coef_t],
            "residual_sd": float(r[ok].std()),
            "by_eye_eccentricity": ecc}


def frame_block(b, w, obs):
    d = b["d"]
    post = obs["post_c1"]
    d_body, d_ret = obs["disparity"], d["x_vis"] - d["x_prop"]
    cb, cr = crossing(d_body, w), crossing(d_ret, w)
    return {"sd_retinal_minus_body": float(np.std(d_ret - d_body)),
            "crossing_body_net": cb, "crossing_retinal_net": cr,
            "crossing_body_R0": crossing(d_body, post), "crossing_retinal_R0": crossing(d_ret, post),
            "lookup_r2_body_net": lookup_r2(d_body, w), "lookup_r2_retinal_net": lookup_r2(d_ret, w),
            "lookup_r2_body_R0": lookup_r2(d_body, post), "lookup_r2_retinal_R0": lookup_r2(d_ret, post),
            "decision_frame_index_net": float(cr / cb) if np.isfinite(cb) and cb > 0 else np.nan,
            "decision_frame_index_R0": float(crossing(d_ret, post) / crossing(d_body, post))}


def partials_block(b, obs, n=3000, delta=0.5):
    """Noiseless central differences of the outputs with respect to x_vis, x_eye
    and x_prop on `n` test trials: the effective eye gain (ratio of partials;
    the observer's is exactly k on every trial), the leakage of the eye signal
    into the hand estimate on segregated trials (observer: 0), and the shared
    error of the two estimates with t_err on segregated trials."""
    d, gen, names = b["d"], b["cfg"]["generative"], b["names"]
    iv, ip = names.index("mu_vis"), names.index("mu_prop")
    base = {k: np.asarray(d[k][:n], float) for k in ("x_vis", "x_prop", "x_eye", "sig2_vis", "sig2_prop", "sig2_eye")}
    post = obs["post_c1"][:n]
    k = eye_gain_k(gen, base["sig2_eye"])
    P = {}
    for var in ("x_vis", "x_eye", "x_prop"):
        dp, dm = dict(base), dict(base)
        dp[var] = base[var] + delta
        dm[var] = base[var] - delta
        P[var] = (predict_on(b, dp, poisson=False) - predict_on(b, dm, poisson=False)) / (2 * delta)
    dv, de, dh = P["x_vis"], P["x_eye"], P["x_prop"]
    okv = np.abs(dv[:, iv]) > 0.05
    ratio = de[okv, iv] / dv[okv, iv]
    fused, seg = post > 0.7, post < 0.2
    okp = fused & (np.abs(dv[:, ip]) > 0.05)
    ratio_p = de[okp, ip] / dv[okp, ip]
    s0 = gen["sigma0_sq"]
    vvb = obs["var_vis_body"][:n]
    # shared error on segregated trials: regress the estimate errors on t_err
    t_err, _ = transformation_error({kk: v[:n] for kk, v in d.items() if isinstance(v, np.ndarray) and len(v) == len(d["x_vis"])}, gen)
    pred0 = predict_on(b, base, poisson=False)
    err_v, err_p = pred0[:, iv] - d["s_vis"][:n], pred0[:, ip] - d["s_prop"][:n]
    Xs = np.column_stack([np.ones(seg.sum()), t_err[seg]])
    cv, cp = lstsq_coefs(Xs, err_v[seg])[1], lstsq_coefs(Xs, err_p[seg])[1]
    return {"n_trials": int(n), "delta": delta,
            "eye_gain_net_median": float(np.median(ratio)), "eye_gain_net_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            "eye_gain_net_fused": float(np.median(ratio[fused[okv]])) if fused[okv].sum() > 20 else np.nan,
            "eye_gain_net_segregated": float(np.median(ratio[seg[okv]])) if seg[okv].sum() > 20 else np.nan,
            "eye_gain_k_observer": float(k.mean()),
            "eye_gain_hand_fused_net": float(np.median(ratio_p)) if okp.sum() > 20 else np.nan,
            "d_muvis_dxvis_median": float(np.median(dv[:, iv])), "d_muvis_dxeye_median": float(np.median(de[:, iv])),
            "leak_d_muprop_dxeye_segregated_median": float(np.median(de[seg, ip])),
            "leak_d_muprop_dxeye_segregated_absmean": float(np.abs(de[seg, ip]).mean()),
            "d_muprop_dxprop_segregated_median": float(np.median(dh[seg, ip])),
            "d_muprop_dxprop_segregated_observer": float(np.mean(s0 / (s0 + base["sig2_prop"][seg]))),
            "d_muvis_dxeye_segregated_median": float(np.median(de[seg, iv])),
            "d_muvis_dxeye_segregated_observer": float(np.mean(k[seg] * s0 / (s0 + vvb[seg]))),
            "d_muprop_dxeye_fused_median": float(np.median(de[fused, ip])),
            "shared_error_muvis_on_terr_segregated": float(cv),
            "shared_error_muvis_on_terr_observer": float(np.mean(s0 / (s0 + vvb[seg]))),
            "shared_error_muprop_on_terr_segregated": float(cp),
            "n_segregated": int(seg.sum()), "n_fused": int(fused.sum())}


def _perturbed(b, trials, obs0, seed=1):
    """Predictions and hybrid weight on a manipulated trial dict, against the
    observer fed the same manipulated signal."""
    gen = b["cfg"]["generative"]
    obs = observer_on(trials, gen)
    pred = predict_on(b, trials, poisson=True, seed=seed)
    w, wp, _ = hybrid_reads(b, pred, obs)
    return pred, w, obs


def eps_block(b, obs0, w0, levels=EPS_LEVELS):
    d, gen, names = b["d"], b["cfg"]["generative"], b["names"]
    ivv = names.index("var_vis")
    t = trial_dict(d)
    true_disp = d["s_vis"] - d["s_prop"]
    near = np.abs(true_disp) <= TRUE_NEAR
    C = d["C"]
    k = float(eye_gain_k(gen, np.mean(gen["sigma2_eye_range"])))
    rows = []
    for eps in levels:
        tt = dict(t); tt["x_eye"] = t["x_eye"] + eps
        pred, w, obs = _perturbed(b, tt, obs0)
        ok = np.isfinite(w)
        rows.append({"eps": float(eps), "predicted_shift": float(-k * eps),
                     "fusion_true_net": float(np.nanmean(w[near])), "fusion_true_R0eps": float(obs["post_c1"][near].mean()),
                     "accuracy_net": _acc(w, C), "accuracy_R0eps": _acc(obs["post_c1"], C),
                     "agreement_net_R0eps": float(np.mean((w[ok] > 0.5) == (obs["post_c1"][ok] > 0.5))),
                     "peak_w_vs_true_disparity_net": peak_location(true_disp, w, TRUE_GRID),
                     "peak_w_vs_true_disparity_R0eps": peak_location(true_disp, obs["post_c1"], TRUE_GRID),
                     "mean_residual_w": float(np.nanmean(w - obs["post_c1"])),
                     "curve_net": binned_mean(true_disp, w, TRUE_GRID), "curve_R0eps": binned_mean(true_disp, obs["post_c1"], TRUE_GRID)})
    return {"k": k, "levels": rows}


def beta_block(b, obs0, levels=BETA_LEVELS):
    d, gen, names = b["d"], b["cfg"]["generative"], b["names"]
    iv = names.index("mu_vis")
    t = trial_dict(d)
    C = d["C"]
    near = np.abs(d["s_vis"] - d["s_prop"]) <= TRUE_NEAR
    rows = []
    for beta in levels:
        tt = dict(t); tt["x_eye"] = t["x_eye"] * beta
        pred, w, obs = _perturbed(b, tt, obs0)
        ok = np.isfinite(w)
        ecc = []
        for lo, hi in zip(EYE_EDGES[:-1], EYE_EDGES[1:]):
            m = ok & (np.abs(d["eye"]) >= lo) & (np.abs(d["eye"]) < hi)
            ecc.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "accuracy_net": _acc(w[m], C[m]),
                        "accuracy_R0beta": _acc(obs["post_c1"][m], C[m]), "accuracy_R0_true_signal": _acc(obs0["post_c1"][m], C[m]),
                        "fusion_true_net": float(np.nanmean(w[m & near])) if (m & near).sum() >= 20 else np.nan,
                        "fusion_true_R0beta": float(obs["post_c1"][m & near].mean()) if (m & near).sum() >= 20 else np.nan})
        bias_net = lstsq_coefs(np.column_stack([np.ones(len(d["eye"])), d["eye"]]), pred[:, iv] - d["s_vis"])[1]
        bias_obs = lstsq_coefs(np.column_stack([np.ones(len(d["eye"])), d["eye"]]), obs["mu_vis"] - d["s_vis"])[1]
        rows.append({"beta": float(beta), "accuracy_net": _acc(w, C), "accuracy_R0beta": _acc(obs["post_c1"], C),
                     "fusion_true_net": float(np.nanmean(w[near])), "fusion_true_R0beta": float(obs["post_c1"][near].mean()),
                     "bias_muvis_on_eye_net": float(bias_net), "bias_muvis_on_eye_R0beta": float(bias_obs),
                     "by_eye_eccentricity": ecc})
    return {"levels": rows}


def silence_block(b, obs0):
    d, gen = b["d"], b["cfg"]["generative"]
    t = trial_dict(d)
    tt = dict(t); tt["x_eye"] = np.full(len(t["x_eye"]), float(gen["eye_mu"]))
    pred, w, obs = _perturbed(b, tt, obs0)
    near = np.abs(d["s_vis"] - d["s_prop"]) <= TRUE_NEAR
    C = d["C"]
    return {"fusion_true_net": float(np.nanmean(w[near])), "fusion_true_R0silent": float(obs["post_c1"][near].mean()),
            "accuracy_net": _acc(w, C), "accuracy_R0silent": _acc(obs["post_c1"], C),
            "accuracy_R0_true_signal": _acc(obs0["post_c1"], C),
            "crossing_true_disparity_net": crossing(d["s_vis"] - d["s_prop"], w),
            "crossing_true_disparity_R0silent": crossing(d["s_vis"] - d["s_prop"], obs["post_c1"])}


def eyerel_block(b, w, obs, n_bins=8):
    """Channel B. Within |disparity| quantile bins: the slope of the network's
    weight on the posterior (the repo's reliability-within-disparity, all
    reliabilities), and the eye-specific version -- the coefficient of w and
    of the posterior on sig2_eye in a regression with sig2_vis and sig2_prop
    as covariates. The observer's coefficient is negative near zero disparity
    (a noisier eye signal makes a common cause less likely) and positive at
    the flanks (the window widens), so the bins are reported, not pooled.
    Also the fusion window per eye-noise tercile."""
    d = b["d"]
    post = obs["post_c1"]
    ad = np.abs(obs["disparity"])
    rel = reliability_within_disparity(w, post, ad, n_bins=n_bins)
    ok = np.isfinite(w)
    edges = np.quantile(ad[ok], np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        m = ok & (ad >= edges[i]) & (ad < edges[i + 1] if i < n_bins - 1 else ad <= edges[i + 1])
        if m.sum() < 60 or np.ptp(d["sig2_eye"][m]) < 1e-6:
            continue
        X = np.column_stack([np.ones(m.sum()), d["sig2_eye"][m], d["sig2_vis"][m], d["sig2_prop"][m]])
        bn, bo = lstsq_coefs(X, w[m])[1], lstsq_coefs(X, post[m])[1]
        rows.append({"d_lo": float(edges[i]), "d_hi": float(edges[i + 1]), "n": int(m.sum()),
                     "slope_w_on_sig2eye": float(bn), "slope_post_on_sig2eye": float(bo),
                     "ratio": float(bn / bo) if abs(bo) > 1e-4 else np.nan})
    near = [r for r in rows if r["d_hi"] <= 3.0]
    flank = [r for r in rows if r["d_lo"] >= 4.0 and r["d_hi"] <= 12.0]
    def pooled(sel):
        if not sel:
            return np.nan, np.nan
        n = np.array([r["n"] for r in sel], float)
        return float((n * np.array([r["slope_w_on_sig2eye"] for r in sel])).sum() / n.sum()), float((n * np.array([r["slope_post_on_sig2eye"] for r in sel])).sum() / n.sum())
    nn, no = pooled(near)
    fn, fo = pooled(flank)
    eq = np.quantile(d["sig2_eye"], [0, 1 / 3, 2 / 3, 1])
    terciles = []
    for j in range(3):
        m = (d["sig2_eye"] >= eq[j]) & (d["sig2_eye"] <= eq[j + 1])
        terciles.append({"sig2_eye_lo": float(eq[j]), "sig2_eye_hi": float(eq[j + 1]), "n": int(m.sum()),
                         "crossing_net": crossing(obs["disparity"][m], w[m]), "crossing_R0": crossing(obs["disparity"][m], post[m]),
                         "fusion_net": float(np.nanmean(w[m & (ad <= NEAR)])), "fusion_R0": float(post[m & (ad <= NEAR)].mean())})
    return {"reliability_within_disparity_slope": float(rel["combined_slope"]), "reliability_within_disparity_se": float(rel["combined_se"]),
            "bins": rows,
            "slope_near_net": nn, "slope_near_R0": no, "slope_flank_net": fn, "slope_flank_R0": fo,
            "ratio_near": float(nn / no) if np.isfinite(no) and abs(no) > 1e-5 else np.nan,
            "ratio_flank": float(fn / fo) if np.isfinite(fo) and abs(fo) > 1e-5 else np.nan,
            "by_sig2_eye_tercile": terciles,
            "sig2_eye_range_in_test": [float(d["sig2_eye"].min()), float(d["sig2_eye"].max())]}

def ecc_block(b, w, obs):
    """Channel A, eccentricity lever: the fusion window and fusion within bins
    of |eye|, network vs own observer (R1 is flat in |eye| by construction).
    With eccentricity-dependent eye noise the observer's window widens with
    |eye|; with constant noise it does not."""
    d, gen = b["d"], b["cfg"]["generative"]
    post = obs["post_c1"]
    ad = np.abs(obs["disparity"])
    k = eye_gain_k(gen, d["sig2_eye"])
    var_eye = 1.0 / (1.0 / d["sig2_eye"] + 1.0 / gen["eye_sigma_sq"])
    rows = []
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 18), (18, 60)):
        m = (np.abs(d["eye"]) >= lo) & (np.abs(d["eye"]) < hi)
        if m.sum() < 200:
            continue
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "sig2_eye_mean": float(d["sig2_eye"][m].mean()), "var_eye_mean": float(var_eye[m].mean()), "k_mean": float(k[m].mean()),
                     "crossing_net": crossing(obs["disparity"][m], w[m], step=1.0), "crossing_R0": crossing(obs["disparity"][m], post[m], step=1.0),
                     "fusion_net": float(np.nanmean(w[m & (ad <= NEAR)])) if (m & (ad <= NEAR)).sum() >= 20 else np.nan,
                     "fusion_R0": float(post[m & (ad <= NEAR)].mean()) if (m & (ad <= NEAR)).sum() >= 20 else np.nan,
                     "accuracy_net": _acc(w[m], d["C"][m]), "accuracy_R0": _acc(post[m], d["C"][m])})
    return {"by_abs_eye": rows}


# --------------------------------------------------------------------------- #
def units_block(b, di=True):
    """Congruent / opposite / mixed classes (the repo's definition, eye held
    at eye_mu) and a light displacement index per MSL unit (alignment over five
    eye positions, usable units only), summarised by class."""
    cfg, enc, encs = b["cfg"], dict(b["cfg"]["encoding"], poisson_noise=False), b["encoders"]
    gen = cfg["generative"]
    net = b["net"]
    rng = np.random.default_rng(0)
    sig2 = {"vis": float(np.mean(gen["sigma2_vis_range"])), "prop": float(np.mean(gen["sigma2_prop_range"])),
            "eye": float(np.mean(gen["sigma2_eye_range"]))}
    eye_mu = float(gen["eye_mu"])

    def sweep(x_vis, x_prop, x_eye, layer=-1):
        n = max(np.size(x_vis), np.size(x_prop))
        dd = {"x_vis": np.full(n, x_vis, float) if np.size(x_vis) == 1 else np.asarray(x_vis, float),
              "x_prop": np.full(n, x_prop, float) if np.size(x_prop) == 1 else np.asarray(x_prop, float),
              "x_eye": np.full(n, float(x_eye)), "sig2_vis": np.full(n, sig2["vis"]),
              "sig2_prop": np.full(n, sig2["prop"]), "sig2_eye": np.full(n, sig2["eye"])}
        from cmsi.data.encoding import encode as _enc
        return net.hidden(_enc(dd, encs, enc, rng))[layer]

    span = 2.0 * np.sqrt(gen["sigma0_sq"])
    pos = np.linspace(-span, span, 41)
    rv = sweep(pos - eye_mu, 0.0, eye_mu)
    rp = sweep(-eye_mu, pos, eye_mu)
    n_units = rv.shape[1]
    index = np.zeros(n_units); tuned = np.zeros(n_units, bool)
    for u in range(n_units):
        tuned[u] = rv[:, u].std() > 1e-4 and rp[:, u].std() > 1e-4
        if tuned[u]:
            index[u] = np.corrcoef(rv[:, u], rp[:, u])[0, 1]
    classes = np.where(~tuned, "untuned", np.where(index > 0.5, "congruent", np.where(index < -0.5, "opposite", "mixed")))
    out = {"n_congruent": int((classes == "congruent").sum()), "n_opposite": int((classes == "opposite").sum()),
           "n_mixed": int((classes == "mixed").sum()), "n_untuned": int((classes == "untuned").sum()),
           "index": index.tolist(), "classes": classes.tolist()}
    if not di:
        return out
    s = np.linspace(-30, 30, 241)
    eyes = eye_mu + np.array([-9.0, -4.5, 0.0, 4.5, 9.0])
    win = (s >= -15) & (s <= 15)
    deltas = np.linspace(-0.5, 1.5, 81)
    R = [sweep(s - e, 0.0, e) for e in eyes]
    DI = np.full(n_units, np.nan); R2 = np.full(n_units, np.nan); AMP = np.zeros(n_units)
    for u in range(n_units):
        r0 = R[2][:, u]
        AMP[u] = np.ptp(r0[win])
        if AMP[u] < 0.1:
            continue
        err = []
        for dl in deltas:
            e_tot = 0.0
            for ei, e in enumerate(eyes):
                if ei == 2:
                    continue
                e_tot += np.mean((R[ei][win, u] - np.interp(s[win] - dl * (e - eye_mu), s, r0)) ** 2)
            err.append(e_tot)
        err = np.array(err)
        var = np.mean([np.var(R[i][win, u]) for i in range(5)])
        DI[u] = deltas[int(np.argmin(err))]
        R2[u] = 1 - err.min() / (4 * var + 1e-12)
    usable = np.isfinite(DI) & (R2 > 0.8)
    by_class = {}
    for c in ("congruent", "opposite", "mixed"):
        m = usable & (classes == c)
        by_class[c] = {"n_usable": int(m.sum()), "n_class": int((classes == c).sum()),
                       "di_median": float(np.median(DI[m])) if m.sum() else np.nan,
                       "share_body": float(np.mean(np.abs(DI[m]) < 0.2)) if m.sum() else np.nan,
                       "share_retinal": float(np.mean(DI[m] >= 0.8)) if m.sum() else np.nan}
    out.update({"di": DI.tolist(), "di_r2": R2.tolist(), "di_amplitude": AMP.tolist(),
                "di_median_usable": float(np.median(DI[usable])) if usable.sum() else np.nan,
                "di_n_usable": int(usable.sum()),
                "di_share_body": float(np.mean(np.abs(DI[usable]) < 0.2)) if usable.sum() else np.nan,
                "di_share_intermediate": float(np.mean((DI[usable] >= 0.2) & (DI[usable] < 0.8))) if usable.sum() else np.nan,
                "di_share_retinal": float(np.mean(DI[usable] >= 0.8)) if usable.sum() else np.nan,
                "di_by_class": by_class})
    return out


# --------------------------------------------------------------------------- #
def analyse(b, do_eps=True, do_beta=True, do_units=True, do_eyerel=True, do_di=True):
    """Everything, for one causal network bundle."""
    d = b["d"]
    pred = b["net"].predict(d["X"])
    obs = {k: d[k] for k in ("post_c1", "fused_mu", "fused_var", "seg_vis_mu", "seg_vis_var",
                             "seg_prop_mu", "seg_prop_var", "disparity", "x_vis_body", "var_vis_body",
                             "mu_vis", "var_vis", "mu_prop", "var_prop")}
    w, wp, flags = hybrid_reads(b, pred, obs)
    out = {"basic": _basic(b)}
    out["R0"] = r0_block(b, pred, w, wp, obs)
    r1 = r1_block(b, w, obs)
    post_free = r1.pop("_post_free")
    out["R1"] = r1
    out["terr"] = terr_block(b, pred, w, obs, post_free)
    out["frame"] = frame_block(b, w, obs)
    out["partials"] = partials_block(b, obs)
    if do_eps:
        out["eps"] = eps_block(b, obs, w)
    if do_beta:
        out["beta"] = beta_block(b, obs)
    out["silence"] = silence_block(b, obs)
    out["ecc"] = ecc_block(b, w, obs)
    if do_eyerel:
        out["eyerel"] = eyerel_block(b, w, obs)
    if do_units:
        out["units"] = units_block(b, di=do_di)
    out["_arrays"] = {"w": w, "post": obs["post_c1"], "post_free": post_free, "disparity": obs["disparity"],
                      "retinal_disparity": d["x_vis"] - d["x_prop"], "true_disparity": d["s_vis"] - d["s_prop"],
                      "eye": d["eye"], "t_err": transformation_error(d, b["cfg"]["generative"])[0], "C": d["C"],
                      "sig2_eye": d["sig2_eye"], "sig2_vis": d["sig2_vis"], "x_eye": d["x_eye"],
                      "var_vis": pred[:, b["names"].index("var_vis")], "var_vis_R0": obs["var_vis"]}
    return out


def flatten(res):
    """The scalar summary of analyse() as one flat row (for tables and sweeps)."""
    row = {}
    for sec in ("basic", "R0", "R1", "frame", "partials", "silence"):
        for k, v in res.get(sec, {}).items():
            if isinstance(v, (int, float, str, np.floating, np.integer)) and not k.startswith("_"):
                row[f"{sec}.{k}"] = v
    t = res.get("terr", {})
    for k in ("t_err_sd", "sensory_noise_diff_sd", "fusion_true_slope_on_abs_terr_net", "fusion_true_slope_on_abs_terr_R0", "residual_sd"):
        row[f"terr.{k}"] = t.get(k)
    if "residual_on_1_xeye_eye" in t:
        row["terr.residual_on_xeye"] = t["residual_on_1_xeye_eye"][1]
        row["terr.residual_on_eye"] = t["residual_on_1_xeye_eye"][2]
    for r in t.get("fusion_true_by_abs_terr", []):
        tag = f"{r['lo']:g}_{r['hi']:g}"
        row[f"terr.fusion_true_net_{tag}"] = r["fusion_net"]; row[f"terr.fusion_true_R0_{tag}"] = r["fusion_R0"]; row[f"terr.fusion_true_R1_{tag}"] = r["fusion_R1"]
    for r in t.get("by_eye_eccentricity", []):
        tag = f"{r['lo']:g}_{r['hi']:g}"
        row[f"terr.acc_net_eye_{tag}"] = r["accuracy_net"]; row[f"terr.acc_R0_eye_{tag}"] = r["accuracy_R0"]
        row[f"terr.abs_err_muvis_eye_{tag}"] = r["abs_err_mu_vis"]
    for r in res.get("eps", {}).get("levels", []):
        tag = f"{r['eps']:+g}"
        for k in ("fusion_true_net", "fusion_true_R0eps", "accuracy_net", "accuracy_R0eps", "peak_w_vs_true_disparity_net", "peak_w_vs_true_disparity_R0eps", "predicted_shift", "mean_residual_w"):
            row[f"eps.{k}_{tag}"] = r[k]
    for r in res.get("beta", {}).get("levels", []):
        tag = f"{r['beta']:g}"
        for k in ("accuracy_net", "accuracy_R0beta", "fusion_true_net", "fusion_true_R0beta", "bias_muvis_on_eye_net", "bias_muvis_on_eye_R0beta"):
            row[f"beta.{k}_{tag}"] = r[k]
    for r in res.get("ecc", {}).get("by_abs_eye", []):
        tag = f"{r['lo']:g}_{r['hi']:g}"
        for k in ("crossing_net", "crossing_R0", "fusion_net", "fusion_R0", "accuracy_net", "accuracy_R0", "var_eye_mean"):
            row[f"ecc.{k}_eye_{tag}"] = r[k]
    e = res.get("eyerel", {})
    for k in ("reliability_within_disparity_slope", "slope_near_net", "slope_near_R0", "slope_flank_net", "slope_flank_R0", "ratio_near", "ratio_flank"):
        row[f"eyerel.{k}"] = e.get(k)
    for t in e.get("by_sig2_eye_tercile", []):
        tag = f"{t['sig2_eye_lo']:.0f}_{t['sig2_eye_hi']:.0f}"
        row[f"eyerel.crossing_net_{tag}"] = t["crossing_net"]; row[f"eyerel.crossing_R0_{tag}"] = t["crossing_R0"]
    u = res.get("units", {})
    for k in ("n_congruent", "n_opposite", "n_mixed", "n_untuned", "di_median_usable", "di_n_usable", "di_share_body", "di_share_intermediate", "di_share_retinal"):
        row[f"units.{k}"] = u.get(k)
    for c, v in u.get("di_by_class", {}).items():
        row[f"units.di_median_{c}"] = v["di_median"]; row[f"units.share_body_{c}"] = v["share_body"]; row[f"units.n_usable_{c}"] = v["n_usable"]
    return row
