"""The implied-weight investigation.

The network never outputs a weight. Every weight in this project is *implied*:
recovered from where the network's position estimate sits between the two
causal hypotheses. That recovery has three moving parts, and this experiment
looks at each of them on its own, away from the main pipeline:

    sigma_out     the read-out noise, one number per output, measured on a
                  p_common = 1 control where the target is single-valued
    sigma_w       sigma_out / |Delta| per trial -- how readable the weight is
                  on that trial, which the sigma_w criterion turns into a filter
    the weight    per trial (a ratio), per disparity bin (least squares), per
                  posterior bin (both estimators side by side), per reliability
                  level, and per network -- and, on every trial, from both
                  position outputs together and from the VARIANCE outputs,
                  whose sensitivity to the weight does not vanish where the
                  two hypotheses coincide (variance_weight, per_trial_weights)

Everything is computed on a network's stored test split, exactly as stage 3
does, but written under results/experiments/implied_weight/<name>/ so that
nothing here can disturb results/<run>/. When a variant looks right, its
config goes into configs/ and the whole pipeline is re-run.

Entry points (scripts/06_implied_weight.py):

    figures       the three per-run figures this grew out of, for one run
    sigma         what sigma_out and sigma_w are on one run, and what the
                  criterion would keep -- nothing is filtered
    reliability   the weight against disparity at many levels of each input's
                  reliability, read from either output
    compare       several trained runs side by side: what their trials allowed
                  the per-trial ratio to show and what each network achieved
                  (figures in cmsi.experiments.design)
    design        a configuration before training: the posterior, |Delta|, the
                  ratios expected outside [-1, 2] for any read-out error, the
                  decoding floor on that error (cmsi.experiments.design)
    train         a new configuration through the whole pipeline (stages 0-4)
                  into results/<name>/, with a p_common = 1 control of the
                  same configuration, so the analyses above apply to it

Other network sizes are a separate experiment, cmsi.experiments.architecture
(scripts/07_architecture.py); one config key over several values another,
cmsi.experiments.sweep (scripts/09_sweep.py). Both reuse the analysis and
figures here.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import (
    binned_implied_weight,
    fusion_weight,
    implied_weight_by_posterior,
    mean_by_bin,
    position_regression,
    sigma_w,
    transition_fit,
)
from cmsi.viz import results as viz_results
from cmsi.viz.style import COLORS, SIZE, label_panels

EXPERIMENT = "implied_weight"

# The disparity grid and the guards on the per-bin least-squares weight, as
# in scripts/05_prior_sweep.py: 18 bins from -30 to +30 deg, coarse near zero
# where the two hypotheses coincide and no weight is identifiable. The
# pipeline's own figure 05 keeps the config's disparity_grid; everything new
# here uses this one, which is what the manuscript's prior-sweep figure uses.
GRID = [-30, -24, -19, -15, -12, -9, -6, -3.5, -1.5,
        1.5, 3.5, 6, 9, 12, 15, 19, 24, 30]
MIN_COUNT = 25
MAX_SE = 0.05
# Per reliability level a curve rests on a fifth of the trials or less, so the
# guards are relaxed there; a bin's error bar still says how good it is.
LEVEL_MIN_COUNT = 10
LEVEL_MAX_SE = 0.1
# The per-trial ratio is drawn on [-1, 2]; ratios outside are "spikes".
RATIO_LIM = (-1.0, 2.0)

READS = (("vis", "mu_vis", "seg_vis_mu"), ("prop", "mu_prop", "seg_prop_mu"))
INPUTS = {"vis": "sig2_vis", "prop": "sig2_prop", "eye": "sig2_eye"}
INPUT_LABEL = {"vis": "visual noise sigma2_vis",
               "prop": "proprioceptive noise sigma2_prop",
               "eye": "eye-position noise sigma2_eye"}


# --------------------------------------------------------------------------- #
# where things go
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/implied_weight/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def experiment_data_dir():
    """data/experiments/implied_weight/ -- datasets the variants train on."""
    from cmsi.utils.paths import DATA

    path = DATA / "experiments" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# loading an existing run
# --------------------------------------------------------------------------- #
def load_run(run):
    """(pred, d_test, names, cfg) for results/<run>/, on its stored test split."""
    from cmsi.data import subset
    from cmsi.models import predict
    from cmsi.utils import dataset_path, load_checkpoint, load_dataset, run_dir

    out = run_dir(run, create=False)
    model, cfg, _, splits = load_checkpoint(out / "model.pt")
    d_full, _ = load_dataset(dataset_path((out / "dataset.txt").read_text().strip()))
    d = subset(d_full, splits["test"])
    return predict(model, d["X"]), d, list(d_full["target_names"]), cfg


def control_residuals(control):
    """Per-trial residuals (network - analytical) of a p_common = 1 control on
    its test split, per output. Their std is sigma_out (SS9.1)."""
    pred, d, names, cfg = load_run(control)
    if cfg["generative"]["p_common"] < 1.0:
        print(f"warning: control '{control}' has p_common = "
              f"{cfg['generative']['p_common']}, not 1 -- its residuals contain "
              f"causal misweighting, so sigma_out will be too large")
    target = np.stack([d[k] for k in names], axis=1)
    return pred - target, names


# --------------------------------------------------------------------------- #
# the per-run analysis
# --------------------------------------------------------------------------- #
def weight_analysis(pred, d, names, sig_out, acfg, grid=GRID,
                    min_count=MIN_COUNT, max_se=MAX_SE):
    """Everything the investigation needs from one network on one test split.

    Returns (metrics, arrays). `metrics` is json-able and goes to metrics.json;
    `arrays` holds the per-trial and per-bin arrays the figures draw.

    Two transition midpoints are reported per read, because the pipeline and
    the sweep use different estimators and the difference is part of what this
    experiment is for:

        midpoint_filtered   logistic fitted to the sigma_w-filtered per-trial
                            ratio, exactly as stage 3 and the sweep do
        midpoint_ls         logistic fitted to the least-squares per-bin
                            weight, which uses every trial
    """
    acfg = dict(acfg)
    crit = float(acfg.get("sigma_w_criterion", 0.1))
    min_sep = float(acfg.get("min_separation", 1.0))
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    target = np.stack([d[k] for k in names], axis=1)
    disp, post = d["disparity"], d["post_c1"]

    metrics = {
        "sigma_w_criterion": crit, "min_separation": min_sep,
        "n_test": int(len(pred)), "grid": grid.tolist(),
        "sigma_out": {names[i]: float(sig_out[i]) for i in range(len(names))},
        "residual_std_self": {names[i]: float((pred[:, i] - target[:, i]).std())
                              for i in range(len(names))},
        "accuracy": accuracy(pred, target, names),
        "reads": {},
    }
    arrays = {"disparity": disp, "post_c1": post}

    c_a, m_a, _ = mean_by_bin(disp, post, grid)
    arrays["analytical_curve"] = (c_a, m_a)
    mid_a, k_a = transition_fit(np.abs(disp), post)
    metrics["analytical"] = {"midpoint_deg": mid_a, "sharpness": k_a}

    for key, out_name, seg_name in READS:
        i, seg = names.index(out_name), d[seg_name]
        delta = d["fused_mu"] - seg
        so = float(sig_out[i])
        w = fusion_weight(pred[:, i], seg, d["fused_mu"], min_sep)
        sw = sigma_w(so, delta)
        w_f = np.where(sw < crit, w, np.nan)
        curve = binned_implied_weight(pred[:, i], seg, d["fused_mu"], disp, grid,
                                      min_count=min_count, sigma_out=so, max_se=max_se)
        by_post = implied_weight_by_posterior(
            pred[:, i], seg, d["fused_mu"], post, so,
            sigma_w_criterion=crit, min_separation=min_sep)
        mid_f, k_f = transition_fit(np.abs(disp), w_f)
        mid_ls, k_ls = transition_fit(np.abs(curve[0]), curve[1])
        reg = position_regression(pred[:, i], seg, d["fused_mu"], post)

        # the ratio on EVERY trial with Delta != 0: no min_separation, no
        # sigma_w criterion. This is what the network's estimated weight looks
        # like before anything is done to it, and it is what the distribution
        # figure draws.
        with np.errstate(divide="ignore", invalid="ignore"):
            w_raw = np.where(delta != 0, (pred[:, i] - seg) / delta, np.nan)
        wr = w_raw[np.isfinite(w_raw)]
        finite = np.isfinite(sw)
        metrics["reads"][key] = {
            "w_raw": {"n": int(wr.size), "median": float(np.median(wr)),
                      "iqr": [float(np.percentile(wr, 25)), float(np.percentile(wr, 75))],
                      "mean": float(wr.mean()), "std": float(wr.std()),
                      "frac_outside_-1_2": float(np.mean((wr < -1) | (wr > 2)))},
            "net_minus_seg_std": float((pred[:, i] - seg).std()),
            "delta_std": float(delta.std()),
            "output": out_name,
            "sigma_out": so,
            "frac_readable": float(np.mean(sw < crit)),
            "n_readable": int(np.sum(sw < crit)),
            "sigma_w_median": float(np.median(sw[finite])),
            "sigma_w_iqr": [float(np.percentile(sw[finite], 25)),
                            float(np.percentile(sw[finite], 75))],
            "delta_threshold_deg": so / crit,
            "midpoint_filtered_deg": mid_f, "sharpness_filtered": k_f,
            "midpoint_ls_deg": mid_ls, "sharpness_ls": k_ls,
            "position_regression": reg,
            "curve": {"centres": curve[0].tolist(), "w": curve[1].tolist(),
                      "n": curve[2].tolist(), "se": curve[3].tolist()},
            "by_posterior": {k: v.tolist() for k, v in by_post.items()},
        }
        metrics["reads"][key]["readability"] = readability(
            pred[:, i], seg, d["fused_mu"], target[:, i], post, so, C=d.get("C"))
        arrays[key] = {"w": w, "w_filtered": w_f, "sigma_w": sw, "delta": delta,
                       "resid": pred[:, i] - target[:, i],
                       "w_raw": w_raw, "net_minus_seg": pred[:, i] - seg,
                       "curve": curve, "by_post": by_post}
    # the per-trial reads need the variance components and all four outputs
    # (a dataset written before fused_var was stored, or a fused-head twin,
    # has neither)
    if all(k in d for k in ("fused_var", "seg_vis_var", "seg_prop_var")) and all(
            k in names for k in ("mu_vis", "var_vis", "mu_prop", "var_prop")):
        metrics["per_trial"], arrays["per_trial"] = per_trial_weights(pred, d, names, sig_out,
                                                                      crit)
    else:
        metrics["per_trial"], arrays["per_trial"] = None, None
    return metrics, arrays


# --------------------------------------------------------------------------- #
# per-trial weights that do not divide by a single Delta: both position
# outputs jointly, and the variance outputs
# --------------------------------------------------------------------------- #
def variance_weight(v_hat, fused_var, seg_var, delta, hint=None):
    """The weight implied by one variance output, on every trial.

    The variance target of a model-averaging observer is the mixture variance

        v(w) = w fused_var + (1 - w) seg_var + w (1 - w) Delta^2

    (Eq. 10; the law of total variance), so a network's variance output v_hat
    can be solved for the weight the network used -- a quadratic in w,

        Delta^2 w^2 - (Delta^2 - c) w - (seg_var - v_hat) = 0,
        c = seg_var - fused_var > 0  (the variance fusion saves).

    Its sensitivity |dv/dw| = |Delta^2 (1 - 2w) - c| does NOT vanish at
    Delta = 0: there it equals c, so the read stays sharp exactly where the
    position-based ratio (sensitivity |Delta|) is blind. The two reads are
    complementary, and on a network that mixes both outputs with one weight
    they must agree.

    Roots: with Delta^2 <= c, v(w) is monotone on [0, 1] and the root whose
    Delta -> 0 limit is (seg_var - v_hat)/c is taken (flag 0). With
    Delta^2 > c, v(w) rises before it falls and two roots can lie in [0, 1];
    the one closer to `hint` -- the position-based joint read, which is
    precise on exactly these large-|Delta| trials -- is taken (flag 1;
    without a hint, the larger root). When v_hat exceeds the largest
    mixture variance any w can produce, the location of that maximum is
    returned (flag 2).

    Returns (w, sensitivity |dv/dw| at w, flags).
    """
    v_hat, delta = np.asarray(v_hat, float), np.asarray(delta, float)
    fused_var, seg_var = np.asarray(fused_var, float), np.asarray(seg_var, float)
    c = seg_var - fused_var
    a = delta ** 2
    b = a - c
    disc = b ** 2 + 4 * a * (seg_var - v_hat)
    w = np.full(v_hat.shape, np.nan)
    flags = np.zeros(v_hat.shape, int)
    linear = a <= 1e-9 * np.maximum(c, 1e-12)
    w[linear] = (seg_var[linear] - v_hat[linear]) / c[linear]
    ok = ~linear & (disc >= 0)
    root = np.sqrt(disc[ok])
    r_plus = (b[ok] + root) / (2 * a[ok])
    r_minus = (b[ok] - root) / (2 * a[ok])
    two = a[ok] > c[ok]
    pick = r_plus.copy()
    if hint is not None:
        h = np.asarray(hint, float)[ok]
        use_minus = two & np.isfinite(h) & (np.abs(r_minus - h) < np.abs(r_plus - h))
        pick[use_minus] = r_minus[use_minus]
    w[ok] = pick
    flags[ok] = np.where(two, 1, 0)
    above = ~linear & (disc < 0)
    w[above] = b[above] / (2 * a[above])
    flags[above] = 2
    sensitivity = np.abs(b - 2 * a * w)
    return w, sensitivity, flags


def _precision_combine(ws, ss, floor=1e-3):
    """Inverse-variance average of several per-trial reads, NaNs left out."""
    ws, ss = np.stack(ws), np.stack(ss)
    prec = np.where(np.isfinite(ws) & np.isfinite(ss), 1 / np.maximum(ss, floor) ** 2, 0.0)
    ws = np.where(np.isfinite(ws), ws, 0.0)
    total = prec.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (prec * ws).sum(0) / total, np.nan), \
            np.where(total > 0, 1 / np.sqrt(total), np.inf)


def _estimator_summary(w, s, post, ad, crit, lim=RATIO_LIM, n_bins=10):
    ok = np.isfinite(w)
    e = w - post
    small, large = ad < 1, ad > 4
    idx = np.clip((post * n_bins).astype(int), 0, n_bins - 1)
    centres, means, counts = [], [], []
    for b in range(n_bins):
        m = ok & (idx == b)
        if m.sum() >= 5:
            centres.append(float(post[m].mean()))
            means.append(float(w[m].mean()))
            counts.append(int(m.sum()))
    fin = ok & np.isfinite(s)
    return {
        "n": int(ok.sum()),
        "outside": float(np.mean(ok & ((w < lim[0]) | (w > lim[1])))),
        "sd_vs_posterior": float(np.std(e[ok])),
        "sd_vs_posterior_absdelta_lt1": float(np.std(e[ok & small])) if (ok & small).sum() > 1
        else np.nan,
        "sd_vs_posterior_absdelta_gt4": float(np.std(e[ok & large])) if (ok & large).sum() > 1
        else np.nan,
        "corr_with_posterior": float(np.corrcoef(w[ok], post[ok])[0, 1]) if ok.sum() > 2
        else np.nan,
        "sigma_w_median": float(np.median(s[fin])) if fin.any() else np.nan,
        "frac_sigma_w_lt_crit": float(np.mean(fin & (s < crit))),
        "by_posterior": {"centres": centres, "mean": means, "n": counts},
    }


def per_trial_weights(pred, d, names, sig_out, crit=0.1, n_bins=10):
    """Three per-trial weights beside the ratio, on every trial, and how
    they compare with the analytical posterior and with each other.

        joint      least squares across BOTH position outputs on the trial
                   (analysis.joint_fusion_weight): the two equations
                   est_k = w fused + (1 - w) seg_k share one w, so
                   w = sum_k Delta_k (est_k - seg_k) / sum_k Delta_k^2 with
                   sd sigma_out / sqrt(Delta_vis^2 + Delta_prop^2) -- the
                   disparity is split between the two reads, and this uses
                   all of it
        variance   the weight the two VARIANCE outputs imply
                   (variance_weight), inverse-variance combined, sharp where
                   Delta is small
        combined   all four outputs, inverse-variance combined

    The per-trial sd of the variance read uses the control's sigma_out for
    the variance outputs, which understates the run's actual error; the
    `coherence` block calibrates it: on the trials where the joint position
    read is precise (its sd below the sigma_w criterion), the two reads are
    compared bin by bin, and the sd of their difference bounds the variance
    read's own error (it still contains the position read's, up to the
    criterion).
    A network that mixes both outputs with one weight puts that comparison
    on the identity; where it leaves it, the network reports fusion in its
    uncertainty that it does not perform in its estimate (or vice versa).

    Returns (metrics, arrays).
    """
    from cmsi.analysis.causal import joint_fusion_weight

    iv, ivv, ip, ipv = (names.index(k) for k in ("mu_vis", "var_vis", "mu_prop", "var_prop"))
    post, fused = d["post_c1"], d["fused_mu"]
    dv, dp = fused - d["seg_vis_mu"], fused - d["seg_prop_mu"]
    ad = np.abs(dv)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_ratio_vis = np.where(dv != 0, (pred[:, iv] - d["seg_vis_mu"]) / dv, np.nan)
        w_ratio_prop = np.where(dp != 0, (pred[:, ip] - d["seg_prop_mu"]) / dp, np.nan)
    s_ratio_vis = sigma_w(float(sig_out[iv]), dv)
    s_ratio_prop = sigma_w(float(sig_out[ip]), dp)
    w_joint = joint_fusion_weight(pred[:, iv], d["seg_vis_mu"], pred[:, ip],
                                  d["seg_prop_mu"], fused)
    s_joint = sigma_w(float(sig_out[iv]), np.sqrt(dv ** 2 + dp ** 2))
    w_vv, sens_vv, flags_v = variance_weight(pred[:, ivv], d["fused_var"], d["seg_vis_var"],
                                             dv, hint=w_joint)
    w_vp, sens_vp, flags_p = variance_weight(pred[:, ipv], d["fused_var"], d["seg_prop_var"],
                                             dp, hint=w_joint)
    s_vv = float(sig_out[ivv]) / np.maximum(sens_vv, 1e-9)
    s_vp = float(sig_out[ipv]) / np.maximum(sens_vp, 1e-9)
    w_var, s_var = _precision_combine([w_vv, w_vp], [s_vv, s_vp])
    w_all, s_all = _precision_combine([w_joint, w_vv, w_vp], [s_joint, s_vv, s_vp])

    estimators = {"ratio_vis": (w_ratio_vis, s_ratio_vis),
                  "ratio_prop": (w_ratio_prop, s_ratio_prop),
                  "joint": (w_joint, s_joint), "variance_vis": (w_vv, s_vv),
                  "variance_prop": (w_vp, s_vp), "variance": (w_var, s_var),
                  "combined": (w_all, s_all)}
    metrics = {"estimators": {k: _estimator_summary(w, s, post, ad, crit)
                              for k, (w, s) in estimators.items()},
               "variance_root_flags": {
                   "vis": {"unique": float(np.mean(flags_v == 0)),
                           "two_roots": float(np.mean(flags_v == 1)),
                           "above_parabola": float(np.mean(flags_v == 2))},
                   "prop": {"unique": float(np.mean(flags_p == 0)),
                            "two_roots": float(np.mean(flags_p == 1)),
                            "above_parabola": float(np.mean(flags_p == 2))}},
               "c_median_deg2": {"vis": float(np.median(d["seg_vis_var"] - d["fused_var"])),
                                 "prop": float(np.median(d["seg_prop_var"] - d["fused_var"]))}}

    # coherence: joint position read vs variance read where the former is
    # precise (its sd below the criterion, i.e. |Delta_eff| > sigma_out/crit)
    precise = np.isfinite(w_joint) & np.isfinite(w_var) & (s_joint < crit)
    idx = np.clip((post * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        mp = m & precise
        rows.append({"bin": [b / n_bins, (b + 1) / n_bins], "n": int(m.sum()),
                     "n_precise": int(mp.sum()), "mean_posterior": float(post[m].mean()) if m.any()
                     else np.nan,
                     "joint_precise": float(w_joint[mp].mean()) if mp.any() else np.nan,
                     "variance_precise": float(w_var[mp].mean()) if mp.any() else np.nan,
                     "variance_all": float(np.nanmean(w_var[m])) if m.any() else np.nan})
    diff = (w_var - w_joint)[precise]
    metrics["coherence"] = {
        "precise_criterion": float(crit), "n_precise": int(precise.sum()),
        "frac_precise": float(precise.mean()),
        "position_read_sd_median_precise": float(np.median(s_joint[precise])) if precise.any()
        else np.nan,
        "mean_diff_variance_minus_joint": float(diff.mean()) if diff.size else np.nan,
        "sd_diff_variance_minus_joint": float(diff.std()) if diff.size else np.nan,
        "sd_joint_vs_posterior_precise": float((w_joint - post)[precise].std()) if diff.size
        else np.nan,
        "sd_variance_vs_posterior_precise": float((w_var - post)[precise].std()) if diff.size
        else np.nan,
        "by_posterior": rows}
    # the no-division headline statistic in the variance domain: the
    # network's variance reduction relative to segregation, v_hat - var_seg,
    # regressed on the optimal mixture's, v(p) - var_seg (slope 1 = optimal);
    # and the variance read itself regressed on the posterior
    metrics["variance_regression"], metrics["weight_regression"] = {}, {}
    ones = np.ones_like(post)
    for key, ivar, seg_var, delta, w_read in (("vis", ivv, d["seg_vis_var"], dv, w_vv),
                                              ("prop", ipv, d["seg_prop_var"], dp, w_vp)):
        v_opt = post * d["fused_var"] + (1 - post) * seg_var + post * (1 - post) * delta ** 2
        metrics["variance_regression"][key] = position_regression(pred[:, ivar], seg_var,
                                                                  v_opt, ones)
        okw = np.isfinite(w_read)
        metrics["weight_regression"][key] = position_regression(w_read[okw], np.zeros(okw.sum()),
                                                                post[okw], ones[okw])
    okw = np.isfinite(w_var)
    metrics["weight_regression"]["variance"] = position_regression(
        w_var[okw], np.zeros(okw.sum()), post[okw], ones[okw])
    arrays = {k: {"w": w, "sigma_w": s} for k, (w, s) in estimators.items()}
    arrays["flags_vis"], arrays["flags_prop"], arrays["precise"] = flags_v, flags_p, precise
    arrays["abs_delta_vis"], arrays["abs_delta_eff"] = ad, np.sqrt(dv ** 2 + dp ** 2)
    arrays["variance_regression"] = {
        "vis": {"x": (post * d["fused_var"] + (1 - post) * d["seg_vis_var"]
                      + post * (1 - post) * dv ** 2) - d["seg_vis_var"],
                "y": pred[:, ivv] - d["seg_vis_var"]},
        "prop": {"x": (post * d["fused_var"] + (1 - post) * d["seg_prop_var"]
                       + post * (1 - post) * dp ** 2) - d["seg_prop_var"],
                 "y": pred[:, ipv] - d["seg_prop_var"]}}
    return metrics, arrays


# --------------------------------------------------------------------------- #
# readability of the per-trial ratio: what it can show and what it cannot
# --------------------------------------------------------------------------- #
def predicted_outside(delta, post, sd, lim=RATIO_LIM, weights=None):
    """Fraction of per-trial ratios w = (network - seg)/Delta expected outside
    `lim`, for a network that is Bayes-optimal up to a Gaussian position
    error of sd `sd` that does not depend on Delta.

    On such a network w = p + e/Delta on every trial (p the analytical
    posterior, e the error), so the ratio leaves [lo, hi] exactly when
    e/Delta < lo - p or e/Delta > hi - p:

        P(outside | Delta, p) = Phi(-(p - lo)|Delta|/sd) + Phi(-(hi - p)|Delta|/sd)

    averaged over the trials (with importance `weights` if given). Nothing
    about the network enters except the one number sd, so this can be
    evaluated for a configuration before anything is trained, for any sd one
    is prepared to assume -- and, on a trained run, with the run's own
    residual sd, in which case the difference from the observed fraction
    measures how far the error is from Gaussian-and-independent-of-Delta
    (in practice it is smaller on small-|Delta| trials, so this over-predicts).
    """
    from scipy.stats import norm

    delta, post = np.asarray(delta, float), np.asarray(post, float)
    ok = np.isfinite(delta) & (delta != 0) & np.isfinite(post)
    ad, p = np.abs(delta[ok]), post[ok]
    if sd <= 0:
        return 0.0
    prob = norm.cdf(-(p - lim[0]) * ad / sd) + norm.cdf(-(lim[1] - p) * ad / sd)
    if weights is None:
        return float(prob.mean())
    w = np.asarray(weights, float)[ok]
    return float((w * prob).sum() / w.sum())


def flat_posterior_weights(post, n_bins=10):
    """Importance weights, one per trial and summing to one, under which the
    histogram of the analytical posterior is flat over `n_bins` equal-width
    bins: every trial is weighted by the inverse of its bin's share.

    Averaging a per-trial quantity with these weights says what its summary
    would be if the trials had been drawn with a uniform distribution of the
    optimal weight -- the evaluation-side reading of a 'balanced weight
    distribution' -- for the same network, on the same trials, with nothing
    removed. Bins that hold no trial get no weight.
    """
    post = np.asarray(post, float)
    idx = np.clip((post * n_bins).astype(int), 0, n_bins - 1)
    share = np.bincount(idx, minlength=n_bins) / post.size
    w = np.where(share[idx] > 0, 1.0 / np.maximum(share[idx], 1e-300), 0.0)
    return w / w.sum()


def readability(estimate, seg, fused, target, post, sig_out, C=None, lim=RATIO_LIM,
                n_bins=10):
    """Why the per-trial ratio looks the way it does on one run, one read.

    Every quantity is computed on every trial (Delta != 0 for the ratio);
    nothing is filtered. json-able. The pieces:

        own_sd                 sd of the run's own error, network - analytical
                               target -- the noise the ratio actually divides
                               by Delta (it contains any misweighting, so it
                               exceeds sigma_out unless the run is optimal)
        own_sd_absdelta_lt1    the same on trials with |Delta| < 1 deg, and
        own_sd_absdelta_gt4    with |Delta| > 4 deg: the error is not one
                               number, it is smaller where the hypotheses
                               nearly coincide
        frac_absdelta_lt1      share of trials with |Delta| < 1 deg -- the
                               trials the ratio cannot read whatever the
                               network does
        outside                observed fraction of ratios outside `lim`
        outside_predicted_*    the fraction predicted_outside gives with the
                               run's own sd, and with sigma_out (what an
                               optimal network with this control's read-out
                               noise would show)
        outside_if_flat_posterior, frac_absdelta_lt1_if_flat_posterior
                               the observed fraction and the small-Delta share
                               re-weighted so that the posterior histogram is
                               flat (flat_posterior_weights)
        outside_share_from_absdelta_lt1
                               of the ratios outside `lim`, the share that
                               comes from |Delta| < 1 deg trials
        delta_sd_c1/c2, own_sd_c1/c2 (when C is given)
                               Delta and the error by true cause: on C = 1
                               trials |Delta| is set by the measurement noise
                               alone, which is where the ratio has to work
    """
    estimate, seg, fused = (np.asarray(a, float) for a in (estimate, seg, fused))
    target, post = np.asarray(target, float), np.asarray(post, float)
    delta, e = fused - seg, estimate - target
    ad = np.abs(delta)
    ok = delta != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(ok, (estimate - seg) / np.where(ok, delta, 1.0), np.nan)
    outside = ok & ((w < lim[0]) | (w > lim[1]))
    own_sd = float(e.std())
    flat = flat_posterior_weights(post, n_bins)
    small, large = ad < 1, ad > 4
    sig_out = float(sig_out)
    out = {
        "lim": [float(lim[0]), float(lim[1])],
        "own_sd": own_sd, "sigma_out": sig_out,
        "own_sd_over_sigma_out": own_sd / sig_out if sig_out > 0 else np.nan,
        "own_sd_absdelta_lt1": float(e[small].std()) if small.sum() > 1 else np.nan,
        "own_sd_absdelta_gt4": float(e[large].std()) if large.sum() > 1 else np.nan,
        "frac_absdelta_lt1": float(small.mean()),
        "median_absdelta": float(np.median(ad)),
        "median_absdelta_over_own_sd": float(np.median(ad) / own_sd) if own_sd > 0 else np.nan,
        "outside": float(outside.mean()),
        "outside_predicted_own_sd": predicted_outside(delta, post, own_sd, lim),
        "outside_predicted_sigma_out": predicted_outside(delta, post, sig_out, lim),
        "outside_if_flat_posterior": float((flat * outside).sum()),
        "frac_absdelta_lt1_if_flat_posterior": float((flat * small).sum()),
        "outside_share_from_absdelta_lt1": (float(np.mean(ad[outside] < 1))
                                            if outside.any() else 0.0),
    }
    if C is not None:
        c1 = np.asarray(C) == 1
        if c1.any() and (~c1).any():
            out.update(delta_sd_c1=float(delta[c1].std()), delta_sd_c2=float(delta[~c1].std()),
                       own_sd_c1=float(e[c1].std()), own_sd_c2=float(e[~c1].std()))
    return out


# --------------------------------------------------------------------------- #
# reliability levels
# --------------------------------------------------------------------------- #
def assign_levels(values, levels):
    """Split trials by a reliability variable.

    `levels` is either an int -- that many quantile bins, so every trial lands
    in exactly one bin and the bins hold equal numbers of trials -- or a list
    of centres, in which case each trial goes to the nearest centre (the
    convention of analysis.by_reliability and the config's reliability_levels).

    Returns (bin index per trial, list of (label, representative value)).
    """
    values = np.asarray(values, float)
    if np.isscalar(levels) or isinstance(levels, (int, np.integer)):
        k = int(levels)
        edges = np.quantile(values, np.linspace(0, 1, k + 1))
        idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, k - 1)
        info = [(f"{edges[b]:.2g}-{edges[b + 1]:.2g}",
                 float(np.median(values[idx == b]))) for b in range(k)]
    else:
        centres = np.asarray(levels, float)
        idx = np.abs(values[:, None] - centres[None, :]).argmin(1)
        info = [(f"~{c:g}", float(c)) for c in centres]
    return idx, info


def weight_by_reliability(pred, d, names, sig_out, acfg, input_key, levels,
                          grid=GRID, min_count=LEVEL_MIN_COUNT, max_se=LEVEL_MAX_SE):
    """Per reliability level of one input, the least-squares weight curve read
    from each output, and the analytical posterior on the same bins.

    The transition midpoint is fitted to the binned curve for the network and
    for the analytical posterior alike, so the two are comparable; NaN when a
    level keeps too few bins for the logistic to be identifiable.

    Returns {level_label: {"value": v, "n": n, "analytical": (c, m),
                           "vis": (c, w, n, se), "prop": (...),
                           "midpoints": {...}}}
    """
    acfg = dict(acfg)
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    var = d[INPUTS[input_key]]
    idx, info = assign_levels(var, levels)
    out = {}
    for b, (label, value) in enumerate(info):
        m = idx == b
        if m.sum() < min_count:
            continue
        disp, post = d["disparity"][m], d["post_c1"][m]
        c_a, m_a, _ = mean_by_bin(disp, post, grid)
        entry = {"value": value, "n": int(m.sum()), "analytical": (c_a, m_a),
                 "grid": grid,
                 "midpoints": {"analytical": transition_fit(np.abs(c_a), m_a)[0]}}
        for key, out_name, seg_name in READS:
            i = names.index(out_name)
            curve = binned_implied_weight(
                pred[m, i], d[seg_name][m], d["fused_mu"][m], disp, grid,
                min_count=min_count, sigma_out=float(sig_out[i]), max_se=max_se)
            entry[key] = curve
            entry["midpoints"][key] = transition_fit(np.abs(curve[0]), curve[1])[0]
        out[label] = entry
    return out


# --------------------------------------------------------------------------- #
# figures: sigma_out and sigma_w
# --------------------------------------------------------------------------- #
def _ramp(n, name="viridis"):
    return plt.get_cmap(name)(np.linspace(0.15, 0.9, n))


def _on_grid(grid, centres, *values):
    """Map per-bin values onto the full grid, NaN where the bin was dropped,
    so that a dropped bin reads as a gap in the line and not as a segment
    drawn straight across it (as in scripts/05_prior_sweep.py)."""
    grid = np.asarray(grid, float)
    out = [np.full(grid.shape, np.nan) for _ in values]
    for c, *vals in zip(np.asarray(centres, float), *values, strict=False):
        j = int(np.abs(grid - c).argmin())
        for o, v in zip(out, vals, strict=False):
            o[j] = v
    return (grid, *out)


def fig_sigma_residuals(arrays, ctrl_resid, names, metrics):
    """The run's own per-trial residuals, network - analytical target, on
    every test trial, per position output. Their std is what stage 3 stores
    as residual_std. The p_common = 1 control's residuals are drawn as a thin
    outline for reference: their std is sigma_out, and the difference between
    the two widths is the causal misweighting the run carries on top of its
    read-out noise. Nothing is filtered; the axis is clipped at +/-4 sd for
    display and the fraction beyond it is printed."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        i = names.index(out_name)
        own = arrays[key]["resid"]
        so = metrics["sigma_out"][out_name]
        own_sd = metrics["residual_std_self"][out_name]
        lim = 4 * max(so, own_sd)
        bins = np.linspace(-lim, lim, 81)
        beyond = np.mean(np.abs(own) > lim)
        ax.hist(np.clip(own, -lim, lim), bins=bins, density=True,
                color=COLORS["network"], alpha=0.75,
                label=f"this run, all {own.size} trials (sd {own_sd:.3f})")
        ax.hist(np.clip(ctrl_resid[:, i], -lim, lim), bins=bins, density=True,
                histtype="step", lw=1.3, color="0.25",
                label=f"p_common=1 control (sd {so:.3f} = sigma_out)")
        for s, c, ls in ((own_sd, COLORS["network"], "-"), (so, "0.25", "--")):
            ax.axvline(s, color=c, ls=ls, lw=1.0)
            ax.axvline(-s, color=c, ls=ls, lw=1.0)
        ax.set(xlabel=f"{out_name}: network - analytical (deg)", ylabel="density",
               title=f"{out_name}: residual sd {own_sd:.3f} deg")
        if beyond > 0:
            ax.text(0.98, 0.97, f"{100 * beyond:.1f}% beyond +/-{lim:.1f}",
                    transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color="0.35")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_weight_distribution(arrays, metrics, lim=(-1.0, 2.0)):
    """The network's estimated weight on every trial, w = (network - seg) /
    Delta, with no filter of any kind. A Bayes-optimal observer would put
    each trial at its posterior, so the analytical posterior's own
    distribution is drawn as an outline for comparison. The ratio is
    unbounded where Delta is small, so the axis is clipped to `lim` and the
    fraction of trials outside it is printed."""
    post = arrays["post_c1"]
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    bins = np.linspace(lim[0], lim[1], 91)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        w = arrays[key]["w_raw"]
        w = w[np.isfinite(w)]
        s = metrics["reads"][key]["w_raw"]
        ax.hist(np.clip(w, *lim), bins=bins, density=True, color=COLORS["network"],
                alpha=0.75, label=f"network, all {w.size} trials")
        ax.hist(post, bins=bins, density=True, histtype="step", lw=1.3,
                color=COLORS["analytical"], label="analytical posterior")
        ax.axvline(s["median"], color=COLORS["network"], lw=1.0)
        ax.set(xlabel=f"w = (network - seg) / Delta, from {out_name}", xlim=lim,
               title=f"{out_name}: median {s['median']:.2f} "
                     f"(IQR {s['iqr'][0]:.2f} to {s['iqr'][1]:.2f})")
        ax.text(0.02, 0.97,
                f"{100 * s['frac_outside_-1_2']:.1f}% of trials outside [{lim[0]:g}, {lim[1]:g}]"
                f"\n(piled into the edge bins)",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.35")
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("density")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_network_minus_seg(arrays, metrics):
    """How far the network's estimate sits from the segregated solution on
    every trial, network - seg, next to how far the fused solution sits from
    it, Delta = fused - seg. The first is w * Delta trial by trial, so a
    network that fuses puts mass where Delta is and a network that
    segregates piles up at zero. No filter."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        nms, delta = arrays[key]["net_minus_seg"], arrays[key]["delta"]
        lim = 4 * max(nms.std(), delta.std())
        bins = np.linspace(-lim, lim, 81)
        ax.hist(np.clip(delta, -lim, lim), bins=bins, density=True, color="0.7",
                alpha=0.8, label=f"Delta = fused - seg (sd {delta.std():.2f})")
        ax.hist(np.clip(nms, -lim, lim), bins=bins, density=True, histtype="step",
                lw=1.5, color=COLORS["network"],
                label=f"network - seg (sd {nms.std():.2f})")
        ax.set(xlabel=f"degrees, {out_name}", ylabel="density (log)", yscale="log",
               title=f"{out_name}: network - seg, all {nms.size} trials")
        ax.legend(fontsize=7.5, frameon=False)
    label_panels(axes)
    fig.tight_layout()
    return fig



def fig_sigma_w_vs_delta(arrays, metrics, n_show=4000, seed=0):
    """Per trial, against |Delta| = |fused - seg|: the network's actual error
    (left) and sigma_w (right), for each position output.

    Left: |network - analytical| does not grow with |Delta| -- the running
    median stays at about sigma_out -- which is what licenses sigma_w =
    sigma_out / |Delta| as the per-trial uncertainty of the weight. Right:
    that ratio, coloured by the analytical posterior, with the criterion and
    the |Delta| it implies. Trials at high posterior sit at small |Delta|,
    which is why the filter removes the fusion end of every curve."""
    rng = np.random.default_rng(seed)
    crit = metrics["sigma_w_criterion"]
    fig, axes = plt.subplots(2, 2, figsize=SIZE["quad"], constrained_layout=True)
    for row, (key, out_name, _) in zip(axes, READS, strict=False):
        a = arrays[key]
        ad = np.abs(a["delta"])
        ok = np.isfinite(a["sigma_w"]) & (ad > 0)
        pick = np.flatnonzero(ok)
        if pick.size > n_show:
            pick = rng.choice(pick, n_show, replace=False)
        so = metrics["sigma_out"][out_name]
        thr = metrics["reads"][key]["delta_threshold_deg"]

        ax = row[0]
        ax.scatter(ad[pick], np.abs(a["resid"][pick]), s=3, color=COLORS["network"],
                   alpha=0.3, linewidths=0, rasterized=True)
        edges = np.quantile(ad[ok], np.linspace(0, 1, 13))
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            m = ok & (ad >= lo) & (ad < hi)
            if m.sum() >= 20:
                mids.append(np.median(ad[m]))
                meds.append(np.median(np.abs(a["resid"][m])))
        ax.plot(mids, meds, "o-", color="0.15", ms=3.5, lw=1.4, label="running median")
        ax.axhline(so * np.sqrt(2 / np.pi), color="0.15", ls="--", lw=1.1,
                   label=f"E|error| for sigma_out {so:.2f}")
        ax.set(xscale="log", yscale="log", ylim=(1e-3, None),
               xlabel=f"|Delta| (deg), {out_name}", ylabel="|network - analytical| (deg)",
               title="the error does not grow with |Delta|")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

        ax = row[1]
        sc = ax.scatter(ad[pick], a["sigma_w"][pick], c=arrays["post_c1"][pick], s=4,
                        cmap="viridis", vmin=0, vmax=1, alpha=0.7, linewidths=0,
                        rasterized=True)
        ax.axhline(crit, color="0.2", ls="--", lw=1.1)
        ax.axvline(thr, color="0.2", ls=":", lw=1.1)
        ax.text(0.97, 0.95, f"criterion {crit:g}  <=>  |Delta| > {thr:.1f} deg\n"
                f"keeps {100 * metrics['reads'][key]['frac_readable']:.0f}% of trials",
                transform=ax.transAxes, fontsize=8, va="top", ha="right")
        ax.set(xscale="log", yscale="log", xlabel=f"|Delta| (deg), {out_name}",
               ylabel="sigma_w = sigma_out / |Delta|",
               title="readability of the weight, per trial")
    cb = fig.colorbar(sc, ax=axes[:, 1], shrink=0.8, pad=0.02)
    cb.set_label("analytical posterior p(C=1|x)", fontsize=9)
    label_panels(axes)
    return fig


def fig_sigma_w_vs_posterior(arrays, metrics, n_bins=10):
    """Median sigma_w (with interquartile band) per posterior bin, and the
    fraction of the bin the criterion keeps. This is the figure that explains
    why the filtered estimator cannot see the fusion end of the curve."""
    crit = metrics["sigma_w_criterion"]
    post = arrays["post_c1"]
    edges = np.linspace(0, 1, n_bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        sw = arrays[key]["sigma_w"]
        c, med, lo, hi, kept = [], [], [], [], []
        for b in range(n_bins):
            top = post <= edges[b + 1] if b == n_bins - 1 else post < edges[b + 1]
            m = (post >= edges[b]) & top & np.isfinite(sw)
            if m.sum() < 5:
                continue
            c.append(post[m].mean())
            med.append(np.median(sw[m]))
            lo.append(np.percentile(sw[m], 25))
            hi.append(np.percentile(sw[m], 75))
            kept.append(np.mean(sw[m] < crit))
        c, med, lo, hi, kept = map(np.asarray, (c, med, lo, hi, kept))
        ax.fill_between(c, lo, hi, color=COLORS["network"], alpha=0.2, lw=0)
        ax.plot(c, med, "o-", color=COLORS["network"], ms=4, label="median sigma_w (IQR band)")
        ax.axhline(crit, color="0.2", ls="--", lw=1.1, label=f"criterion {crit:g}")
        ax.set(yscale="log", xlabel="analytical posterior p(C=1|x)",
               ylabel="sigma_w per trial", xlim=(0, 1), title=f"read from {out_name}")
        ax2 = ax.twinx()
        ax2.plot(c, 100 * kept, ":", color="0.45", lw=1.3)
        ax2.set_ylim(-4, 104)
        ax2.set_ylabel("trials kept by the criterion (%)", color="0.35", fontsize=9)
        ax2.tick_params(axis="y", colors="0.35", labelsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_kept_vs_criterion(arrays, metrics):
    """Fraction of trials the sigma_w filter keeps, as the criterion is varied,
    overall and within the ambiguous zone (0.2 < posterior < 0.8)."""
    crit = metrics["sigma_w_criterion"]
    grid = np.logspace(-2, 0, 60)
    post = arrays["post_c1"]
    amb = (post > 0.2) & (post < 0.8)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for (key, out_name, _), color in zip(READS, (COLORS["visual"], COLORS["prop"]), strict=False):
        sw = arrays[key]["sigma_w"]
        ax.plot(grid, [np.mean(sw < g) for g in grid], "-", color=color,
                label=f"{out_name}, all trials")
        ax.plot(grid, [np.mean(sw[amb] < g) for g in grid], "--", color=color,
                label=f"{out_name}, 0.2 < posterior < 0.8")
    ax.axvline(crit, color="0.2", ls=":", lw=1.1)
    ax.text(crit, 0.02, f" criterion {crit:g}", fontsize=8, ha="left")
    ax.set(xscale="log", xlabel="sigma_w criterion", ylabel="fraction of trials kept",
           ylim=(0, 1.02), title="what the filter keeps")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


DELTA_BANDS = (0.0, 1.0, 2.0, 4.0, 8.0, np.inf)


def weight_by_delta_table(arrays, post, bands=DELTA_BANDS, lim=(-1.0, 2.0)):
    """The per-trial ratio w = (network - seg)/Delta grouped by |Delta|, on
    every trial with Delta != 0, per read. For each band: the share of all
    trials, the share of the least-squares weight (sum of Delta^2) the band
    carries, the fraction of its ratios outside `lim`, median and IQR, and
    the mean analytical posterior.

    This is the bridge between the two estimators. Within a bin the
    least-squares weight is exactly the Delta^2-weighted mean of the
    per-trial ratios, sum(Delta^2 w_i)/sum(Delta^2), so a band that holds a
    large share of the trials but a tiny share of Delta^2 can fill the
    histogram with wild ratios while barely touching the least-squares curve.
    """
    out = {}
    for key, _, _ in READS:
        w, delta = arrays[key]["w_raw"], arrays[key]["delta"]
        ok = np.isfinite(w)
        ad = np.abs(delta)
        d2_total = float((ad[ok] ** 2).sum())
        rows = []
        for lo, hi in zip(bands[:-1], bands[1:], strict=False):
            m = ok & (ad >= lo) & (ad < hi)
            if not m.any():
                continue
            wm = w[m]
            rows.append({
                "band": [float(lo), None if np.isinf(hi) else float(hi)],
                "label": f"|Delta| {lo:g}-{hi:g}" if np.isfinite(hi) else f"|Delta| >= {lo:g}",
                "n": int(m.sum()), "share_trials": float(m.mean()),
                "share_delta2": float((ad[m] ** 2).sum() / d2_total),
                "frac_outside": float(np.mean((wm < lim[0]) | (wm > lim[1]))),
                "median": float(np.median(wm)),
                "iqr": [float(np.percentile(wm, 25)), float(np.percentile(wm, 75))],
                "mean_posterior": float(post[m].mean()),
            })
        out[key] = rows
    return out


def fig_weight_by_delta(arrays, metrics, bands=DELTA_BANDS, lim=(-1.0, 2.0)):
    """The per-trial ratio's distribution, every trial, split by |Delta|.

    Same ratios as figure 05, nothing removed: the trials are only grouped
    by how far apart the two hypotheses were, and each group is drawn as
    its own density (each normalised to itself, so groups of different size
    are comparable in shape). The legend gives each group's share of the
    trials, its share of sum(Delta^2) -- which is its weight in the
    least-squares estimate -- and the fraction of its ratios outside the
    axis. Axis clipped to `lim` with the overflow piled into the edge bins,
    as in figure 05."""
    table = weight_by_delta_table(arrays, arrays["post_c1"], bands, lim)
    colors = _ramp(len(bands) - 1, "plasma")
    edges = np.linspace(lim[0], lim[1], 91)
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        w, ad = arrays[key]["w_raw"], np.abs(arrays[key]["delta"])
        ok = np.isfinite(w)
        for row, color, (lo, hi) in zip(table[key], colors,
                                        zip(bands[:-1], bands[1:], strict=False), strict=False):
            m = ok & (ad >= lo) & (ad < hi)
            ax.hist(np.clip(w[m], *lim), bins=edges, density=True, histtype="step",
                    lw=1.5, color=color,
                    label=(f"{row['label']}: {100 * row['share_trials']:.0f}% of trials, "
                           f"{100 * row['share_delta2']:.1f}% of Delta^2, "
                           f"{100 * row['frac_outside']:.0f}% outside"))
        ax.hist(arrays["post_c1"], bins=edges, density=True, histtype="step", lw=1.1,
                color=COLORS["analytical"], ls="--", label="analytical posterior, all trials")
        ax.set(xlabel=f"w = (network - seg) / Delta, from {out_name}", xlim=lim,
               title=f"{out_name}, all {int(ok.sum())} trials, grouped by |Delta|")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("density (each group to itself)")
    label_panels(axes)
    fig.tight_layout()
    return fig


ESTIMATOR_LABEL = {"ratio_vis": "ratio, mu_vis", "ratio_prop": "ratio, mu_prop",
                   "joint": "both position outputs", "variance": "both variance outputs",
                   "combined": "all four outputs"}
ESTIMATOR_COLOR = {"ratio_vis": COLORS["visual"], "ratio_prop": COLORS["prop"],
                   "joint": COLORS["network"], "variance": "#ff7f0e", "combined": "#111111"}


def fig_weight_from_variance(arrays, metrics, n_show=4000, seed=0, lim=(-0.25, 1.25)):
    """The per-trial weight read four ways, on every trial, nothing filtered.

    A: its distribution -- the ratio from mu_vis (as in figure 05), the
    least-squares read across both position outputs, the read from the two
    variance outputs and the combination of all four -- against the
    analytical posterior's; the axis is clipped to `lim` with the overflow
    piled into the edge bins, and the fraction outside [-1, 2] is printed.
    B: the variance read against the analytical posterior, trial by trial
    (a subsample), with its binned mean. C: the per-trial uncertainty of the
    position-based and the variance-based read against |Delta|: the first
    grows without bound as Delta -> 0, the second does not. D: coherence --
    per posterior bin, the position read and the variance read on the trials
    where the position read is precise; a network that mixes both outputs
    with one weight puts these on the identity."""
    rng = np.random.default_rng(seed)
    pt, m = arrays["per_trial"], metrics["per_trial"]
    post = arrays["post_c1"]
    crit = metrics["sigma_w_criterion"]
    fig, axes = plt.subplots(2, 2, figsize=SIZE["quad"], constrained_layout=True)

    ax = axes[0, 0]
    edges = np.linspace(lim[0], lim[1], 76)
    for key in ("ratio_vis", "joint", "variance", "combined"):
        w = pt[key]["w"]
        w = w[np.isfinite(w)]
        e = m["estimators"][key]
        ax.hist(np.clip(w, *lim), bins=edges, density=True, histtype="step", lw=1.5,
                color=ESTIMATOR_COLOR[key],
                label=f"{ESTIMATOR_LABEL[key]}: {100 * e['outside']:.1f}% outside [-1, 2]")
    ax.hist(post, bins=edges, density=True, color="0.75", alpha=0.5, lw=0,
            label="analytical posterior")
    ax.set(xlabel="per-trial weight (clipped to the axis)", ylabel="density", xlim=lim,
           title="the weight on every trial, four reads")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center")

    ax = axes[0, 1]
    w = pt["variance"]["w"]
    ok = np.flatnonzero(np.isfinite(w))
    pick = rng.choice(ok, min(n_show, ok.size), replace=False)
    ax.scatter(post[pick], w[pick], s=3, color=ESTIMATOR_COLOR["variance"], alpha=0.25,
               linewidths=0, rasterized=True)
    bp = m["estimators"]["variance"]["by_posterior"]
    ax.plot(bp["centres"], bp["mean"], "o-", color="0.15", ms=4, lw=1.4, label="binned mean")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["analytical"], lw=1.1, label="identity")
    e = m["estimators"]["variance"]
    ax.text(0.03, 0.97, f"corr {e['corr_with_posterior']:.3f}\nsd vs posterior "
            f"{e['sd_vs_posterior']:.3f} (|Delta| < 1: {e['sd_vs_posterior_absdelta_lt1']:.3f})",
            transform=ax.transAxes, fontsize=7.5, va="top")
    ax.set(xlabel="analytical posterior p(C=1|x)", ylabel="weight from the variance outputs",
           xlim=(0, 1), ylim=lim, title="variance read, trial by trial")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    ax = axes[1, 0]
    ad = arrays["per_trial"]["abs_delta_vis"]
    ok = np.flatnonzero(np.isfinite(pt["variance"]["sigma_w"]) & (ad > 0))
    pick = rng.choice(ok, min(n_show, ok.size), replace=False)
    for key in ("joint", "variance"):
        ax.scatter(ad[pick], pt[key]["sigma_w"][pick], s=3, color=ESTIMATOR_COLOR[key],
                   alpha=0.3, linewidths=0, rasterized=True,
                   label=f"{ESTIMATOR_LABEL[key]} (median "
                         f"{m['estimators'][key]['sigma_w_median']:.3f}, "
                         f"{100 * m['estimators'][key]['frac_sigma_w_lt_crit']:.0f}% < {crit:g})")
    ax.axhline(crit, color="0.2", ls="--", lw=1.1)
    ax.set(xscale="log", yscale="log", xlabel="|Delta| (deg), mu_vis",
           ylabel="per-trial sigma_w (from the control's sigma_out)",
           title="readability per trial, position vs variance read")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    ax = axes[1, 1]
    rows = m["coherence"]["by_posterior"]
    c = np.array([r["mean_posterior"] for r in rows])
    jp = np.array([r["joint_precise"] for r in rows])
    vp = np.array([r["variance_precise"] for r in rows])
    va = np.array([r["variance_all"] for r in rows])
    npx = np.array([r["n_precise"] for r in rows])
    ax.plot([0, 1], [0, 1], "--", color=COLORS["analytical"], lw=1.1, label="identity")
    ax.plot(c, va, "s-", color=ESTIMATOR_COLOR["variance"], ms=4, lw=1.2, alpha=0.5,
            label="variance read, all trials in the bin")
    ax.plot(c, jp, "o-", color=ESTIMATOR_COLOR["joint"], ms=4, lw=1.4,
            label="position read, precise trials")
    ax.plot(c, vp, "^-", color=ESTIMATOR_COLOR["variance"], ms=4, lw=1.4,
            label="variance read, same trials")
    for x, n in zip(c, npx, strict=False):
        if np.isfinite(x):
            ax.text(x, -0.08, f"{n}", fontsize=6, ha="center", color="0.4")
    co = m["coherence"]
    ax.text(0.97, 0.12, f"precise = position read's sigma_w < {co['precise_criterion']:g} "
            f"({100 * co['frac_precise']:.0f}% of trials)\nvariance - position "
            f"{co['mean_diff_variance_minus_joint']:+.3f} "
            f"(sd {co['sd_diff_variance_minus_joint']:.3f})",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right")
    ax.set(xlabel="analytical posterior p(C=1|x) (n precise below)", ylabel="mean weight",
           xlim=(0, 1), ylim=(-0.12, 1.1), title="coherence of the two reads")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    label_panels(axes)
    return fig


def fig_fusion_weight_variance(arrays, metrics, grid):
    """The pipeline's 04_fusion_weight, read from the variance outputs.

    Same axes, same disparity grid, same analytical curve as figure 04: the
    per-trial weight is averaged per disparity bin. The pipeline's reads
    (the ratio from mu_vis and from mu_prop, NaN within min_separation of
    zero disparity) are drawn faded for comparison; the variance reads from
    var_vis and var_prop, which exist on every trial, are drawn in full and
    carry the curve through zero disparity, where the ratio has nothing."""
    pt = arrays["per_trial"]
    disp, post = arrays["disparity"], arrays["post_c1"]
    grid = np.asarray(grid, float)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for key, color, mk, lab in (("vis", COLORS["network"], "o", "mu_vis"),
                                ("prop", COLORS["prop"], "s", "mu_prop")):
        c, m, _ = mean_by_bin(disp, arrays[key]["w"], grid)
        ax.plot(c, m, f"{mk}-", color=color, alpha=0.3, lw=1.0, ms=3,
                label=f"ratio from {lab} (pipeline figure 04)")
    for key, color, mk, lab in (("variance_vis", COLORS["network"], "o", "var_vis"),
                                ("variance_prop", COLORS["prop"], "s", "var_prop")):
        c, m, _ = mean_by_bin(disp, pt[key]["w"], grid)
        ax.plot(c, m, f"{mk}-", color=color, lw=1.8, ms=5,
                label=f"implied, from {lab} (variance read)")
    # the analytical curve last, so it stays visible where the reads sit on it
    c, m, _ = mean_by_bin(disp, post, grid)
    ax.plot(c, m, "--", color=COLORS["analytical"], lw=1.3, zorder=5, label="analytical p(C=1)")
    ax.plot(c, m, "o", color=COLORS["analytical"], ms=3, zorder=6)
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1),
           title="fusion -> segregation transition, weight read from the variance outputs")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def fig_variance_regression(arrays, metrics):
    """The pipeline's 08_position_regression, in the variance domain.

    A: the no-division headline statistic with the variance outputs in the
    role of the position outputs -- the network's variance reduction relative
    to segregation, var_out - var_seg, against the optimal mixture's,
    v(p) - var_seg, with the least-squares fit (slope 1 = Bayes-optimal).
    Like the position regression it needs no per-trial weight and every
    trial enters with its natural leverage; unlike it, the leverage is
    largest where the mixture's hump term p(1-p)Delta^2 is largest.
    B: the weight read from both variance outputs against the analytical
    posterior, with its fit -- the same test in the weight domain, now
    possible because that read exists on every trial."""
    pt, m = arrays["per_trial"], metrics["per_trial"]
    post = arrays["post_c1"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])

    ax = axes[0]
    x, y = pt["variance_regression"]["vis"]["x"], pt["variance_regression"]["vis"]["y"]
    reg = m["variance_regression"]["vis"]
    ax.scatter(x, y, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"],
            label="Bayes-optimal (slope 1)")
    ax.plot([lo, hi], [reg["slope"] * lo + reg["intercept"], reg["slope"] * hi + reg["intercept"]],
            lw=1.5, color=COLORS["network"],
            label=f"fit: slope {reg['slope']:.2f} "
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}]")
    ax.set(xlabel="v_opt - var_seg  (deg^2)", ylabel="var_vis output - var_seg  (deg^2)",
           title="variance-domain regression: var_vis")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    ax = axes[1]
    w = pt["variance"]["w"]
    ok = np.flatnonzero(np.isfinite(w))
    pick = rng.choice(ok, min(6000, ok.size), replace=False)
    ax.scatter(post[pick], w[pick], s=2, alpha=0.15, color=ESTIMATOR_COLOR["variance"],
               rasterized=True)
    reg = m["weight_regression"]["variance"]
    ax.plot([0, 1], [0, 1], "--", lw=1, color=COLORS["analytical"], label="Bayes-optimal (slope 1)")
    ax.plot([0, 1], [reg["intercept"], reg["slope"] + reg["intercept"]], lw=1.5,
            color=ESTIMATOR_COLOR["variance"],
            label=f"fit: slope {reg['slope']:.2f} "
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}], "
                  f"intercept {reg['intercept']:+.2f}")
    ax.set(xlabel="analytical posterior p(C=1|x)", ylabel="weight from the variance outputs",
           xlim=(0, 1), ylim=(-0.25, 1.25), title="weight-domain regression, every trial")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def sigma_figures(arrays, ctrl_resid, names, metrics):
    """Nothing here is filtered. The sigma_w criterion appears only as a
    reference line, and figure 04 shows what applying it *would* keep."""
    figs = {
        "01_residuals_all_trials": fig_sigma_residuals(arrays, ctrl_resid, names, metrics),
        "02_sigma_w_vs_delta": fig_sigma_w_vs_delta(arrays, metrics),
        "03_sigma_w_vs_posterior": fig_sigma_w_vs_posterior(arrays, metrics),
        "04_kept_vs_criterion": fig_kept_vs_criterion(arrays, metrics),
        "05_weight_distribution_all_trials": fig_weight_distribution(arrays, metrics),
        "06_network_minus_seg_all_trials": fig_network_minus_seg(arrays, metrics),
        "07_weight_by_delta_all_trials": fig_weight_by_delta(arrays, metrics),
    }
    if metrics.get("per_trial"):
        figs["08_weight_from_variance_all_trials"] = fig_weight_from_variance(arrays, metrics)
    return figs


def print_per_trial(metrics, crit=None):
    """The per-trial estimators side by side, for the console."""
    m = metrics.get("per_trial")
    if not m:
        return
    crit = metrics.get("sigma_w_criterion", 0.1) if crit is None else crit
    print("\nthe weight on every trial, read four ways (sd vs the analytical posterior; "
          "sigma_w from the control's sigma_out):")
    head = "s_w<" + f"{crit:g}"
    print(f"  {'read':24s} {'outside':>7} {'sd all':>7} {'sd |D|<1':>8} {'sd |D|>4':>8} "
          f"{'corr':>6} {'med s_w':>7} {head:>7} | mean w at p~0.05 / 0.5 / 0.85 / 0.95")
    for key in ("ratio_vis", "ratio_prop", "joint", "variance_vis", "variance_prop",
                "variance", "combined"):
        e = m["estimators"][key]
        bp = e["by_posterior"]
        want = {}
        for c, w in zip(bp["centres"], bp["mean"], strict=False):
            for target in (0.05, 0.5, 0.85, 0.95):
                if abs(c - target) < 0.06 and target not in want:
                    want[target] = w
        bins = " / ".join(f"{want[t]:.2f}" if t in want else "  - "
                          for t in (0.05, 0.5, 0.85, 0.95))
        print(f"  {key:24s} {100 * e['outside']:6.1f}% {e['sd_vs_posterior']:7.3f} "
              f"{e['sd_vs_posterior_absdelta_lt1']:8.3f} {e['sd_vs_posterior_absdelta_gt4']:8.3f} "
              f"{e['corr_with_posterior']:6.3f} {e['sigma_w_median']:7.3f} "
              f"{100 * e['frac_sigma_w_lt_crit']:6.0f}% | {bins}")
    co, fl = m["coherence"], m["variance_root_flags"]["vis"]
    print(f"  variance read: unique root on {100 * fl['unique']:.0f}% of trials, two roots "
          f"resolved by the position read on {100 * fl['two_roots']:.0f}%, above any mixture "
          f"on {100 * fl['above_parabola']:.1f}%")
    print(f"  coherence on the {100 * co['frac_precise']:.0f}% of trials where the position read "
          f"has sigma_w < {co['precise_criterion']:g}: variance - position = "
          f"{co['mean_diff_variance_minus_joint']:+.3f} "
          f"(sd {co['sd_diff_variance_minus_joint']:.3f}); against the posterior the position "
          f"read is off by sd {co['sd_joint_vs_posterior_precise']:.3f}, the variance read by "
          f"{co['sd_variance_vs_posterior_precise']:.3f}")


# --------------------------------------------------------------------------- #
# figures: reliability
# --------------------------------------------------------------------------- #
def fig_weight_by_reliability(curves, input_key):
    """Weight against signed disparity at each reliability level of one input,
    read from mu_vis (left) and mu_prop (right). Solid: the network, by least
    squares within the bin; dashed, same colour: the analytical posterior at
    that level. The two dashed families differ between inputs because the
    optimal weight depends on which cue is noisy."""
    labels = list(curves)
    colors = _ramp(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        for label, color in zip(labels, colors, strict=False):
            e = curves[label]
            ca, ma = e["analytical"]
            ax.plot(ca, ma, "--", color=color, lw=1.1)
            g, w, se = _on_grid(e["grid"], e[key][0], e[key][1], e[key][3])
            ax.errorbar(g, w, yerr=1.96 * np.nan_to_num(se), fmt="o-", ms=3.5, lw=1.5,
                        capsize=2, color=color, label=f"{label} (n={e['n']})")
        ax.set(xlabel="body-frame disparity (deg)", ylim=(-0.1, 1.1),
               title=f"read from {out_name}")
        ax.plot([], [], "--", color="0.3", label="analytical, same level")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                  title=INPUT_LABEL[input_key], title_fontsize=7)
    axes[0].set_ylabel("weight on fused estimate")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_midpoint_by_reliability(curves, input_key):
    """Transition midpoint against the reliability level: network (both
    reads) and analytical. The prediction is that a noisier cue tolerates more
    disparity before segregation, so the midpoint moves outward."""
    vals = np.array([e["value"] for e in curves.values()])
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot(vals, [e["midpoints"]["analytical"] for e in curves.values()], "--o",
            color=COLORS["analytical"], ms=4, label="analytical posterior")
    for (key, out_name, _), color, mk in zip(READS, (COLORS["visual"], COLORS["prop"]),
                                             ("o", "s"), strict=False):
        ax.plot(vals, [e["midpoints"][key] for e in curves.values()], f"{mk}-",
                color=color, ms=4, label=f"network, read from {out_name}")
    ax.set(xlabel=f"{INPUT_LABEL[input_key]} (deg^2)",
           ylabel="transition midpoint (deg)",
           title="where fusion gives way to segregation")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def reliability_figures(pred, d, names, sig_out, acfg, inputs, levels, **kw):
    figs, table = {}, {}
    for n, key in enumerate(inputs, start=1):
        curves = weight_by_reliability(pred, d, names, sig_out, acfg, key, levels, **kw)
        if not curves:
            print(f"reliability: no level of {INPUTS[key]} holds enough trials; skipped")
            continue
        figs[f"{n:02d}a_weight_by_{INPUTS[key]}"] = fig_weight_by_reliability(curves, key)
        figs[f"{n:02d}b_midpoint_by_{INPUTS[key]}"] = fig_midpoint_by_reliability(curves, key)
        table[INPUTS[key]] = {
            label: {"value": e["value"], "n": e["n"],
                    "midpoint_analytical": e["midpoints"]["analytical"],
                    "midpoint_vis": e["midpoints"]["vis"],
                    "midpoint_prop": e["midpoints"]["prop"]}
            for label, e in curves.items()}
    return figs, table


# --------------------------------------------------------------------------- #
# figures: the three per-run figures this experiment grew out of
# --------------------------------------------------------------------------- #
def baseline_figures(pred, d, names, sig_out, acfg, arrays, metrics=None):
    """05_fusion_weight_by_reliability, 15/16_weight_vs_post_* exactly as the
    main pipeline draws them, plus the least-squares version of 05 -- and,
    when the per-trial block exists, 04 and 08 with the weight read from the
    variance outputs (04v, 08v). `metrics` is the weight_analysis result;
    without it the per-trial regressions are recomputed."""
    from cmsi.analysis.causal import by_reliability

    figs = {}
    curves = by_reliability(d["disparity"], arrays["vis"]["w"], d["sig2_vis"],
                            acfg["reliability_levels"], acfg["disparity_grid"])
    figs["05_fusion_weight_by_reliability"] = viz_results.fusion_weight_by_reliability(curves)
    figs["15_weight_vs_post_ratio"] = viz_results.implied_weight_vs_posterior(
        arrays["vis"]["by_post"], "ratio")
    figs["16_weight_vs_post_leastsq"] = viz_results.implied_weight_vs_posterior(
        arrays["vis"]["by_post"], "least_squares")
    figs["04_fusion_weight"] = viz_results.fusion_weight_curve(
        d["disparity"], arrays["vis"]["w"], d["post_c1"], acfg["disparity_grid"],
        w_prop=arrays["prop"]["w"])
    if arrays.get("per_trial"):
        pt = (metrics or {}).get("per_trial") or per_trial_weights(pred, d, names, sig_out)[0]
        figs["04v_fusion_weight_variance_read"] = fig_fusion_weight_variance(
            arrays, {"per_trial": pt}, acfg["disparity_grid"])
        figs["08v_variance_regression"] = fig_variance_regression(arrays, {"per_trial": pt})
    return figs


# --------------------------------------------------------------------------- #
# config overrides from the command line
# --------------------------------------------------------------------------- #
def parse_overrides(items):
    """['rf_width=4', 'n_vis=100'] -> {'rf_width': 4.0, 'n_vis': 100}, values
    parsed as yaml so lists and floats work: --set hidden=[32,32]."""
    import yaml

    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
