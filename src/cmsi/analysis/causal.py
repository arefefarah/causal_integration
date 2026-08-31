"""Is the network doing causal inference, and is it doing it optimally?

The network is never shown p(C=1). These functions ask what its behaviour
implies about the weight it puts on the fused solution, and whether that weight
matches the Bayes-optimal one.
"""

import numpy as np
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------- #
# implied weight and implied evidence
# --------------------------------------------------------------------------- #
def fusion_weight(estimate, segregated, fused, min_separation=1.0):
    """Solve  estimate = w*fused + (1-w)*segregated  for w.

    Optimal model averaging predicts w == p(C=1|x), so comparing w to the
    analytical p(C=1) tests whether the network averages optimally.

    When the two references nearly coincide the ratio is meaningless -- the
    denominator goes to zero and readout noise is amplified without bound, so a
    handful of trials otherwise dominate every summary. Trials with
    |fused - segregated| < min_separation (deg) return NaN. Raise it if the
    curve is still noisy; lower it to keep more low-disparity trials.
    """
    denom = fused - segregated
    w = np.full(np.shape(estimate), np.nan)
    ok = np.abs(denom) > min_separation
    w[ok] = (estimate[ok] - segregated[ok]) / denom[ok]
    return w


def implied_log_bf(p, p_common):
    """Invert p(C=1|x) back to a log Bayes factor: logit(p) - logit(prior)."""
    prior = np.clip(p_common, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p) - (np.log(prior) - np.log1p(-prior))


def compare(reference, estimate):
    """Slope / R^2 / RMSE of estimate against reference, ignoring NaNs.

    Per-trial R^2 on the implied weight is often poor even when the binned curve
    tracks the optimum closely -- single-trial w is a noisy ratio. Read the slope
    and the curve together.
    """
    ok = np.isfinite(reference) & np.isfinite(estimate)
    a, b = np.asarray(reference)[ok], np.asarray(estimate)[ok]
    return {"slope": float(np.polyfit(a, b, 1)[0]),
            "r2": float(r2_score(a, b)),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "n": int(ok.sum())}


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #
def mean_by_bin(x, y, grid):
    """Mean of y grouped by nearest bin centre in grid -> (centres, means, counts).

    Hold the reliabilities roughly fixed before calling (mask the trials), or the
    curve mixes reliability levels together.
    """
    grid = np.asarray(grid, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    idx = np.abs(x[:, None] - grid[None, :]).argmin(1)
    centres, means, counts = [], [], []
    for b in range(len(grid)):
        m = idx == b
        if m.any():
            centres.append(grid[b])
            means.append(y[m].mean())
            counts.append(int(m.sum()))
    return np.array(centres), np.array(means), np.array(counts)


def transition_fit(abs_disparity, weight):
    """Fit w = 1/(1 + exp(k*(|d| - d0))) -> (midpoint d0, sharpness k).

    d0 is the disparity at which fusion gives way to segregation, k how abruptly.
    Both NaN if the fit does not converge.
    """
    from scipy.optimize import curve_fit

    ok = np.isfinite(abs_disparity) & np.isfinite(weight)
    d, w = np.asarray(abs_disparity)[ok], np.asarray(weight)[ok]
    if d.size < 4 or np.ptp(d) == 0:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(lambda dd, d0, k: 1 / (1 + np.exp(k * (dd - d0))),
                            d, w, p0=[np.median(d), 1.0], maxfev=10000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan
    d0, k = float(popt[0]), float(popt[1])
    # curve_fit can "converge" on a midpoint far outside the data when the
    # weight has no usable transition -- an undertrained network, or a prior so
    # extreme that every trial sits on one side. Such a fit is not a small
    # error, it is a meaningless number (values in the thousands of degrees
    # have been observed), and silently averaging it would destroy any summary
    # it entered. Reject anything outside the observed range.
    lo, hi = float(d.min()), float(d.max())
    span = hi - lo
    if not np.isfinite(d0) or not np.isfinite(k) or k <= 0 \
            or d0 < lo - 0.25 * span or d0 > hi + 0.25 * span:
        return np.nan, np.nan
    return d0, k


def by_reliability(x, y, reliability, levels, grid):
    """Run mean_by_bin separately per reliability level (nearest match).

    The prediction: less reliable cues tolerate more disparity before
    segregating, so the transition midpoint should move outward.
    """
    levels = np.asarray(levels, float)
    nearest = np.abs(np.asarray(reliability)[:, None] - levels[None, :]).argmin(1)
    return {float(lv): mean_by_bin(x[nearest == b], y[nearest == b], grid)
            for b, lv in enumerate(levels) if (nearest == b).any()}


# --------------------------------------------------------------------------- #
# decision strategy
# --------------------------------------------------------------------------- #
def strategy_fit(estimate, post, fused, segregated, seed=0):
    """RMSE of three candidate strategies against an observed estimate.

    `post` is the trial-wise posterior p(C=1|x) (d["post_c1"]), NOT the prior.

        averaging   p*fused + (1-p)*segregated      (the Bayes-optimal one)
        selection   fused if p > 0.5 else segregated
        matching    fused with probability p, else segregated

    Lowest wins. Run it on the population-decoded estimate as well as the
    readout, to ask what the representation is doing rather than the output layer.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(estimate)

    def rmse(prediction):
        return float(np.sqrt(np.mean((prediction[ok] - estimate[ok]) ** 2)))

    scores = {
        "averaging": rmse(post * fused + (1 - post) * segregated),
        "selection": rmse(np.where(post > 0.5, fused, segregated)),
        "matching": rmse(np.where(rng.random(post.shape) < post,
                                  fused, segregated)),
    }
    scores["best"] = min(scores, key=scores.get)
    return scores


# --------------------------------------------------------------------------- #
# SS7.1 -- headline statistics
# --------------------------------------------------------------------------- #
def position_regression(estimate, segregated, fused, w_optimal):
    """The headline statistic (SS7.1): regress (estimate - seg) on w_opt * Delta.

    No division, so every trial enters with its natural leverage and the
    small-|Delta| trials -- where behaviour cannot reveal the weight -- carry
    almost none. Bayes-optimal model averaging predicts slope = 1, intercept = 0.
    Returns slope/intercept with standard errors and 95% CIs, plus r2 and n.
    """
    x = np.asarray(w_optimal) * (np.asarray(fused) - np.asarray(segregated))
    y = np.asarray(estimate) - np.asarray(segregated)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    X = np.column_stack([x, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "slope": float(beta[0]), "slope_se": float(se[0]),
        "slope_ci95": [float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
        "intercept": float(beta[1]), "intercept_se": float(se[1]),
        "intercept_ci95": [float(beta[1] - 1.96 * se[1]),
                           float(beta[1] + 1.96 * se[1])],
        "r2": float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(n),
    }


def sigma_out(pred, target):
    """Per-output readout noise: std of (network - analytical) residuals.

    Estimate it on the p_common = 1 CONTROL network (SS9.1), where the target
    is single-valued and residuals are pure output noise -- on the flagship the
    residuals also carry any causal-inference misweighting.
    """
    resid = np.asarray(pred) - np.asarray(target)
    return resid.std(axis=0)


def sigma_w(sig_out, delta):
    """Per-trial uncertainty of the implied weight: sigma_w ~ sigma_out / |Delta|.

    Filter scatter plots by a stated criterion (design suggests sigma_w < 0.1);
    small |Delta| makes w unreadable -- expected and meaningful, not a failure.
    """
    delta = np.abs(np.asarray(delta, float))
    out = np.full(delta.shape, np.inf)
    ok = delta > 0
    out[ok] = sig_out / delta[ok]
    return out


def joint_fusion_weight(est_vis, seg_vis, est_prop, seg_prop, fused):
    """One w per trial from BOTH outputs jointly (SS7.1).

    Each trial gives two equations est_k = w*fused + (1-w)*seg_k with a common
    w; the least-squares solution pools them with their natural leverage:
        w = sum_k Delta_k (est_k - seg_k) / sum_k Delta_k^2 .
    """
    dv = np.asarray(fused) - np.asarray(seg_vis)
    dp = np.asarray(fused) - np.asarray(seg_prop)
    num = dv * (np.asarray(est_vis) - np.asarray(seg_vis)) \
        + dp * (np.asarray(est_prop) - np.asarray(seg_prop))
    den = dv ** 2 + dp ** 2
    w = np.full(den.shape, np.nan)
    ok = den > 0
    w[ok] = num[ok] / den[ok]
    return w


def weight_consistency(w_vis, w_prop, sw_vis, sw_prop, criterion=0.1):
    """Hand-vs-visual agreement of the implied weight on well-conditioned trials.

    Internal-coherence test of model averaging (SS7.1): both outputs are
    mixtures with the SAME w, so on trials where both are readable
    (sigma_w < criterion on both) the two readings must agree.
    """
    ok = (np.asarray(sw_vis) < criterion) & (np.asarray(sw_prop) < criterion) \
        & np.isfinite(w_vis) & np.isfinite(w_prop)
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "corr": np.nan, "mean_abs_diff": np.nan,
                "criterion": criterion}
    a, b = np.asarray(w_vis)[ok], np.asarray(w_prop)[ok]
    return {"n": int(ok.sum()),
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "criterion": criterion}


# --------------------------------------------------------------------------- #
# SS7.2 -- Bayes vs disparity heuristic
# --------------------------------------------------------------------------- #
def reliability_within_disparity(w_implied, w_optimal, abs_disparity,
                                 n_bins=8, min_per_bin=30, min_spread=0.05):
    """At MATCHED disparity, does the implied weight track the reliability-driven
    variation of the optimal weight? (SS7.2)

    A pure disparity heuristic predicts within-bin slope 0 everywhere; Bayes
    predicts slope ~ 1. Bins |disparity| into quantile bins; within each,
    regresses w_implied on w_optimal. Returns per-bin slopes and a combined
    inverse-variance-weighted slope with its standard error.

    Bins whose w_optimal spans less than min_spread are DROPPED, not merely
    downweighted: with almost no variation in the predictor the slope is
    0/0-ish and can come back in the tens with an equally large standard error.
    Inverse-variance weighting mostly discounts such a bin, but it still leaks
    into the combined estimate, and it makes the per-bin table unreadable. The
    dropped bins are reported under "skipped" so the exclusion stays visible.
    """
    ok = np.isfinite(w_implied) & np.isfinite(w_optimal) & np.isfinite(abs_disparity)
    wi, wo, ad = (np.asarray(a)[ok] for a in (w_implied, w_optimal, abs_disparity))
    edges = np.quantile(ad, np.linspace(0, 1, n_bins + 1))
    rows, skipped = [], []
    for b in range(n_bins):
        m = (ad >= edges[b]) & (ad <= edges[b + 1] if b == n_bins - 1
                                else ad < edges[b + 1])
        if m.sum() < min_per_bin:
            skipped.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                            "n": int(m.sum()), "reason": "too few trials"})
            continue
        if np.ptp(wo[m]) < min_spread:
            skipped.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                            "n": int(m.sum()), "w_opt_spread": float(np.ptp(wo[m])),
                            "reason": "no reliability-driven variation to test"})
            continue
        x, y = wo[m], wi[m]
        X = np.column_stack([x, np.ones(len(x))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = float(resid @ resid) / max(len(x) - 2, 1)
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]))
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "n": int(m.sum()), "slope": float(beta[0]), "slope_se": se,
                     "w_opt_spread": float(np.ptp(wo[m]))})
    if not rows:
        return {"bins": [], "skipped": skipped,
                "combined_slope": np.nan, "combined_se": np.nan}
    w_inv = np.array([1 / max(r["slope_se"], 1e-9) ** 2 for r in rows])
    slopes = np.array([r["slope"] for r in rows])
    comb = float((w_inv * slopes).sum() / w_inv.sum())
    comb_se = float(np.sqrt(1 / w_inv.sum()))
    return {"bins": rows, "skipped": skipped,
            "combined_slope": comb, "combined_se": comb_se,
            "combined_ci95": [comb - 1.96 * comb_se, comb + 1.96 * comb_se]}


# --------------------------------------------------------------------------- #
# SS7.3 -- the mixture-variance signature of causal ambiguity
# --------------------------------------------------------------------------- #
def variance_signature(var_net, post, fused_mu, fused_var, seg_mu, seg_var,
                       n_bins=10):
    """Network variance output binned by the analytical posterior (SS7.3).

    The full mixture variance carries a between-component term
    w(1-w)(fused_mu - seg_mu)^2 that peaks at intermediate ambiguity -- the
    hump no fixed-weight model can produce. Returns, per posterior bin: the
    network's mean Var output, the analytical mixture variance, its
    between-component term alone, and the best FIXED-weight prediction
    (w_bar * fused_var + (1-w_bar) * seg_var with w_bar = mean posterior),
    which by construction has no hump.
    """
    post, var_net = np.asarray(post), np.asarray(var_net)
    fused_mu, fused_var = np.asarray(fused_mu), np.asarray(fused_var)
    seg_mu, seg_var = np.asarray(seg_mu), np.asarray(seg_var)

    mix_var = post * (fused_var + fused_mu ** 2) \
        + (1 - post) * (seg_var + seg_mu ** 2) \
        - (post * fused_mu + (1 - post) * seg_mu) ** 2
    between = post * (1 - post) * (fused_mu - seg_mu) ** 2
    w_bar = float(post.mean())
    fixed = w_bar * fused_var + (1 - w_bar) * seg_var

    edges = np.linspace(0, 1, n_bins + 1)
    centres, rows = [], {"net": [], "mixture": [], "between": [], "fixed": [],
                         "count": []}
    for b in range(n_bins):
        m = (post >= edges[b]) & (post <= edges[b + 1] if b == n_bins - 1
                                  else post < edges[b + 1])
        if not m.any():
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        rows["net"].append(float(var_net[m].mean()))
        rows["mixture"].append(float(mix_var[m].mean()))
        rows["between"].append(float(between[m].mean()))
        rows["fixed"].append(float(fixed[m].mean()))
        rows["count"].append(int(m.sum()))
    out = {"centres": np.array(centres)}
    out.update({k: np.array(v) for k, v in rows.items()})
    # the scalar summary: is the network's variance elevated at intermediate
    # ambiguity relative to its own confident-zone level?
    c = out["centres"]
    mid = (c > 0.2) & (c < 0.8)
    conf = ~mid
    if mid.any() and conf.any():
        out["hump_net"] = float(out["net"][mid].mean() - out["net"][conf].mean())
        out["hump_analytical"] = float(out["mixture"][mid].mean()
                                       - out["mixture"][conf].mean())
    return out


# --------------------------------------------------------------------------- #
# SS7.4 -- binned comparison against the five candidate observers
# --------------------------------------------------------------------------- #
def model_comparison(estimate, post, fused, segregated, n_bins=10, seed=0):
    """Network output against five candidate strategies, binned by posterior
    decile (SS7.4, mirroring Kording Table 1):

        averaging    p*fused + (1-p)*seg           (the target)
        integration  fused always                  (p_common = 1)
        segregation  seg always                    (p_common = 0)
        selection    fused if p > 0.5 else seg     (model selection)
        fixed        w**fused + (1-w*)seg, w* the single best fixed weight
        matching     fused with probability p      (bonus comparator)

    Averaging predicts smooth sigmoidal bias curves across the deciles;
    selection predicts a step at p = 0.5 -- the discrimination lives in the
    intermediate bins. Returns overall and per-bin RMSEs plus the per-bin mean
    of (estimate - seg) for the network and each strategy (the bias curves).
    """
    rng = np.random.default_rng(seed)
    estimate, post = np.asarray(estimate), np.asarray(post)
    fused, seg = np.asarray(fused), np.asarray(segregated)
    ok = np.isfinite(estimate)
    delta = fused - seg
    denom = float((delta[ok] ** 2).sum())
    w_star = float((delta[ok] * (estimate - seg)[ok]).sum() / denom) \
        if denom > 0 else 0.5

    preds = {
        "averaging": post * fused + (1 - post) * seg,
        "integration": fused,
        "segregation": seg,
        "selection": np.where(post > 0.5, fused, seg),
        "fixed": w_star * fused + (1 - w_star) * seg,
        "matching": np.where(rng.random(post.shape) < post, fused, seg),
    }

    def rmse(a, m):
        return float(np.sqrt(np.mean((a[m] - estimate[m]) ** 2)))

    def bin_weight(values, m):
        """Least-squares weight within a bin: the slope of (values - seg) on
        Delta. This is the per-bin implied weight WITHOUT per-trial division,
        so it survives the sign of the disparity -- averaging the raw signed
        bias would cancel, since Delta is symmetric about zero within a bin.
        """
        den = float((delta[m] ** 2).sum())
        if den <= 0:
            return np.nan
        return float((delta[m] * (values - seg)[m]).sum() / den)

    edges = np.quantile(post, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, post, side="right") - 1, 0, n_bins - 1)
    out = {"fixed_w": w_star,
           "overall": {k: rmse(v, ok) for k, v in preds.items()},
           "bin_centres": [], "bin_rmse": {k: [] for k in preds},
           "bin_weight_net": [], "bin_weight": {k: [] for k in preds},
           "bias_net": [], "bias": {k: [] for k in preds}}
    for b in range(n_bins):
        m = ok & (bins == b)
        if not m.any():
            continue
        out["bin_centres"].append(float(post[m].mean()))
        out["bias_net"].append(float((estimate - seg)[m].mean()))
        out["bin_weight_net"].append(bin_weight(estimate, m))
        for k, v in preds.items():
            out["bin_rmse"][k].append(rmse(v, m))
            out["bias"][k].append(float((v - seg)[m].mean()))
            out["bin_weight"][k].append(bin_weight(v, m))
    # "fixed" nests integration (w*=1), segregation (w*=0) and every other
    # constant weight, so in-sample it can never lose to them -- prefer a
    # parameter-free strategy whenever it comes within 0.5% of the fitted one
    # (an implicit complexity penalty for the free parameter).
    ranked = sorted(out["overall"], key=out["overall"].get)
    best = ranked[0]
    if best == "fixed":
        tol = 1.005 * out["overall"]["fixed"]
        for k in ranked[1:]:
            if k != "matching" and out["overall"][k] <= tol:
                best = k
                break
    out["best"] = best
    return out


def binned_implied_weight(estimate, segregated, fused, x, grid,
                          min_count=25, sigma_out=None, max_se=0.1):
    """Implied weight per bin of `x`, as a least-squares slope (no division).

    Within each bin, regress (estimate - seg) on Delta = fused - seg through
    the origin:  w_bin = sum(Delta * (estimate - seg)) / sum(Delta^2),
    whose standard error is  sigma_out / sqrt(sum(Delta^2)).

    This is the unbiased way to draw a weight-vs-disparity curve. The per-trial
    ratio (estimate - seg)/Delta needs a sigma_w filter to stay finite, and that
    filter keeps preferentially high-|Delta| trials -- which within a disparity
    bin are the ones with the most separated hypotheses, and therefore the
    lowest weights. Binning the filtered ratio pulls the curve systematically
    below the analytical posterior even for a perfectly Bayesian network. The
    least-squares form uses every trial and has no such selection.

    Bins are dropped when they hold fewer than `min_count` trials, or (given
    `sigma_out`) when the slope's standard error exceeds `max_se`. The second
    guard is the bin-level analogue of the per-trial sigma_w criterion: near
    zero disparity the two hypotheses coincide, sum(Delta^2) collapses, and the
    weight is simply not identifiable there however many trials the bin holds.
    Without it the curve spikes at the origin -- an artefact of the estimator,
    not a property of the network.

    Returns (centres, weights, counts, standard_errors).
    """
    estimate, segregated = np.asarray(estimate), np.asarray(segregated)
    fused, x = np.asarray(fused), np.asarray(x)
    grid = np.asarray(grid, float)
    delta = fused - segregated
    resid = estimate - segregated
    ok = np.isfinite(delta) & np.isfinite(resid) & np.isfinite(x)
    idx = np.abs(x[:, None] - grid[None, :]).argmin(1)

    centres, weights, counts, ses = [], [], [], []
    for b in range(len(grid)):
        m = ok & (idx == b)
        den = float((delta[m] ** 2).sum())
        if m.sum() < min_count or den <= 0:
            continue
        w = float((delta[m] * resid[m]).sum() / den)
        if sigma_out is None:
            se = float(np.sqrt(((resid[m] - w * delta[m]) ** 2).sum()
                               / max(int(m.sum()) - 1, 1) / den))
        else:
            se = float(sigma_out / np.sqrt(den))
        if not np.isfinite(se) or se > max_se:
            continue
        centres.append(float(grid[b]))
        weights.append(w)
        counts.append(int(m.sum()))
        ses.append(se)
    return (np.array(centres), np.array(weights),
            np.array(counts), np.array(ses))
