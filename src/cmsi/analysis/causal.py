"""Is the network doing causal inference, and is it doing it optimally?

The network is never shown p(C=1). These functions ask what its behaviour
implies about the weight it puts on the fused solution, and whether that weight
matches the Bayes-optimal one.

The weight is read per channel by the HYBRID read (hybrid_weight): the
mixture-variance root where it is unique (Delta^2 <= c, which includes zero
disparity) and the channel's own position ratio where the variance has two
roots (Delta^2 > c). It exists on every trial and nothing is filtered.
"""

import numpy as np
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------- #
# implied evidence and a generic comparison
# --------------------------------------------------------------------------- #
def implied_log_bf(p, p_common):
    """Invert p(C=1|x) back to a log Bayes factor: logit(p) - logit(prior)."""
    prior = np.clip(p_common, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p) - (np.log(prior) - np.log1p(-prior))


def compare(reference, estimate):
    """Slope / R^2 / RMSE of estimate against reference, ignoring NaNs.

    Read the slope and the binned curve together: per-trial R^2 also carries
    the read-out noise of every single trial.
    """
    ok = np.isfinite(reference) & np.isfinite(estimate)
    a, b = np.asarray(reference)[ok], np.asarray(estimate)[ok]
    if ok.sum() < 2:
        return {"slope": np.nan, "r2": np.nan, "rmse": np.nan, "n": int(ok.sum())}
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


def binned_weight(x, w, grid, min_count=25, max_se=None):
    """Mean and standard error of a per-trial weight per bin of `x` (nearest
    grid centre) -> (centres, means, counts, standard_errors).

    The weight-vs-disparity curve of the hybrid read: every trial enters, the
    error bar is the sd of the trials in the bin over sqrt(n). Bins holding
    fewer than `min_count` trials are dropped, and so are bins whose standard
    error exceeds `max_se` when one is given, so that a sparse or noisy bin
    reads as a gap rather than as a point.
    """
    x, w, grid = np.asarray(x, float), np.asarray(w, float), np.asarray(grid, float)
    ok = np.isfinite(x) & np.isfinite(w)
    idx = np.abs(x[:, None] - grid[None, :]).argmin(1)
    centres, means, counts, ses = [], [], [], []
    for b in range(len(grid)):
        m = ok & (idx == b)
        n = int(m.sum())
        if n < max(min_count, 2):
            continue
        se = float(w[m].std(ddof=1) / np.sqrt(n))
        if max_se is not None and (not np.isfinite(se) or se > max_se):
            continue
        centres.append(float(grid[b]))
        means.append(float(w[m].mean()))
        counts.append(n)
        ses.append(se)
    return np.array(centres), np.array(means), np.array(counts), np.array(ses)


def weight_by_posterior(w, post, n_bins=10, min_count=25):
    """A per-trial weight per equal-width bin of the analytical posterior:
    the bin's mean posterior, the mean weight, its standard error and the
    count, for the bins holding at least `min_count` trials. A Bayes-optimal
    model-averaging observer puts every point on the identity line."""
    w, post = np.asarray(w, float), np.asarray(post, float)
    ok = np.isfinite(w) & np.isfinite(post)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {k: [] for k in ("centres", "w", "se", "n")}
    for b in range(n_bins):
        hi = post <= edges[b + 1] if b == n_bins - 1 else post < edges[b + 1]
        m = ok & (post >= edges[b]) & hi
        n = int(m.sum())
        if n < max(min_count, 2):
            continue
        out["centres"].append(float(post[m].mean()))
        out["w"].append(float(w[m].mean()))
        out["se"].append(float(w[m].std(ddof=1) / np.sqrt(n)))
        out["n"].append(n)
    return {k: np.array(v) for k, v in out.items()}


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


def mixture_variance(w, fused_var, seg_var, delta):
    """The model-averaged variance for a weight w (Eq. 10, law of total
    variance): w fused_var + (1 - w) seg_var + w (1 - w) Delta^2."""
    w = np.asarray(w, float)
    return w * fused_var + (1 - w) * seg_var + w * (1 - w) * np.asarray(delta, float) ** 2


def variance_weight(v_hat, fused_var, seg_var, delta):
    """The weight implied by a VARIANCE output, on every trial (SS7.1/7.3).

    The variance target of a model-averaging observer is the mixture
    variance v(w) = w fused_var + (1 - w) seg_var + w (1 - w) Delta^2, so a
    variance output v_hat can be solved for the weight the network used -- a
    quadratic in w,

        Delta^2 w^2 - (Delta^2 - c) w - (seg_var - v_hat) = 0,
        c = seg_var - fused_var > 0  (the variance fusion saves).

    v(w) is a parabola in w opening downward, with v(0) = seg_var, v(1) =
    fused_var and its peak at w* = 1/2 - c / (2 Delta^2). Its sensitivity
    |dv/dw| = |Delta^2 (1 - 2w) - c| does NOT vanish at Delta = 0: there it
    equals c, so this read stays sharp exactly where a position ratio
    (sensitivity |Delta|) is blind.

    Roots: with Delta^2 <= c the peak lies at or left of w = 0, v(w) is
    monotone on [0, 1] and the root is unique -- the one whose Delta -> 0
    limit is (seg_var - v_hat)/c (flag 0). With Delta^2 > c the peak lies
    inside (0, 1/2), v(w) rises before it falls and two roots can lie in
    [0, 1]; the variance output alone cannot tell them apart, so the larger
    root is returned with flag 1 and the caller decides -- the hybrid read
    (hybrid_weight) uses the channel's position ratio on those trials
    instead. When v_hat exceeds the largest mixture variance any w can
    produce, the location of that maximum is returned (flag 2).

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
    w[ok] = (b[ok] + np.sqrt(disc[ok])) / (2 * a[ok])
    flags[ok] = np.where(a[ok] > c[ok], 1, 0)
    above = ~linear & (disc < 0)
    w[above] = b[above] / (2 * a[above])
    flags[above] = 2
    sensitivity = np.abs(b - 2 * a * w)
    return w, sensitivity, flags


def hybrid_weight(v_hat, fused_var, seg_var, delta, mu_hat, seg_mu, sig_out_mu=None,
                  sig_out_var=None):
    """The HYBRID per-trial weight of one channel: the variance root where it
    is unique, the position ratio where it is not (SS7.1/7.3).

    One channel's two outputs read the same weight in two regimes of the
    mixture-variance parabola v(w) (see variance_weight):

      Delta^2 <= c  the parabola is monotone on [0, 1], the variance output
                    gives ONE root -- taken as is (flag 0; the peak position
                    if v_hat exceeds the largest mixture variance, flag 2).
                    This includes Delta -> 0, where the ratio is blind.
      Delta^2 >  c  two roots; instead of choosing between them, the
                    weight is read from the channel's POSITION output,
                    w = (mu_hat - seg_mu) / Delta, which is well conditioned
                    exactly here (|Delta| > sqrt(c)); flag 1.

    `mu_hat` and `seg_mu` are the channel's position output and segregated
    mean (the same channel as v_hat, fused_var and seg_var). With the two
    readout noises given, `sigma` is the nominal per-trial sd of the read:
    sig_out_var / |dv/dw| on the root trials, sig_out_mu / |Delta| on the
    ratio trials (else NaN). Nothing is filtered or clipped.

    Returns (w, sigma, flags) with flags 0 = variance root, 1 = position
    ratio, 2 = variance peak (no real root, Delta^2 <= c).
    """
    delta = np.asarray(delta, float)
    w_var, sens, flags_var = variance_weight(v_hat, fused_var, seg_var, delta)
    c = np.asarray(seg_var, float) - np.asarray(fused_var, float)
    ratio_regime = delta ** 2 > c
    w = w_var.copy()
    flags = flags_var.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (np.asarray(mu_hat, float) - np.asarray(seg_mu, float)) / delta
    w[ratio_regime] = ratio[ratio_regime]
    flags[ratio_regime] = 1
    sigma = np.full(w.shape, np.nan)
    if sig_out_var is not None:
        root = ~ratio_regime
        sigma[root] = sig_out_var / np.maximum(sens[root], 1e-12)
    if sig_out_mu is not None:
        sigma[ratio_regime] = sig_out_mu / np.abs(delta[ratio_regime])
    return w, sigma, flags


def variance_regression(var_out, fused_var, seg_var, delta, post):
    """The headline statistic in the VARIANCE domain (the analogue of
    position_regression, no division): the network's variance reduction
    relative to segregation, var_out - seg_var, regressed on the optimal
    mixture's, v(p) - seg_var. Bayes-optimal model averaging predicts slope
    1, intercept 0. Every trial enters with its natural leverage, which is
    largest where the between-component term p(1-p)Delta^2 is largest."""
    v_opt = mixture_variance(post, fused_var, seg_var, delta)
    return position_regression(var_out, seg_var, v_opt, np.ones_like(np.asarray(post, float)))


def weight_regression(w, post):
    """A per-trial weight regressed on the analytical posterior: slope 1,
    intercept 0 for an optimal observer. Meant for a read that exists on
    every trial (the variance read); NaNs are left out."""
    w, post = np.asarray(w, float), np.asarray(post, float)
    ok = np.isfinite(w) & np.isfinite(post)
    return position_regression(w[ok], np.zeros(ok.sum()), post[ok], np.ones(ok.sum()))


def weight_consistency(w_vis, w_prop):
    """Hand-vs-visual agreement of the implied weight, trial by trial.

    Internal-coherence test of model averaging (SS7.1): both channels are
    mixtures with the SAME w, so the two hybrid reads must agree. Every trial
    with both reads finite enters; nothing is filtered.
    """
    ok = np.isfinite(w_vis) & np.isfinite(w_prop)
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "corr": np.nan, "mean_abs_diff": np.nan}
    a, b = np.asarray(w_vis)[ok], np.asarray(w_prop)[ok]
    return {"n": int(ok.sum()),
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(a - b)))}


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
