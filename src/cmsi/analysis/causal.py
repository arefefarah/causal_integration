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
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


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
def strategy_fit(estimate, p_common, fused, segregated, seed=0):
    """RMSE of three candidate strategies against an observed estimate.

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
        "averaging": rmse(p_common * fused + (1 - p_common) * segregated),
        "selection": rmse(np.where(p_common > 0.5, fused, segregated)),
        "matching": rmse(np.where(rng.random(p_common.shape) < p_common,
                                  fused, segregated)),
    }
    scores["best"] = min(scores, key=scores.get)
    return scores
