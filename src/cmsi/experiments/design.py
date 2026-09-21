"""Readability of the implied weight at the level of the trials.

The network never outputs a weight; the per-trial weight is the ratio
(network - seg)/Delta, and on a network that is Bayes-optimal up to a position
error e it is exactly

    w = p + e / Delta        p the analytical posterior, Delta = fused - seg

so what the ratio looks like is decided by two things only: how the trials
distribute |Delta| (a property of the configuration, fixed before training)
and how large e is (a property of the trained network). This module keeps
the two apart:

    design_analysis   for a configuration, before any network is trained:
                      the posterior distribution, |Delta| by true cause, the
                      fraction of ratios that would fall outside [-1, 2] for
                      any error sd one is prepared to assume, the fraction the
                      sigma_w criterion would keep, and the floor on the error
                      set by the spike code itself (decoding_floor). Several
                      configurations go on one set of figures, so a range or
                      a prior width can be screened without training.
    runs figures      the same quantities measured on trained pipeline runs,
                      side by side, with what each run achieved
                      (scripts/06_implied_weight.py compare).
    balance_posterior a dataset with a flat posterior histogram, by an
                      acceptance rule that depends on the measurements only,
                      so the targets stay exactly right (the safe reading of
                      "balance the weight distribution"; used by the sweep
                      experiment and by `design --balance`).

Entry points: scripts/06_implied_weight.py design / compare.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.experiments.implied_weight import (
    RATIO_LIM,
    READS,
    _ramp,
    flat_posterior_weights,
    predicted_outside,
)
from cmsi.viz.style import COLORS, SIZE, label_panels

# error sds (deg) the readability curves are evaluated at: 0.05 to 5 deg
SD_GRID = np.logspace(np.log10(0.05), np.log10(5.0), 60)
SIGMA_W_CRITERION = 0.1


# --------------------------------------------------------------------------- #
# the trial-level quantities
# --------------------------------------------------------------------------- #
def readability_curves(delta, post, sds=SD_GRID, crit=SIGMA_W_CRITERION, lim=RATIO_LIM,
                       flat=None):
    """For each error sd in `sds`: the fraction of ratios predicted outside
    `lim` and the fraction of trials the sigma_w criterion keeps
    (sd/|Delta| < crit). With `flat` (flat_posterior_weights), the same
    under a flat posterior histogram."""
    delta = np.asarray(delta, float)
    ad = np.abs(delta)
    sds = np.asarray(sds, float)
    out = {"sd": sds,
           "outside": np.array([predicted_outside(delta, post, s, lim) for s in sds]),
           "readable": np.array([float(np.mean(ad > s / crit)) for s in sds])}
    if flat is not None:
        out["outside_flat"] = np.array([predicted_outside(delta, post, s, lim, weights=flat)
                                        for s in sds])
        out["readable_flat"] = np.array([float((flat * (ad > s / crit)).sum()) for s in sds])
    return out


def posterior_at_zero_disparity(cfg):
    """The analytical posterior on a trial whose two cues agree exactly, both
    at the prior mean, at the least and the most noisy end of the ranges:
    a reference for how confident about C = 1 the observer can get at all in
    this configuration. It is bounded by the prior widths relative to the
    sensory noise -- when the two independent sources of C = 2 could well be
    this close together, agreement is weak evidence -- and rises further
    from the prior mean, so this is a typical maximum, not the maximum."""
    from cmsi.data.generative import common_cause_posterior, log_bayes_factor

    gen = cfg["generative"]
    out = {}
    for tag, pick in (("low_noise", min), ("high_noise", max)):
        sv, sp, se = (pick(gen[f"sigma2_{k}_range"]) for k in ("vis", "prop", "eye"))
        prior_precision = 0.0 if np.isinf(gen["eye_sigma_sq"]) else 1.0 / gen["eye_sigma_sq"]
        var_eye = 1.0 / (1.0 / se + prior_precision)
        lbf = log_bayes_factor(np.array([gen["mu0"]]), np.array([sv + var_eye]),
                               np.array([gen["mu0"]]), np.array([sp]),
                               gen["mu0"], gen["sigma0_sq"])
        out[tag] = float(common_cause_posterior(lbf, gen["p_common"])[0])
    return out


def decoding_floor(d, cfg, n=3000, step=0.1, seed=0):
    """A lower bound on the read-out noise from the spike code alone.

    Each population is decoded by maximum likelihood on a grid of positions
    (the visual code as in the calibration's Poisson-validity check, the two
    push-pull codes the same way with their own tuning), each trial's gain
    taken as known; the observer is then run on the decoded measurements with
    the true variances, and its estimates are compared with the ones computed
    from the exact measurements. The sd of that difference is the error a
    network would make if it decoded every population as well as an ideal
    observer that knows each trial's noise level and made no other error. A
    trained network cannot do better, so its control's sigma_out sits above
    `fused_mu`, and its own residual sd on a causal run above `mu_vis` /
    `mu_prop`. How far above is what training, size and the extent of the
    input domain decide.

    Runs on `n` trials of `d` (which must carry X, the measurements, the
    variances and the encoders); `step` is the grid spacing in degrees.
    """
    from cmsi.data.encoding import group_slices
    from cmsi.data.generative import observer

    enc, gen = cfg["encoding"], cfg["generative"]
    encoders, K = d["encoders"], enc["gain_K"]
    rng = np.random.default_rng(seed)
    total = len(d["X"])
    pick = rng.choice(total, n, replace=False) if n < total else np.arange(total)
    X, sl = d["X"][pick], group_slices(enc)

    def decode(counts, rates_of, grid, gain):
        F = rates_of(grid)                                   # (n_grid, n_units)
        log_f = np.log(np.clip(F, 1e-12, None))
        # log p(counts | x = g) up to terms that do not depend on g
        ll = counts @ log_f.T - gain[:, None] * F.sum(1)[None, :]
        return grid[ll.argmax(1)]

    def push_pull(slope, intercept):
        return lambda g: np.clip(g[:, None] * slope[None, :] + intercept[None, :], 0, None)

    lo, hi = enc["visual_field"]
    grid_v = np.arange(lo, hi + step, step)
    x_vis = decode(X[:, sl["visual_hand"]],
                   lambda g: np.exp(-0.5 * ((g[:, None] - encoders["rf_centers"][None, :])
                                            / enc["rf_width"]) ** 2),
                   grid_v, K / d["sig2_vis"][pick])
    span = 4 * np.sqrt(gen["sigma0_sq"]) + 4 * np.sqrt(max(gen["sigma2_prop_range"]))
    grid_p = np.arange(gen["mu0"] - span, gen["mu0"] + span + step, step)
    x_prop = decode(X[:, sl["prop_hand"]],
                    push_pull(encoders["prop_slope"], encoders["prop_intercept"]),
                    grid_p, K / d["sig2_prop"][pick])
    eye_sd = 0.0 if np.isinf(gen["eye_sigma_sq"]) else np.sqrt(gen["eye_sigma_sq"])
    span = 4 * eye_sd + 4 * np.sqrt(max(gen["sigma2_eye_range"]))
    grid_e = np.arange(gen["eye_mu"] - span, gen["eye_mu"] + span + step, step)
    x_eye = decode(X[:, sl["prop_eye"]],
                   push_pull(encoders["eye_slope"], encoders["eye_intercept"]),
                   grid_e, K / d["sig2_eye"][pick])

    noise = {k: d[k][pick] for k in ("sig2_vis", "sig2_prop", "sig2_eye")}
    exact = observer({**noise, **{k: d[k][pick] for k in ("x_vis", "x_eye", "x_prop")}}, gen)
    decoded = observer({**noise, "x_vis": x_vis, "x_eye": x_eye, "x_prop": x_prop}, gen)
    out = {"n": int(len(pick)), "step_deg": float(step),
           "decode_sd": {k: float((v - d[k][pick]).std())
                         for k, v in (("x_vis", x_vis), ("x_prop", x_prop), ("x_eye", x_eye))}}
    for k in ("fused_mu", "mu_vis", "mu_prop"):
        out[k] = float((decoded[k] - exact[k]).std())
    return out


def balance_posterior(d, keep, seed=0, n_bins=10):
    """Subsample a dataset so that the histogram of the analytical posterior
    is as flat as keeping a fraction `keep` of the trials allows.

    Trial i is kept with probability min(1, q / share_i), where share_i is
    the share of trials in its posterior bin and q is set so that the
    expected kept fraction is `keep`: the bins that hold the most trials are
    thinned down to a common level, the sparse ones are kept whole. If
    `keep` is small enough the result is exactly flat.

    The rule sees a trial only through its posterior, a function of the
    measurements x. The acceptance a(x) then cancels in
    p(C | x) = p(C, x) a(x) / (p(x) a(x)): the observer's outputs -- the
    targets -- remain exactly right on the kept trials, and no cue to C is
    created that the observer does not already use. What changes is the
    distribution of trials the network trains on: more in the ambiguous
    zone, fewer at the confident ends. (The single-channel AUCs of the
    calibration audit are marginal statistics of a distribution that has
    changed, so they can move a little; `design --balance` reports them.)

    The acceptance draw uses a generator derived from `seed` together with a
    fixed salt, never `default_rng(seed)` itself: the sampler's first draw
    from `default_rng(seed)` is the uniform that decides C on every trial,
    so an acceptance draw from the same stream would accept trials by their
    true cause -- exactly the leak this rule is meant not to have.

    Returns (the subset dataset, info).
    """
    from cmsi.data import subset

    rng = np.random.default_rng([int(seed), 0x5EED5])
    post = np.asarray(d["post_c1"], float)
    n = post.size
    idx = np.clip((post * n_bins).astype(int), 0, n_bins - 1)
    share = np.bincount(idx, minlength=n_bins) / n
    per_trial = share[idx]

    def kept_fraction(q):
        return float(np.mean(np.minimum(1.0, q / per_trial)))

    lo, hi = 0.0, 1.0
    for _ in range(60):                       # kept_fraction is increasing in q
        mid = 0.5 * (lo + hi)
        if kept_fraction(mid) < keep:
            lo = mid
        else:
            hi = mid
    q = hi
    accept = np.minimum(1.0, q / per_trial)
    take = np.flatnonzero(rng.random(n) < accept)
    after = np.bincount(idx[take], minlength=n_bins) / max(take.size, 1)
    info = {"keep_requested": float(keep), "kept": float(take.size / n),
            "n_before": int(n), "n_after": int(take.size), "q": float(q),
            "posterior_hist_before": share.tolist(), "posterior_hist_after": after.tolist(),
            "acceptance_by_bin": np.minimum(1.0, q / np.maximum(share, 1e-300)).tolist()}
    return subset(d, take), info


def design_analysis(cfg, n=20000, seed=None, balance=None, floor=True, sds=SD_GRID):
    """Everything the trials of one configuration say about the readability
    of the implied weight, with no network involved. Returns (stats, arrays).

    `balance` (a kept fraction in (0, 1]) draws n/balance trials and thins
    them with balance_posterior, so the analysis describes the trials a
    balanced sweep variant would train on.
    """
    from cmsi.data import make_dataset

    n_draw = int(round(n / balance)) if balance else int(n)
    d = make_dataset(cfg, n=n_draw, seed=seed)
    stats = {"n": int(len(d["X"])), "p_common": float(cfg["generative"]["p_common"]),
             "hidden": list(cfg["model"]["hidden"]), "generative": dict(cfg["generative"])}
    if balance:
        from cmsi.analysis.calibration import _anticonfound

        d, info = balance_posterior(d, balance, seed=cfg["seed"] if seed is None else seed)
        info["anticonfound"] = _anticonfound(d)
        stats["balance"] = info
        stats["n"] = int(len(d["X"]))
    post, C = d["post_c1"], d["C"]
    hist = np.histogram(post, bins=10, range=(0, 1))[0] / post.size
    stats["posterior"] = {
        "mass_intermediate": float(((post > 0.2) & (post < 0.8)).mean()),
        "mass_confident": float(((post < 0.05) | (post > 0.95)).mean()),
        "mass_gt_0.95": float((post > 0.95).mean()), "mass_lt_0.05": float((post < 0.05).mean()),
        "hist": hist.tolist(), "at_zero_disparity": posterior_at_zero_disparity(cfg)}
    stats["reads"], arrays = {}, {"post_c1": post, "C": C}
    flat = flat_posterior_weights(post)
    c1 = C == 1
    for key, _, seg_name in READS:
        delta = d["fused_mu"] - d[seg_name]
        ad = np.abs(delta)
        stats["reads"][key] = {
            "delta_sd_c1": float(delta[c1].std()) if c1.any() else np.nan,
            "delta_sd_c2": float(delta[~c1].std()) if (~c1).any() else np.nan,
            "median_absdelta": float(np.median(ad)),
            "median_absdelta_c1": float(np.median(ad[c1])) if c1.any() else np.nan,
            "frac_absdelta_lt1": float(np.mean(ad < 1)),
            "frac_absdelta_lt1_if_flat_posterior": float((flat * (ad < 1)).sum()),
        }
        arrays[key] = {"delta": delta, **readability_curves(delta, post, sds, flat=flat)}
    if floor:
        stats["floor"] = decoding_floor(d, cfg)
    return stats, arrays


# --------------------------------------------------------------------------- #
# figures: configurations (design)
# --------------------------------------------------------------------------- #
def _read_color(key):
    return COLORS["visual"] if key == "vis" else COLORS["prop"]


def fig_design_posterior(rows):
    """Histogram of the analytical posterior per configuration, with the
    masses the calibration gate looks at. This is the distribution of the
    optimal weight the trials offer; the ratio can only read it where
    |Delta| is large, which is not where it is high."""
    colors = _ramp(len(rows), "plasma")
    edges = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for r, color in zip(rows, colors, strict=False):
        p = r["stats"]["posterior"]
        ax.hist(r["arrays"]["post_c1"], bins=edges, density=True, histtype="step", lw=1.5,
                color=color, label=f"{r['label']}: {100 * p['mass_intermediate']:.0f}% "
                                   f"intermediate, {100 * p['mass_confident']:.0f}% confident")
    ax.set(xlabel="analytical posterior p(C=1|x)", ylabel="density", xlim=(0, 1),
           title="the optimal weight the trials offer")
    ax.legend(fontsize=7, frameon=False, title="configuration", title_fontsize=8)
    fig.tight_layout()
    return fig


def _cdf(ax, values, color, label, ls="-"):
    v = np.sort(values[np.isfinite(values)])
    ax.plot(v, np.arange(1, v.size + 1) / v.size, ls, color=color, lw=1.5, label=label)


def fig_design_delta(rows):
    """Cumulative distribution of |Delta| = |fused - seg| per configuration,
    for each read. The ratio divides by this number: the share below 1 deg is
    the share of trials no network can be read on."""
    colors = _ramp(len(rows), "plasma")
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        for r, color in zip(rows, colors, strict=False):
            s = r["stats"]["reads"][key]
            _cdf(ax, np.abs(r["arrays"][key]["delta"]), color,
                 f"{r['label']}: {100 * s['frac_absdelta_lt1']:.0f}% < 1 deg, "
                 f"median {s['median_absdelta']:.1f}")
        ax.axvline(1.0, color="0.3", ls=":", lw=1.0)
        ax.set(xscale="log", xlabel=f"|Delta| (deg), read from {out_name}", ylim=(0, 1),
               title=f"how far apart the hypotheses are, {out_name}")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("fraction of trials")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_design_readability(rows, marks=None, crit=SIGMA_W_CRITERION, lim=RATIO_LIM):
    """Top: the fraction of per-trial ratios expected outside [-1, 2] against
    the read-out error sd, per configuration (solid: the trials as drawn;
    dashed: re-weighted to a flat posterior histogram). Bottom: the fraction
    of trials the sigma_w criterion would keep. Dotted verticals: each
    configuration's decoding floor -- no network's error can be below it.
    Grey verticals: `marks`, the errors trained runs actually achieved
    ({run: {"own_sd": ..., "sigma_out": ...}}), to read the curves at."""
    colors = _ramp(len(rows), "plasma")
    fig, axes = plt.subplots(2, 2, figsize=SIZE["quad"], sharex=True)
    for col, (key, out_name, _) in enumerate(READS):
        top, bot = axes[0, col], axes[1, col]
        for r, color in zip(rows, colors, strict=False):
            a = r["arrays"][key]
            top.plot(a["sd"], 100 * a["outside"], "-", color=color, lw=1.5, label=r["label"])
            bot.plot(a["sd"], 100 * a["readable"], "-", color=color, lw=1.5, label=r["label"])
            if "outside_flat" in a:
                top.plot(a["sd"], 100 * a["outside_flat"], "--", color=color, lw=1.1)
                bot.plot(a["sd"], 100 * a["readable_flat"], "--", color=color, lw=1.1)
            fl = r["stats"].get("floor")
            if fl:
                for ax in (top, bot):
                    ax.axvline(fl["fused_mu"], color=color, ls=":", lw=1.0)
        for name, m in (marks or {}).items():
            for ax in (top, bot):
                ax.axvline(m["own_sd"], color="0.45", lw=0.9)
                ax.axvline(m["sigma_out"], color="0.45", lw=0.9, ls="--")
            top.text(m["own_sd"], top.get_ylim()[1] if top.get_ylim()[1] > 0 else 1, f" {name}",
                     fontsize=6, color="0.35", rotation=90, va="top", ha="left")
        top.plot([], [], "--", color="0.3", lw=1.1, label="same, flat posterior")
        top.plot([], [], ":", color="0.3", lw=1.0, label="decoding floor")
        if marks:
            top.plot([], [], "-", color="0.45", lw=0.9, label="trained run: own sd")
            top.plot([], [], "--", color="0.45", lw=0.9, label="trained run: sigma_out")
        top.set(xscale="log", ylabel=f"ratios outside [{lim[0]:g}, {lim[1]:g}] (%)",
                title=f"read from {out_name}")
        bot.set(xscale="log", xlabel="read-out error sd (deg)",
                ylabel=f"trials with sd/|Delta| < {crit:g} (%)", ylim=(0, 100))
        top.legend(fontsize=6.5, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def design_figures(rows, marks=None):
    return {"01_posterior_by_config": fig_design_posterior(rows),
            "02_delta_by_config": fig_design_delta(rows),
            "03_readability_by_config": fig_design_readability(rows, marks)}


def design_table(rows):
    out = []
    for r in rows:
        s = r["stats"]
        entry = {"label": r["label"], "n": s["n"], "hidden": s["hidden"],
                 "posterior": {k: s["posterior"][k] for k in
                               ("mass_intermediate", "mass_confident", "at_zero_disparity")},
                 "floor": s.get("floor"), "balance": s.get("balance"), "reads": s["reads"]}
        for key in ("vis", "prop"):
            a = r["arrays"][key]
            entry["reads"][key]["outside_at_sd"] = {
                f"{sd:g}": float(np.interp(sd, a["sd"], a["outside"]))
                for sd in (0.25, 0.5, 1.0, 2.0)}
        out.append(entry)
    return out


def print_design(rows):
    width = max(12, max(len(r["label"]) for r in rows))
    print(f"\n{'configuration':>{width}} {'inter':>5} {'conf':>5} {'p@d=0':>11} | "
          f"{'sd(D|C1)v':>9} {'|D|<1 v':>7} {'|D|<1 p':>7} | {'floor':>5} | "
          f"{'out% @sd 0.25/0.5/1/2 (vis)':>27}")
    for r in rows:
        s, a = r["stats"], r["arrays"]["vis"]
        p = s["posterior"]
        z = p["at_zero_disparity"]
        fl = s.get("floor", {}).get("fused_mu", np.nan)
        outs = "/".join(f"{100 * np.interp(sd, a['sd'], a['outside']):.0f}"
                        for sd in (0.25, 0.5, 1.0, 2.0))
        print(f"{r['label']:>{width}} {100 * p['mass_intermediate']:5.0f} "
              f"{100 * p['mass_confident']:5.0f} {z['low_noise']:.2f}-{z['high_noise']:.2f}    | "
              f"{s['reads']['vis']['delta_sd_c1']:9.2f} "
              f"{100 * s['reads']['vis']['frac_absdelta_lt1']:6.0f}% "
              f"{100 * s['reads']['prop']['frac_absdelta_lt1']:6.0f}% | {fl:5.2f} | {outs:>27}")
    print("  inter/conf: posterior mass at 0.2-0.8 / beyond 0.05-0.95; p@d=0: posterior at zero "
          "disparity (least-most noisy trial);\n  sd(D|C1): sd of Delta_vis on C=1 trials; "
          "|D|<1: trials with |Delta| < 1 deg; floor: decoding floor on sigma_out (deg);\n"
          "  out%: ratios expected outside [-1, 2] if the read-out error sd were 0.25 / 0.5 / "
          "1 / 2 deg")


# --------------------------------------------------------------------------- #
# figures: trained runs side by side (compare)
# --------------------------------------------------------------------------- #
def fig_runs_readability(rows, lim=RATIO_LIM):
    """Per run and read: the observed fraction of ratios outside [-1, 2]
    beside what a Gaussian error of the run's own sd would give, what the
    control's sigma_out alone would give (an optimal network with this
    read-out noise), and the observed fraction under a flat posterior
    histogram (same trials, re-weighted)."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    keys = (("outside", "observed", 1.0), ("outside_predicted_own_sd", "Gaussian, own sd", 0.55),
            ("outside_predicted_sigma_out", "Gaussian, sigma_out", 0.3),
            ("outside_if_flat_posterior", "observed, flat posterior", 0.8))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    wdt = 0.2
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        col = _read_color(key)
        for j, (k, lab, alpha) in enumerate(keys):
            vals = [100 * r["metrics"]["reads"][key]["readability"][k] for r in rows]
            ax.bar(x + (j - 1.5) * wdt, vals, wdt, color=col, alpha=alpha, label=lab,
                   hatch="//" if k.endswith("flat_posterior") else None)
        ax.set(xticks=x, xticklabels=labels, title=f"read from {out_name}")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel(f"ratios outside [{lim[0]:g}, {lim[1]:g}] (%)")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_runs_noise(rows):
    """Left: the read-out noise per run -- the control's sigma_out, the run's
    own residual sd (which adds any misweighting), and the decoding floor.
    Right: the position-regression slope with its 95% CI, the one number that
    says whether the pull toward fusion is Bayes-optimal (1)."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    ax, wdt = axes[0], 0.18
    for j, (key, out_name, _) in enumerate(READS):
        col = _read_color(key)
        so = [r["metrics"]["sigma_out"][out_name] for r in rows]
        own = [r["metrics"]["residual_std_self"][out_name] for r in rows]
        ax.bar(x + (j - 0.5) * 2 * wdt - wdt / 2, so, wdt, color=col,
               label=f"sigma_out, {out_name}")
        ax.bar(x + (j - 0.5) * 2 * wdt + wdt / 2, own, wdt, color=col, alpha=0.4,
               label=f"own residual sd, {out_name}")
    floors = [r.get("floor", {}).get("fused_mu", np.nan) for r in rows]
    if np.isfinite(floors).any():
        ax.plot(x, floors, "_", color="0.15", ms=14, mew=1.5, label="decoding floor")
    ax.set(xticks=x, xticklabels=labels, ylabel="deg", title="read-out noise")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.axhline(1, color=COLORS["analytical"], ls="--", lw=1.1, label="Bayes-optimal")
    for j, (key, out_name, _) in enumerate(READS):
        pr = [r["metrics"]["reads"][key]["position_regression"] for r in rows]
        s = np.array([q["slope"] for q in pr])
        lo = np.array([q["slope_ci95"][0] for q in pr])
        hi = np.array([q["slope_ci95"][1] for q in pr])
        ax.errorbar(x + (j - 0.5) * 0.12, s, yerr=[s - lo, hi - s], fmt="o", ms=4, capsize=3,
                    color=_read_color(key), label=out_name)
    ax.set(xticks=x, xticklabels=labels, ylabel="position-regression slope",
           title="pull toward fusion")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7, frameon=False)
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_runs_delta_over_error(rows, crit=SIGMA_W_CRITERION):
    """Cumulative distribution of |Delta| / own residual sd per run: the
    dimensionless quantity the ratio depends on. Left of 1 the ratio's
    noise exceeds the whole [0, 1] range; right of 1/criterion the sigma_w
    filter would keep the trial."""
    colors = _ramp(len(rows), "plasma")
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        for r, color in zip(rows, colors, strict=False):
            sd = r["metrics"]["reads"][key]["readability"]["own_sd"]
            ratio = np.abs(r["arrays"][key]["delta"]) / sd
            _cdf(ax, ratio, color, f"{r['label']}: median {np.median(ratio):.1f}")
        ax.axvline(1.0, color="0.3", ls=":", lw=1.0)
        ax.axvline(1 / crit, color="0.3", ls="--", lw=1.0)
        ax.set(xscale="log", xlabel=f"|Delta| / own residual sd, {out_name}", ylim=(0, 1),
               title=f"readability, {out_name}")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("fraction of trials")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_runs_posterior(rows):
    colors = _ramp(len(rows), "plasma")
    edges = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for r, color in zip(rows, colors, strict=False):
        post = r["arrays"]["post_c1"]
        inter = np.mean((post > 0.2) & (post < 0.8))
        ax.hist(post, bins=edges, density=True, histtype="step", lw=1.5, color=color,
                label=f"{r['label']}: {100 * inter:.0f}% intermediate")
    ax.set(xlabel="analytical posterior p(C=1|x)", ylabel="density", xlim=(0, 1),
           title="the optimal weight on each run's test split")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def runs_figures(rows):
    """rows: [{"label", "metrics", "arrays", "floor"?}] from weight_analysis
    on each run. The weight curves reuse the architecture experiment's
    cross-variant figures, with each run's own analytical curve."""
    from cmsi.experiments import architecture as arch

    vrows = []
    for r in rows:
        m = dict(r["metrics"])
        m.update(label=r["label"], experiment=r["label"], variant=r["label"],
                 _analytical_curve=r["arrays"]["analytical_curve"])
        vrows.append(m)
    return {
        "A_readability_by_run": fig_runs_readability(rows),
        "B_noise_by_run": fig_runs_noise(rows),
        "C_delta_over_error_by_run": fig_runs_delta_over_error(rows),
        "D_posterior_by_run": fig_runs_posterior(rows),
        "E_weight_vs_posterior_by_run": arch.fig_variants_weight_vs_posterior(
            vrows, "vis", by="run", legend="run"),
        "F_weight_vs_disparity_by_run": arch.fig_variants_weight_curves(
            vrows, "vis", by="run", legend="run"),
    }


def runs_table(rows):
    out = []
    for r in rows:
        m = r["metrics"]
        entry = {"run": r["label"], "control": m.get("control"), "hidden": r.get("hidden"),
                 "sigma0_sq": r.get("sigma0_sq"), "eye_sigma_sq": r.get("eye_sigma_sq"),
                 "floor": r.get("floor"),
                 "analytical_midpoint_deg": m["analytical"]["midpoint_deg"],
                 "posterior_mass_intermediate": float(np.mean((r["arrays"]["post_c1"] > 0.2)
                                                              & (r["arrays"]["post_c1"] < 0.8))),
                 "reads": {}}
        for key in ("vis", "prop"):
            rd = m["reads"][key]
            entry["reads"][key] = {
                "sigma_out": rd["sigma_out"], "midpoint_ls_deg": rd["midpoint_ls_deg"],
                "slope": rd["position_regression"]["slope"],
                "slope_ci95": rd["position_regression"]["slope_ci95"],
                "frac_readable": rd["frac_readable"], "w_raw": rd["w_raw"],
                "readability": rd["readability"]}
        entry["per_trial"] = m.get("per_trial")
        out.append(entry)
    return out


def print_runs(rows):
    width = max(10, max(len(r["label"]) for r in rows))
    print(f"\n{'run':>{width}} {'sig_out':>7} {'own sd':>6} {'floor':>5} | {'|D|<1':>5} "
          f"{'med|D|':>6} | {'out%':>5} {'gauss':>5} {'optim':>5} {'flatW':>5} {'from<1':>6} | "
          f"{'joint':>5} {'var sd':>6} | "
          f"{'slope':>5} {'mid_ls':>6} {'mid_an':>6} {'kept':>4} {'inter':>5}")
    for r in rows:
        m, rd = r["metrics"], r["metrics"]["reads"]["vis"]
        rb = rd["readability"]
        pt = (m.get("per_trial") or {"estimators": {}})["estimators"]
        pj = pt.get("joint", {}).get("outside", np.nan)
        pv = pt.get("variance", {}).get("sd_vs_posterior", np.nan)
        fl = r.get("floor", {}).get("fused_mu", np.nan)
        post = r["arrays"]["post_c1"]
        print(f"{r['label']:>{width}} {rd['sigma_out']:7.3f} {rb['own_sd']:6.3f} {fl:5.2f} | "
              f"{100 * rb['frac_absdelta_lt1']:5.1f} {rb['median_absdelta']:6.2f} | "
              f"{100 * rb['outside']:5.1f} {100 * rb['outside_predicted_own_sd']:5.1f} "
              f"{100 * rb['outside_predicted_sigma_out']:5.1f} "
              f"{100 * rb['outside_if_flat_posterior']:5.1f} "
              f"{100 * rb['outside_share_from_absdelta_lt1']:5.0f}% | "
              f"{100 * pj:5.1f} {pv:6.3f} | "
              f"{rd['position_regression']['slope']:5.3f} {rd['midpoint_ls_deg']:6.2f} "
              f"{m['analytical']['midpoint_deg']:6.2f} {100 * rd['frac_readable']:3.0f}% "
              f"{100 * np.mean((post > 0.2) & (post < 0.8)):4.0f}%")
    print("  (vis read) sig_out: control's residual sd; own sd: this run's; floor: decoding floor "
          "on sigma_out;\n  |D|<1: trials with |Delta| < 1 deg; out%: ratios outside [-1, 2]; "
          "gauss/optim: the fraction a Gaussian error of the own sd / of sigma_out predicts;\n"
          "  flatW: the observed fraction re-weighted to a flat posterior; from<1: share of "
          "the out-of-range ratios from |Delta| < 1 trials;\n  joint: ratios outside [-1, 2] "
          "when both position outputs are read together; var sd: sd of the weight read from "
          "the variance outputs\n  against the posterior, all trials; slope: position regression "
          "(1 = optimal); kept: trials with sigma_w < criterion; inter: posterior mass in 0.2-0.8")
