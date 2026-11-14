"""Stage 8 -- behavioral analyses of one trained network: what its hand
estimate does, read from the network's outputs alone (no unit is looked at),
as figures under

    results/<run>/figures/model/behavioral_analysis/<figure>/

one folder per figure (the composed figure named as its folder and copied flat
beside it, its panels, a numbers.json). The paper's Fig. 4 is also written to
the run's manuscript folder (results/manuscript/ for the flagship,
results/manuscript_<run>/ otherwise) as fig4_bias_disparity; no other
analysis of this script writes there. Torch-free (scripts/neuronal/npmodel.py runs the network
in numpy); house style (cmsi.viz.style / manuscript); png, tif and svg.

    python scripts/08_behavioral.py fig4             the paper's Fig. 4, three panels: A the relative bias
                                                     against true disparity at two priors (bias-disparity),
                                                     B the hand bias against signed true disparity on probe
                                                     trials (panel B of dokka-probe), C the hand bias by
                                                     inferred cause (decision-bias). Runs the three analyses, which
                                                     also write their own folders
    python scripts/08_behavioral.py bias-disparity   relative bias against true disparity (Körding et al. 2007,
                                                     Fig. 2e) at two priors (0.3, 0.5) of the run's own
                                                     configuration (see prior-configs): the hand report's
                                                     shift from the TRUE hand position over the true distance, on
                                                     probe trials of fixed pairs (nine positions, 2.5 deg apart
                                                     for the main network)
    python scripts/08_behavioral.py decision-bias    hand bias toward vision by the cause the network inferred
                                                     (implied visual weight above or below 0.5), as a bar plot
                                                     (Dokka et al. 2019, Fig. 4C; Rideaux et al. 2021,
                                                     Fig. 2D-E), on the test trials with a matched stimulus:
                                                     two true causes, a true distance within 40% of the run's
                                                     transition midpoint (3 to 7 deg for the main network).
                                                     Panel C of the paper's Fig. 4;
    python scripts/08_behavioral.py dokka-probe      the three curves of Dokka et al. (2019, Fig. 1E-G, 2A, C, D)
                                                     on probe trials of fixed TRUE disparities: share of trials
                                                     inferred as one cause, bias of the hand estimate from the
                                                     true hand position, and its SD over trials (dokka_probe/,
                                                     a model figure, no manuscript copy)
    python scripts/08_behavioral.py prior-configs    write the config files of the run's prior family (the
                                                     run's config with p_common 0.3, 0.5) that are
                                                     missing, and say which members still have to be trained;
                                                     panel A of Fig. 4 draws these networks
    python scripts/08_behavioral.py all              every analysis above except prior-configs (the default)
    python scripts/08_behavioral.py --run large_bias fig4      another configuration

Option: --run <name> (default flagship).

The unit-level analyses (units, r5) are in scripts/neuronal/run.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "neuronal"))   # the torch-free helpers: common.py, npmodel.py
from common import (COLORS, PANEL_LW, PANEL_MS, Bundle,                      # noqa: E402
                    apply_style, bootstrap_ci, finish_panel, hybrid_reads,
                    manuscript_dir, manuscript_grid, panel_axes,
                    save_figures, save_json)
from cmsi.viz.style import FORMATS  # noqa: E402  (png, tif, svg)
from cmsi.viz.manuscript import FIG_BIAS_DISPARITY  # noqa: E402
from npmodel import ROOT  # noqa: E402  (the repository root)

_BUNDLES = {}

# Per-run folders are named by content (like the run's other model figures);
# the paper name (FIG_BIAS_DISPARITY in viz/manuscript.py) is used only for
# the copy of Fig. 4 in the run's manuscript folder. The composed figure in
# each folder is named as the folder, so `save_figures` also drops a flat png
# copy beside it.
K_BIAS = "decision_bias"                     # the hand bias by inferred cause, matched trials: panel C of the paper's Fig. 4
K_DOKKA = "dokka_probe"                      # Dokka et al. (2019) Fig. 1E-G on probe trials; model figure only
K_BIAS_DISPARITY = "bias_disparity"   # relative bias against true disparity (Körding et al. 2007, Fig. 2e); panel A of the paper's Fig. 4
K_FIG4 = "bias_three_panels"          # the paper's Fig. 4: bias_disparity, panel B of dokka_probe and decision_bias
PAPER = {K_FIG4: FIG_BIAS_DISPARITY}
_PANELS = {}   # (analysis key, run) -> the panel callables of an analysis already run in this process (used by fig4)


def out_dir(run="flagship"):
    """Where the behavioral analyses of `run` go: beside the run's other model
    figures, results/<run>/figures/model/behavioral_analysis/, one folder per
    figure (composed figure named as its folder, panels, numbers)."""
    return ROOT / "results" / run / "figures" / "model" / "behavioral_analysis"


# set by main() from --run; module defaults for interactive use
RUN = "flagship"
OUT, MDIR = out_dir(RUN), manuscript_dir(RUN)


def configure(run="flagship"):
    """Point every analysis at `run`: outputs under out_dir(run), manuscript
    copies under manuscript_dir(run)."""
    global RUN, OUT, MDIR
    RUN = run
    OUT, MDIR = out_dir(run), manuscript_dir(run)


def save_all(figs, key):
    """Every figure of one analysis -> OUT/<key>/ (panels, the composed
    figure named `key`, numbers alongside); the composed figure also -> the
    run's manuscript folder under its paper name (PAPER), flat copy
    included, so the manuscript folder holds it like every other figure."""
    paper = PAPER[key]
    save_figures({f"{paper}/{paper}": figs[f"{key}/{key}"]}, MDIR, formats=FORMATS, close=False)
    save_figures(figs, OUT, formats=FORMATS)
    print(f"   -> {OUT / key}/ and {MDIR / paper}/ (+ flat copies)")


def bundle(run):
    if run not in _BUNDLES:
        _BUNDLES[run] = Bundle(run)
    return _BUNDLES[run]


# --------------------------------------------------------------------------- #
# decision-bias: the hand bias by inferred cause, on the test trials (Fig. 4C)
# --------------------------------------------------------------------------- #
# Dokka et al. (2019, Fig. 4C) split their trials by the reported percept
# (object stationary or moving) and Rideaux et al. (2021, Fig. 2D-E) by the
# number of events their network reported (decision unit above or below 0.5),
# and compared the mean bias of the two groups in a bar plot. Here the test
# split of the run is divided the same way by the network's own implied
# visual weight (the hybrid read, exactly as 03_analyze computes it): above
# `threshold` the trial counts as one cause, otherwise as two. The bias of a
# trial is the pull of the hand estimate from the observer's hand-alone
# estimate, signed so that it is positive toward the seen hand (the sign of
# the measured body-frame disparity), in degrees: the quantity of the model
# figure 12_behavioral_bias, signed toward vision. Error bars: bootstrap 95%
# CI of the mean.
#
# Both studies compare the two groups at the SAME stimulus (Dokka et al.: the
# object speeds at which the two reports were about equally frequent; Rideaux
# et al.: cues on the decision boundary). This analysis does the same on the
# test split: only trials with two true causes whose true
# positions are `band` deg apart. The band has to lie around the run's OWN
# transition midpoint, where about half the trials fall on each side of the
# threshold: below the midpoint most trials count as one cause and the few
# two-cause trials are those whose noise widened the disparity, so their pull
# is the larger one and the bars reverse. By default the band is therefore
# 0.6 to 1.4 times the run's fitted midpoint (metrics.json, `transition`),
# rounded to 0.5 deg: 3 to 7 deg for the main network (midpoint 4.92 deg).
def matched_band(b, rel=(0.6, 1.4)):
    """The band of true distances around the run's transition midpoint (deg)."""
    try:
        mid = float(b.metrics["transition"]["network"]["midpoint_deg"])
    except (KeyError, TypeError):
        raise ValueError(f"{b.run}: no fitted transition midpoint in metrics.json; pass `band` explicitly") from None
    if not np.isfinite(mid) or mid <= 0:
        raise ValueError(f"{b.run}: transition midpoint {mid}; pass `band` explicitly")
    return (round(2 * rel[0] * mid) / 2, round(2 * rel[1] * mid) / 2), mid


def decision_bias(run=None, threshold=0.5, band=None):
    run = run or RUN
    b = bundle(run)
    midpoint = None
    if band is None:
        band, midpoint = matched_band(b)
    d, pred = b.d, b.pred
    w, w_hand = hybrid_reads(b, pred)
    i_prop = b.idx("mu_prop")
    toward = np.sign(d["disparity"])                       # side of the seen hand (measured, body frame)
    bias = toward * (pred[:, i_prop] - d["seg_prop_mu"])
    bias_obs = toward * d["post_c1"] * (d["fused_mu"] - d["seg_prop_mu"])
    one = np.where(np.isfinite(w), w, d["post_c1"]) > threshold
    true_disp = d["s_vis"] - d["s_prop"]
    # the hand estimate's shift from the TRUE hand position toward the TRUE seen position (defined for two causes)
    bias_true = np.sign(true_disp) * (pred[:, i_prop] - d["s_prop"])
    use = (d["C"] == 2) & (np.abs(true_disp) >= band[0]) & (np.abs(true_disp) < band[1])

    def group(mask, values=bias):
        mask = mask & use
        v = values[mask]
        out = {"n": int(mask.sum()), "share": float(mask.sum() / use.sum()),
               "bias": float(v.mean()), "bias_sem": float(v.std(ddof=1) / np.sqrt(len(v))),
               "bias_ci": bootstrap_ci(v),
               "observer_bias": float(bias_obs[mask].mean()), "observer_bias_ci": bootstrap_ci(bias_obs[mask]),
               "abs_measured_disparity": float(np.abs(d["disparity"][mask]).mean()),
               "true_one_cause_share": float(np.mean(d["C"][mask] == 1)),
               "posterior": float(d["post_c1"][mask].mean()), "weight": float(w[mask].mean())}
        out["abs_true_disparity"] = float(np.abs(true_disp[mask]).mean())
        out["bias_from_true_hand"] = float(bias_true[mask].mean())
        out["bias_from_true_hand_ci"] = bootstrap_ci(bias_true[mask])
        return out

    one_hand = w_hand > threshold                          # the same split read from the hand channel
    one_obs = d["post_c1"] > threshold                     # the observer split by its own posterior
    res = {"run": run, "threshold": float(threshold), "n_test": int(len(w)), "n": int(use.sum()),
           "trials": f"test trials with two true causes, true positions {band[0]:g} to {band[1]:g} deg apart",
           "band_deg": list(band), "transition_midpoint_deg": midpoint,
           "n_weight_not_finite": int((~np.isfinite(w)).sum()),
           "bias": "sign(measured disparity) * (network hand estimate - observer's hand-alone estimate), deg",
           "one": group(one), "two": group(~one),
           "checks": {"split_on_hand_weight": {"one": group(one_hand), "two": group(~one_hand)},
                      "observer_split_on_own_posterior": {
                          "agreement_with_network_split": float((one_obs == one).mean()),
                          "one": group(one_obs, bias_obs), "two": group(~one_obs, bias_obs)}}}
    save_json(res, OUT / K_BIAS / "numbers.json")
    print(f"decision-bias on {res['trials']} (n {res['n']})")
    for g in ("one", "two"):
        r = res[g]
        print(f"decision-bias {g:3s} cause: n {r['n']:4d} ({100 * r['share']:.1f}%), bias {r['bias']:+.3f} "
              f"[{r['bias_ci'][0]:+.3f}, {r['bias_ci'][1]:+.3f}] deg, observer on the same trials "
              f"{r['observer_bias']:+.3f}; mean |measured disparity| {r['abs_measured_disparity']:.2f} deg")

    def panel(ax=None):
        ax = panel_axes(ax)
        x = np.array([1, 2])
        m = np.array([res["one"]["bias"], res["two"]["bias"]])
        lo = np.array([res["one"]["bias_ci"][0], res["two"]["bias_ci"][0]])
        hi = np.array([res["one"]["bias_ci"][1], res["two"]["bias_ci"][1]])
        ax.bar(x, m, width=0.6, color="0.5", zorder=2)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="k", elinewidth=PANEL_LW,
                    capsize=3, zorder=3)
        ax.set_xlim(0.3, 2.7)
        ax.set_xticks(x)
        ax.set_ylim(0, 1.25 * hi.max())
        return finish_panel(ax, "inferred number of causes", "hand bias toward vision (deg)", "", None)

    save_figures({f"{K_BIAS}/{K_BIAS}": panel()}, OUT, formats=FORMATS)
    print(f"   -> {OUT / K_BIAS}/ (+ flat copy)")
    _PANELS[(K_BIAS, run)] = panel
    return panel


# --------------------------------------------------------------------------- #
# dokka-probe: the three curves of Dokka et al. (2019), Fig. 1E-G, on probe trials
# --------------------------------------------------------------------------- #
# Dokka et al. fixed the conflict (the object's speed in the world, both
# directions), repeated each level over many trials and reported, against the
# signed speed: the proportion of "object stationary" reports (Fig. 1E, 2A),
# the bias of perceived heading (1F, 2C) and the heading discrimination
# threshold (1G, 2D), the last two read from a psychometric function fitted
# to left/right choices. Here the conflict is the TRUE disparity between the
# seen and the felt hand, fixed per level:
#
#   felt hand at the prior mean (mu0), so that the hand-alone estimate has no
#     pull toward the prior and the bias can return to zero at large disparities;
#   seen hand at mu0 + D; eyes at their mean (eye_mu); the three noise
#     variances at the middle of their training ranges;
#   n trials per level, each with its own measurement noise and Poisson spikes;
#     the same noisy measurements go to the network and to the observer.
#
# The network gives a continuous estimate, not a choice, so no psychometric
# function is fitted: per level
#   A  the share of trials with the implied visual weight above 0.5 (inferred
#      one cause, as in Fig. 4C); for the observer, its posterior above 0.5;
#   B  the mean of (hand estimate - true hand position), deg, signed;
#   C  the SD over trials of the hand estimate, deg (what a psychometric
#      function's width measures).
# The levels follow the run's own transition midpoint (multiples PROBE_MULT of
# it, rounded to 0.5 deg, both signs, and 0), because the curve's peak sits
# near the midpoint and a fixed grid would miss it in another configuration.
# A level is dropped when it lies outside what the network was trained on:
# beyond 3 sd of the position prior, or with the seen hand within 3 sd of
# visual noise of the edge of the reliably encoded visual field.
PROBE_MULT = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)


def probe_levels(b):
    """Signed true disparities (deg) of the probe trials of run `b`, and its midpoint."""
    from cmsi.data.generative import containment_bounds
    from common import mid_sigmas
    gen, enc = b.cfg["generative"], b.cfg["encoding"]
    try:
        mid = float(b.metrics["transition"]["network"]["midpoint_deg"])
    except (KeyError, TypeError):
        raise ValueError(f"{b.run}: no fitted transition midpoint in metrics.json") from None
    if not np.isfinite(mid) or mid <= 0:
        raise ValueError(f"{b.run}: transition midpoint {mid}")
    lo, hi = containment_bounds(enc)
    margin = 3.0 * np.sqrt(mid_sigmas(gen)["vis"])
    retinal0 = float(gen["mu0"]) - float(gen["eye_mu"])       # retinal position of the seen hand at D = 0
    pos = sorted({round(2 * m * mid) / 2 for m in PROBE_MULT})
    signed = [-D for D in reversed(pos)] + [0.0] + pos
    keep = [D for D in signed
            if abs(D) <= 3.0 * np.sqrt(gen["sigma0_sq"]) and lo + margin <= retinal0 + D <= hi - margin]
    return keep, mid


def dokka_probe(run=None, n=4000, seed=0, threshold=0.5):
    run = run or RUN
    from cmsi.data.generative import observer
    from common import encode, hybrid_weight, mid_sigmas
    b = bundle(run)
    gen, enc, encoders = b.cfg["generative"], b.cfg["encoding"], b.d_full["encoders"]
    levels, mid = probe_levels(b)
    sig2 = mid_sigmas(gen)
    mu0, eye_mu = float(gen["mu0"]), float(gen["eye_mu"])
    i_prop, i_vis, i_vv = b.idx("mu_prop"), b.idx("mu_vis"), b.idx("var_vis")
    rng = np.random.default_rng(seed)
    rows = []
    for D in levels:
        s_p, s_v = mu0, mu0 + D
        d = {"x_vis": rng.normal(s_v - eye_mu, np.sqrt(sig2["vis"]), n),      # retinal = s_vis - eye
             "x_eye": rng.normal(eye_mu, np.sqrt(sig2["eye"]), n),
             "x_prop": rng.normal(s_p, np.sqrt(sig2["prop"]), n),
             "sig2_vis": np.full(n, sig2["vis"]), "sig2_prop": np.full(n, sig2["prop"]),
             "sig2_eye": np.full(n, sig2["eye"])}
        obs = observer(d, gen)
        pred = b.net.predict(encode(d, encoders, enc, rng))
        w, _, _ = hybrid_weight(pred[:, i_vv], obs["fused_var"], obs["seg_vis_var"],
                                obs["fused_mu"] - obs["seg_vis_mu"], pred[:, i_vis], obs["seg_vis_mu"],
                                b.sig_out[i_vis], b.sig_out[i_vv])
        one = {"network": np.where(np.isfinite(w), w, obs["post_c1"]) > threshold,
               "observer": obs["post_c1"] > threshold}
        est = {"network": pred[:, i_prop], "observer": obs["mu_prop"],
               "fused": obs["fused_mu"], "segregated": obs["seg_prop_mu"]}
        row = {"disparity": float(D), "s_p": s_p, "s_v": s_v, "n": int(n),
               "posterior": float(obs["post_c1"].mean()), "weight": float(np.mean(w)),
               "n_weight_not_finite": int((~np.isfinite(w)).sum())}
        for k, e in est.items():
            err = e - s_p
            row[k] = {"bias": float(err.mean()), "bias_sem": float(err.std(ddof=1) / np.sqrt(n)),
                      "sd": float(err.std(ddof=1)), "sd_ci": bootstrap_ci(err, stat=lambda v: v.std(ddof=1), n=1000)}
            if k in one:
                p1 = float(one[k].mean())
                row[k]["one_cause"] = p1
                row[k]["one_cause_se"] = float(np.sqrt(p1 * (1 - p1) / n))
        rows.append(row)
        print(f"dokka-probe D {D:+6.1f}: one cause {row['network']['one_cause']:.2f} ({row['observer']['one_cause']:.2f}), "
              f"bias {row['network']['bias']:+.2f} ({row['observer']['bias']:+.2f}), "
              f"SD {row['network']['sd']:.2f} ({row['observer']['sd']:.2f})   network (observer)")
    res = {"run": run, "n_per_level": int(n), "seed": int(seed), "threshold": float(threshold),
           "transition_midpoint_deg": mid, "felt_hand_deg": mu0, "eye_deg": eye_mu, "sig2": sig2,
           "bias": "mean of (hand estimate - true hand position), deg",
           "sd": "SD over trials of the hand estimate, deg; sd_ci bootstrap 95% CI (1,000 resamples)",
           "one_cause": "share of trials with the implied visual weight above the threshold "
                        "(observer: posterior above the threshold)",
           "levels": rows}
    save_json(res, OUT / K_DOKKA / "numbers.json")
    who = (("network", COLORS["network"], "o-", {}, "network"),
           ("observer", COLORS["analytical"], "o--", {"mfc": "none"}, "Bayesian observer"))

    def draw(ax, value, lower, upper, ylabel, legend=None, xmax=None):
        # xmax: draw only the levels with |disparity| <= xmax, on an x axis from -xmax to xmax
        ax = panel_axes(ax)
        shown = [r for r in rows if xmax is None or abs(r["disparity"]) <= xmax + 1e-9]
        xs = np.array([r["disparity"] for r in shown])
        for k, col, fmt, kw, lab in who:
            m = np.array([value(r[k]) for r in shown])
            lo = np.array([lower(r[k]) for r in shown])
            hi = np.array([upper(r[k]) for r in shown])
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=fmt, lw=PANEL_LW if k == "network" else 0.9,
                        ms=PANEL_MS, color=col, capsize=1.5, label=lab, zorder=4 if k == "network" else 3, **kw)
        if xmax is not None:
            ax.set_xlim(-xmax, xmax)
            ax.set_xticks(np.arange(-xmax, xmax + 1e-9, xmax / 2))
        ax.axvline(0, lw=0.5, color="gray", ls=":")
        fig = finish_panel(ax, "true disparity (deg)", ylabel, "", None)
        if legend:
            ax.legend(fontsize=7, loc=legend, handlelength=1.8, borderaxespad=0.3, labelspacing=0.3)
        return ax, fig

    def pA(ax=None):                       # Dokka Fig. 1E, 2A
        ax, fig = draw(ax, lambda v: v["one_cause"], lambda v: v["one_cause"] - 1.96 * v["one_cause_se"],
                       lambda v: v["one_cause"] + 1.96 * v["one_cause_se"], "proportion inferred one cause")
        ax.set_ylim(0, 1.05)
        return fig

    def pB(ax=None, xmax=None):            # Dokka Fig. 1F, 2C
        ax, fig = draw(ax, lambda v: v["bias"], lambda v: v["bias"] - 1.96 * v["bias_sem"],
                       lambda v: v["bias"] + 1.96 * v["bias_sem"], "hand bias (deg)", legend="upper left",
                       xmax=xmax)
        ax.axhline(0, lw=0.5, color="gray")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.35 * (hi - lo))          # headroom for the legend
        return fig

    def pC(ax=None):                       # Dokka Fig. 1G, 2D
        ax, fig = draw(ax, lambda v: v["sd"], lambda v: v["sd_ci"][0], lambda v: v["sd_ci"][1],
                       "SD of hand estimate (deg)")
        ax.set_ylim(0, 1.15 * ax.get_ylim()[1])
        return fig

    figs = {f"{K_DOKKA}/{K_DOKKA}": manuscript_grid([pA, pB, pC], ncols=3)}
    save_figures(figs, OUT, formats=FORMATS)
    print(f"   -> {OUT / K_DOKKA}/ (+ flat copy)")
    _PANELS[(K_DOKKA, run)] = {"A": pA, "B": pB, "C": pC}
    return _PANELS[(K_DOKKA, run)]


# --------------------------------------------------------------------------- #
# bias-disparity: the relative bias of Körding et al. (2007), Fig. 2e, on probe trials
# --------------------------------------------------------------------------- #
# Their measure is the shift of the reported position of one cue from its TRUE
# position, as a fraction of the true distance to the other cue,
# (s_A_hat - s_A) / (s_V - s_A), averaged over repeated noisy trials of fixed
# stimulus pairs and pooled over pairs at the same distance. Here the sound is
# the hand and the flash is the seen object: (y_hand - s_p) / (s_v - s_p). The
# network never sees s_p or s_v; it sees spike counts of noisy measurements of
# them. The bias curve of the model figure 12_behavioral_bias is a different
# quantity: the pull from the hand-alone ESTIMATE, in degrees, against the
# MEASURED disparity.
REFERENCES = ("network", "observer", "fused", "segregated")


def _probe_pairs(b, pairs, n, rng):
    """n noisy trials of each (s_p, s_v) pair of true positions, with the eyes
    at their mean (0 deg in the main network and its satellites, where the
    retinal and the body-frame position then coincide) and
    the three noise variances at the middle of their training ranges. Returns
    per pair the mean shift of the hand
    report from the true hand position, est - s_p, and its SEM over trials,
    for the network and three references: the Bayesian observer (model
    averaging), always fuse (the fused estimate) and never fuse (the
    hand-alone estimate)."""
    from cmsi.data.generative import observer
    from common import encode, mid_sigmas
    gen, enc, encoders = b.cfg["generative"], b.cfg["encoding"], b.d_full["encoders"]
    sig2 = mid_sigmas(gen)
    i_prop = b.idx("mu_prop")
    eye = float(gen["eye_mu"])      # the eyes at their mean: 0 in the main network and its satellites
    rows = []
    for s_p, s_v in pairs:
        d = {"x_vis": rng.normal(s_v - eye, np.sqrt(sig2["vis"]), n),      # retinal = s_vis - eye
             "x_eye": rng.normal(eye, np.sqrt(sig2["eye"]), n),
             "x_prop": rng.normal(s_p, np.sqrt(sig2["prop"]), n),
             "sig2_vis": np.full(n, sig2["vis"]), "sig2_prop": np.full(n, sig2["prop"]),
             "sig2_eye": np.full(n, sig2["eye"])}
        obs = observer(d, gen)
        pred = b.net.predict(encode(d, encoders, enc, rng))
        est = {"network": pred[:, i_prop], "observer": obs["mu_prop"],
               "fused": obs["fused_mu"], "segregated": obs["seg_prop_mu"]}
        row = {"s_p": float(s_p), "s_v": float(s_v), "disparity": float(s_v - s_p), "n": int(n),
               "posterior": float(obs["post_c1"].mean())}
        for k, e in est.items():
            shift = e - s_p
            row[k] = {"shift": float(shift.mean()), "shift_sem": float(shift.std(ddof=1) / np.sqrt(n))}
        rows.append(row)
    return rows


def _pool_by_distance(rows):
    """Körding's average: per pair the relative bias, (mean shift) / (s_v - s_p),
    which is positive when the report moves toward vision on either side;
    pairs at the same distance averaged with equal weight. Also the shift
    toward vision in degrees (the shift times the sign of the disparity)."""
    out = []
    for D in sorted({abs(r["disparity"]) for r in rows}):
        sel = [r for r in rows if abs(r["disparity"]) == D]
        entry = {"distance": D, "pairs": len(sel), "posterior": float(np.mean([r["posterior"] for r in sel]))}
        for k in REFERENCES:
            rel = np.array([r[k]["shift"] / r["disparity"] for r in sel])
            rel_sem = np.array([r[k]["shift_sem"] / abs(r["disparity"]) for r in sel])
            deg = np.array([r[k]["shift"] * np.sign(r["disparity"]) for r in sel])
            deg_sem = np.array([r[k]["shift_sem"] for r in sel])
            entry[k] = {"relative": float(rel.mean()),
                        "relative_sem": float(np.sqrt((rel_sem ** 2).sum()) / len(sel)),
                        "toward_vision_deg": float(deg.mean()),
                        "toward_vision_deg_sem": float(np.sqrt((deg_sem ** 2).sum()) / len(sel))}
        out.append(entry)
    return out


def _test_split_check(b, edges=(2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30)):
    """The same measure on the stored test trials, binned by the true distance
    (random positions and eye positions instead of fixed pairs): a check on
    the probe design, not part of the figure."""
    d = b.d
    D = d["s_vis"] - d["s_prop"]
    shifts = {"network": b.pred[:, b.idx("mu_prop")] - d["s_prop"], "observer": d["mu_prop"] - d["s_prop"]}
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (np.abs(D) >= lo) & (np.abs(D) < hi)
        if m.sum() < 30:
            continue
        e = {"from": lo, "to": hi, "n": int(m.sum()), "mean_distance": float(np.abs(D[m]).mean())}
        for k, s in shifts.items():
            e[k] = float(np.mean(s[m] * np.sign(D[m])) / np.mean(np.abs(D[m])))
        out.append(e)
    return out


# Panel A is drawn at these priors, always in the configuration of the run
# asked for: the run's config with only p_common changed.
PRIOR_FAMILY = (0.3, 0.5)


def prior_family(run):
    """(base, [(p_common, run name)]) for the priors PRIOR_FAMILY of `run`'s configuration.

    Main configuration (flagship, pcommonNN): flagship is p = 0.5, p = 0.3 is
    pcommon03. Any other configuration `base` (a run named base or base_pNN):
    the member at the base config's own p_common is `base`, the others are
    base_p03 and base_p05. A member's config
    is its base's with only p_common changed (`prior-configs` writes them)."""
    import re
    import yaml
    if run == "flagship" or re.fullmatch(r"pcommon\d+", run):
        base, base_p = "flagship", 0.5
        name = lambda q: "flagship" if abs(q - 0.5) < 1e-9 else f"pcommon{int(round(10 * q)):02d}"   # noqa: E731
    else:
        base = re.sub(r"_p\d\d$", "", run)
        src = [f for f in (ROOT / "configs" / f"{base}.yaml", ROOT / "results" / base / "config.yaml") if f.exists()]
        if not src:
            raise ValueError(f"{run}: neither configs/{base}.yaml nor results/{base}/config.yaml exists")
        base_p = float(yaml.safe_load(src[0].read_text())["generative"]["p_common"])
        name = lambda q: base if abs(q - base_p) < 1e-9 else f"{base}_p{int(round(10 * q)):02d}"   # noqa: E731
    return base, [(q, name(q)) for q in PRIOR_FAMILY]


def _trained(r):
    return (ROOT / "results" / r / "model.pt").exists()


def prior_configs(run=None):
    """Write the config files of the missing members of `run`'s prior family:
    configs/<member>.yaml = configs/<base>.yaml with ONLY the p_common line
    changed. Existing files are never overwritten. Prints what to train."""
    import re
    run = run or RUN
    base, members = prior_family(run)
    src = ROOT / "configs" / f"{base}.yaml"
    text = src.read_text()
    pat = re.compile(r"^(\s+p_common:\s*)([0-9.]+)", re.M)
    assert len(pat.findall(text)) == 1, f"{src}: expected one p_common line"
    for q, r in members:
        dst = ROOT / "configs" / f"{r}.yaml"
        if not dst.exists():
            head = (f"# {r}: configs/{base}.yaml with ONLY p_common changed to {q:g}, for the prior panel\n"
                    f"# (Fig. 4A; scripts/08_behavioral.py prior-configs). Everything below, the header\n"
                    f"# included, is the base file's; where its header describes p_common, read {q:g}.\n#\n")
            dst.write_text(head + pat.sub(lambda m: f"{m.group(1)}{q:g}", text))
            print(f"   wrote configs/{r}.yaml (p_common {q:g})")
        state = "trained" if _trained(r) else f"NOT trained -> bash scripts/run_all.sh --config configs/{r}.yaml"
        print(f"   p = {q:g}: {r}: {state}")


def kording_positions(b):
    """The nine probe positions of run `b` (deg): the prior mean and 0.25,
    0.5, 0.75 and 1 prior sd on either side. For the main network and its
    satellites (prior mean 0, sd 10 deg) these are -10 to 10 deg in 2.5 deg
    steps, so the distances are 2.5 to 20 deg (Körding et al. used the five
    positions -10, -5, 0, 5, 10 deg). A
    configuration with another prior gets the same grid in units of ITS prior,
    so the distances cover its own transition and the positions stay where
    it was trained."""
    gen = b.cfg["generative"]
    mu0, sd0 = float(gen["mu0"]), float(np.sqrt(gen["sigma0_sq"]))
    return tuple(float(mu0 + k * sd0) for k in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0))


def _crosses(ax, legend, lines, steps=60):
    """True when any of `lines` (their markers or the straight segments between
    them) passes through the box of `legend`."""
    ax.figure.canvas.draw()
    box = legend.get_window_extent().padded(2)
    for line in lines:
        xy = ax.transData.transform(np.asarray(line.get_xydata(), float))
        for a, b in zip(xy[:-1], xy[1:]):
            t = np.linspace(0.0, 1.0, steps)[:, None]
            pts = a + t * (b - a)
            if np.any((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)):
                return True
        if len(xy) == 1 and box.contains(*xy[0]):
            return True
    return False


BIAS_XMAX_DEG = 25.0   # x axis of panel A: 0 to 25 deg of true disparity, whatever the configuration
# Per-configuration settings of panel A, by base configuration (the first name
# prior_family returns). "points": the distances drawn (deg, both ends kept);
# "ylim", "ystep": the y axis (%) and its tick step. A configuration not listed
# draws every distance up to BIAS_XMAX_DEG on a y axis from 0 fitted to the data.
PANEL_A_AXES = {
    "large_bias": {"points": (5.0, 20.0), "ylim": (10.0, 45.0), "ystep": 5.0},
}
# Panel B of Fig. 4 (the probe-trial bias curve), by base configuration: only the
# levels with |true disparity| <= this many deg, on an x axis from -x to x. A
# configuration not listed shows every level. The model figure dokka_probe is not affected.
PANEL_B_XMAX = {
    "large_bias": 50.0,
}


def bias_disparity(run=None, n=4000, seed=0, runs=None):
    """Körding et al. (2007), Fig. 2e: the relative bias of the hand report
    toward vision on probe trials of fixed pairs of true positions (the nine
    positions of `kording_positions`, every pair of different positions), n
    noisy trials per pair, for each network and its Bayesian observer on the
    same trials. The networks are the priors of PRIOR_FAMILY (0.3, 0.5) in the
    configuration of `run` (`prior_family`), i.e. the run's config with only
    p_common changed; a member that is not trained yet is left out and named
    with the command that trains it. Every network gets the same noise draws
    (the same seed), so the curves differ only by the network, and each
    network is compared with its own observer. The always-fuse and never-fuse
    values are kept in numbers.json but not drawn. Output goes to the folder
    of `run` (the main network by default).

    The x axis runs from 0 to BIAS_XMAX_DEG = 25 deg, ticks every 5 deg;
    distances beyond it are left out of the panel and kept in numbers.json.
    PANEL_A_AXES sets the points drawn and the y axis per configuration."""
    run = run or RUN
    from common import mid_sigmas
    if runs is None:
        base, members = prior_family(run)
        runs = tuple(r for _, r in members if _trained(r))
        for q, r in members:
            if not _trained(r):
                print(f"   bias-disparity: no network at p_common = {q:g} for the configuration '{base}' "
                      f"(results/{r} missing); train it with\n"
                      f"      python scripts/08_behavioral.py prior-configs --run {run}\n"
                      f"      bash scripts/run_all.sh --config configs/{r}.yaml")
        if not runs:
            runs = (run,)
    res = {"main_run": run, "seed": seed,
           "design": {"positions": "per run: prior mean and 0.25, 0.5, 0.75, 1 prior sd on either side",
                      "trials_per_pair": n, "eye": "eye_mu of the run"},
           "runs": {}}
    for r in runs:
        b = bundle(r)
        positions = kording_positions(b)
        grid = [(s_p, s_v) for s_p in positions for s_v in positions if s_v != s_p]
        rows = _probe_pairs(b, grid, n, np.random.default_rng(seed))
        res["runs"][r] = {"p_common": b.p_common, "visual_field": b.cfg["encoding"]["visual_field"],
                          "positions": positions,
                          "variances": mid_sigmas(b.cfg["generative"]),
                          "kording_design": _pool_by_distance(rows), "pairs": rows,
                          "test_split_check": _test_split_check(b)}
        for e in res["runs"][r]["kording_design"]:
            print(f"{r:9s} p={b.p_common:.1f} D={e['distance']:4.1f}: network {100 * e['network']['relative']:5.1f}% "
                  f"(sem {100 * e['network']['relative_sem']:.2f}) observer {100 * e['observer']['relative']:5.1f}%")
    save_json(res, OUT / K_BIAS_DISPARITY / "numbers.json")

    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.15, vmax=0.95)        # the viridis ramp of Fig. 6 (priors 0.2 to 0.9)
    ramp = plt.get_cmap("viridis")

    def panel(ax=None):
        ax = panel_axes(ax)
        top, h_prior, drawn = 0.0, [], []
        axes_cfg = PANEL_A_AXES.get(prior_family(run)[0], {})
        d_lo, d_hi = axes_cfg.get("points", (0.0, BIAS_XMAX_DEG))
        lo_shown, hi_shown = np.inf, -np.inf
        for r in runs:
            rr = res["runs"][r]
            entries = rr["kording_design"]
            entries = [e for e in entries if d_lo - 1e-9 <= e["distance"] <= d_hi + 1e-9]   # what the panel shows
            x = np.array([e["distance"] for e in entries])
            net = 100 * np.array([e["network"]["relative"] for e in entries])
            net_se = 100 * np.array([e["network"]["relative_sem"] for e in entries])
            obs = 100 * np.array([e["observer"]["relative"] for e in entries])
            c = ramp(norm(rr["p_common"]))
            h_prior.append(ax.errorbar(x, net, yerr=net_se, fmt="o-", lw=PANEL_LW, ms=PANEL_MS, color=c,
                                       capsize=2, zorder=4))
            drawn.append(h_prior[-1].lines[0])
            drawn += ax.plot(x, obs, "o--", lw=0.9, ms=PANEL_MS, mfc="none", color=c, zorder=3)
            top = max(top, net.max(), obs.max())
            lo_shown = min(lo_shown, (net - net_se).min(), obs.min())
            hi_shown = max(hi_shown, (net + net_se).max(), obs.max())
        k_net = ax.plot([], [], "o-", lw=PANEL_LW, ms=PANEL_MS, color="#555555")[0]
        k_obs = ax.plot([], [], "o--", lw=0.9, ms=PANEL_MS, mfc="none", color="#555555")[0]
        ax.axhline(0, lw=0.5, color="gray")
        # 0 to 25 deg in every configuration, ticks every 5 deg
        ax.set_xlim(0.0, BIAS_XMAX_DEG)
        ax.set_xticks(np.arange(0.0, BIAS_XMAX_DEG + 1e-9, 5.0))
        if "ylim" in axes_cfg:
            y0, y1 = axes_cfg["ylim"]
            ax.set_ylim(y0, y1)
            ax.set_yticks(np.arange(y0, y1 + 1e-9, axes_cfg["ystep"]))
            if lo_shown < y0 or hi_shown > y1:
                print(f"   bias-disparity: WARNING, values from {lo_shown:.1f} to {hi_shown:.1f} % fall outside "
                      f"the fixed y axis {y0:g} to {y1:g} % (PANEL_A_AXES)")
        else:
            ax.set_ylim(0, 1.15 * top)
        fig = finish_panel(ax, "true disparity (deg)", "hand bias toward vision (%)",
                           "", None)
        first = ax.legend(h_prior, [f"{res['runs'][r]['p_common']:.1f}" for r in runs],
                          title="$p_{\\mathrm{common}}$", title_fontsize=7, fontsize=7, loc="upper right",
                          handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)
        ax.add_artist(first)
        # the network / observer key, on two lines so that it stays narrow: right under the prior
        # legend if no curve passes there, otherwise the first other corner no curve crosses
        # (the curves of a configuration with a wide prior fall slowly and cross the first spot)
        ax.figure.canvas.draw()
        y0 = first.get_window_extent().transformed(ax.transAxes.inverted()).y0
        for loc, anchor in (("upper right", (1.0, y0)), ("lower left", None), ("center right", None),
                            ("lower right", None), ("center left", None)):
            kw = {"bbox_to_anchor": anchor} if anchor else {}
            key = ax.legend([k_net, k_obs], ["network", "Bayesian\nobserver"], fontsize=7, loc=loc,
                            handlelength=1.6, borderaxespad=0.3, labelspacing=0.3, **kw)
            if not _crosses(ax, key, drawn):
                break
        return fig

    # its own folder only: the paper's Fig. 4 is the three-panel figure of `fig4` below
    save_figures({f"{K_BIAS_DISPARITY}/{K_BIAS_DISPARITY}": panel()}, OUT, formats=FORMATS)
    print(f"   -> {OUT / K_BIAS_DISPARITY}/ (+ flat copy)")
    _PANELS[(K_BIAS_DISPARITY, run)] = panel
    return panel


# --------------------------------------------------------------------------- #
# fig4: the paper's Fig. 4, three panels
#   A  relative bias against true disparity at two priors     (bias_disparity; Körding et al. 2007, Fig. 2e)
#   B  hand bias against the signed true disparity, probe trials  (panel B of dokka_probe; Dokka Fig. 1F, 2C)
#   C  hand bias by the cause the network inferred            (decision_bias; Dokka et al. 2019, Fig. 4C,
#                                                              Rideaux et al. 2021, Fig. 2D-E)
# Each panel is drawn by the analysis it belongs to, unchanged; an analysis
# already run in this process is not run again. The composed figure goes to
# the run's manuscript folder as fig4_bias_disparity and, with the three
# panels, to bias_three_panels/ of the run.
# --------------------------------------------------------------------------- #
def fig4(run=None):
    run = run or RUN
    pA = _PANELS.get((K_BIAS_DISPARITY, run)) or bias_disparity(run)
    pB_full = (_PANELS.get((K_DOKKA, run)) or dokka_probe(run))["B"]
    xmax_b = PANEL_B_XMAX.get(prior_family(run)[0])
    pB = (lambda ax=None: pB_full(ax, xmax=xmax_b)) if xmax_b else pB_full
    pC = _PANELS.get((K_BIAS, run)) or decision_bias(run)
    figs = {f"{K_FIG4}/{K_FIG4}": manuscript_grid([pA, pB, pC], ncols=3),
            f"{K_FIG4}/A_relative_bias_two_priors": pA(),
            f"{K_FIG4}/B_bias_probe_trials": pB(),
            f"{K_FIG4}/C_bias_by_inferred_cause": pC()}
    save_all(figs, K_FIG4)


ANALYSES = {"decision-bias": decision_bias, "dokka-probe": dokka_probe,
            "bias-disparity": bias_disparity,
            "fig4": fig4}          # fig4 last: with `all` it reuses the three analyses run before it
EXTRA = {"prior-configs": prior_configs}   # run by name only, never by `all`


def main(argv=None):
    import argparse
    import warnings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analyses", nargs="*", default=["all"],
                    help="fig4, bias-disparity, decision-bias, dokka-probe, prior-configs or all")
    ap.add_argument("--run", default="flagship", help="the network analysed (default flagship)")
    args = ap.parse_args(argv)
    unknown = [w for w in args.analyses if w != "all" and w not in ANALYSES and w not in EXTRA]
    if unknown:
        ap.error(f"unknown analysis: {', '.join(unknown)}")
    configure(args.run)
    apply_style()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    which = list(ANALYSES) if args.analyses == ["all"] else args.analyses
    print(f"run {RUN} -> {OUT}")
    for w in which:
        print(f"=== {w} ===")
        (ANALYSES.get(w) or EXTRA[w])()


if __name__ == "__main__":
    main()
