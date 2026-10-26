"""Stage 7 -- the eye-position sweep: how much does the reference-frame
transformation change the game?

    python scripts/07_eye_sweep.py --eye-sd 1 3 9 18 27 --seeds 0 1 2 --control pcommon1
    python scripts/07_eye_sweep.py --replot                      # redraw from the stored results

Trains one network per eye-position spread, changing ONLY `eye_sigma_sq`
(the variance of the eye position across trials; the flagship's is 81,
sd 9 deg) -- same visual and proprioceptive inputs, same eye-signal noise,
same encoders, same architecture, same seed -- and asks how the causal
behaviour the network implements depends on how much of a reference-frame
transformation vision needs. With the eyes nearly fixed (sd 1 deg) the
retinal and the body frame coincide and the eye signal is irrelevant; with
the eyes moving over sd 27 deg every visual measurement has to be shifted by
a large, noisily known amount before it can be compared with the hand.

Two things change with the lever, and the analysis keeps them apart. The
observer changes: shrinking the eye estimate toward its prior gives
var_eye = 1 / (1/sig2_eye + 1/eye_sigma_sq), so the body-frame visual
variance sig2_vis + var_eye is 1.1 deg^2 at sd 1 deg and 5.1 deg^2 at sd 27
deg, the optimal visual weight falls and the posterior's transition moves.
Every statistic here is therefore reported against the network's OWN
observer (slopes on the posterior, midpoint against the analytical midpoint,
fusion and segregation of both), so that "the network does what Bayes says
for its world" and "the world got harder" are separate readings. The network
changes: the reference-frame shift it has to learn, measured by the RF-shift
gain of the multisensory units (0 = spatial code, 1 = retinal), probed at a
FIXED set of eye positions (+-9 deg, the flagship's) for every network so
the gains are comparable, and at each network's own +-sd.

The same script sweeps the MEAN eye position instead (`--param eye_mu
--levels 0 10 20 30 40`, the spread kept at the flagship's sd 9 deg): the
eyes are deviated from straight ahead on average, so the retinal visual
input is shifted by -eye_mu while the body-frame task is unchanged, and the
observer barely notices (k = 0.95, the prior mean enters the eye estimate
with weight 0.05). The unit classes of such a network are measured with
the eye held at its eye_mu (`analysis.congruency(..., eye=eye_mu)`), i.e.
inside its trained range, and the RF shift is probed both around its own
mean (eye_mu +- 9 deg) and around straight ahead (0 +- 9 deg), the latter
an extrapolation for the shifted networks.

Writes (eye_mu: results/eye_mu_sweep/ and figS_eye_mu_sweep):
    results/eye_sweep/sweep.json    per-level statistics (rows) and across-seed means (aggregated)
    results/eye_sweep/curves.npz    per-level weight-vs-disparity curves (+ per-trial arrays, seed 0)
    results/eye_sweep/figures/      the composed figure and its panels
    results/manuscript/figS_eye_sweep/   the manuscript copy (standard panel)
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi import analysis
from cmsi.data import make_dataset, subset
from cmsi.models import hidden_activations, predict, train
from cmsi.utils import load_config, load_json, run_dir, save_json, tweak
from cmsi.utils.paths import CONFIGS, RESULTS
from cmsi.viz.manuscript import FIG_S_EYE_MU, FIG_S_EYE_SD

GRID = np.array([-30, -24, -19, -15, -12, -9, -6, -3.5, -1.5,
                 1.5, 3.5, 6, 9, 12, 15, 19, 24, 30], float)
MIN_COUNT, MAX_SE = 25, 0.05
PROBE_EYE = np.linspace(-9.0, 9.0, 5)     # the fixed RF-shift probe: the flagship's +-sd around straight ahead
NEAR, FAR = 2.0, 15.0                     # fusion: |d| <= NEAR; segregation: |d| >= FAR
PARAMS = {"eye_sd": {"out": "eye_sweep", "paper": FIG_S_EYE_SD, "label": "eye-position sd (deg)",
                     "legend": "eye sd {:g}°", "default": [1, 3, 9, 18, 27]},
          "eye_mu": {"out": "eye_mu_sweep", "paper": FIG_S_EYE_MU, "label": "mean eye position (deg)",
                     "legend": "eye mean {:g}°", "default": [0, 10, 20, 30, 40]}}


def level_config(cfg, param, value, **more):
    """The flagship with one generative lever changed."""
    if param == "eye_sd":
        return tweak(cfg, eye_sigma_sq=float(value) ** 2, **more)
    if param == "eye_mu":
        return tweak(cfg, eye_mu=float(value), **more)
    raise KeyError(param)


def _on_grid(grid, centres, values):
    out = np.full(len(grid), np.nan)
    for c, v in zip(centres, values, strict=True):
        out[int(np.argmin(np.abs(np.asarray(grid) - c)))] = v
    return out


def fusion_segregation(abs_disparity, w, near=NEAR, far=FAR):
    ad = np.abs(abs_disparity)
    ok = np.isfinite(w)
    return (float(np.nanmean(w[ok & (ad <= near)])), float(1.0 - np.nanmean(w[ok & (ad >= far)])))


def one_level(cfg, param, value, sig_out, n_trials, epochs, seed):
    """Train one network at one level of the lever and return its statistics,
    its curves and the arrays the figure needs."""
    c = level_config(cfg, param, value, n_trials=int(n_trials), epochs=int(epochs), seed=int(seed))
    gen = c["generative"]
    eye_mu, eye_sd = float(gen["eye_mu"]), float(np.sqrt(gen["eye_sigma_sq"]))
    d_full = make_dataset(c)
    model, history, splits = train(d_full, c, verbose=False)
    d = subset(d_full, splits["test"])
    names = d_full["target_names"]
    iv, ip, ivv, ivp = (names.index(k) for k in ("mu_vis", "mu_prop", "var_vis", "var_prop"))
    pred = predict(model, d["X"])
    target = np.stack([d[k] for k in names], axis=1)
    post = d["post_c1"]
    dv, dp = d["fused_mu"] - d["seg_vis_mu"], d["fused_mu"] - d["seg_prop_mu"]
    so = np.asarray(sig_out) if sig_out is not None else analysis.sigma_out(pred, target)

    # the implied weight of each channel on every trial: the hybrid read
    w_vis, _, _ = analysis.hybrid_weight(pred[:, ivv], d["fused_var"], d["seg_vis_var"], dv,
                                         pred[:, iv], d["seg_vis_mu"], so[iv], so[ivv])
    w_prop, _, _ = analysis.hybrid_weight(pred[:, ivp], d["fused_var"], d["seg_prop_var"], dp,
                                          pred[:, ip], d["seg_prop_mu"], so[ip], so[ivp])
    cmp_v, cmp_p = analysis.compare(post, w_vis), analysis.compare(post, w_prop)
    wr_v, wr_p = analysis.weight_regression(w_vis, post), analysis.weight_regression(w_prop, post)
    reg_v = analysis.position_regression(pred[:, iv], d["seg_vis_mu"], d["fused_mu"], post)
    reg_p = analysis.position_regression(pred[:, ip], d["seg_prop_mu"], d["fused_mu"], post)
    net_mid, net_k = analysis.transition_fit(np.abs(d["disparity"]), w_vis)
    opt_mid, opt_k = analysis.transition_fit(np.abs(d["disparity"]), post)
    vs = analysis.variance_signature(pred[:, ivv], post, d["fused_mu"], d["fused_var"],
                                     d["seg_vis_mu"], d["seg_vis_var"])
    rel = analysis.reliability_within_disparity(w_vis, post, np.abs(d["disparity"]))
    mc = analysis.model_comparison(pred[:, iv], post, d["fused_mu"], d["seg_vis_mu"])
    f_net, s_net = fusion_segregation(d["disparity"], w_vis)
    f_opt, s_opt = fusion_segregation(d["disparity"], post)
    acc = analysis.accuracy(pred, target, names)

    # what the layers carry: the posterior, the eye position, the spatial
    # (body-frame) visual position and the retinal one
    acts = hidden_activations(model, d["X"])
    acfg = c["analysis"]
    alpha, ts = acfg.get("ridge_alpha", 1.0), acfg.get("decoder_test_size", 0.25)
    dec_post = analysis.summarise(analysis.decode_by_layer(acts, post, alpha, ts))
    dec = {}
    for key, tgt in (("eye", d["eye"]), ("s_vis", d["s_vis"]), ("retinal", d["retinal"])):
        r = analysis.decode_by_layer(acts, tgt, alpha, ts)
        dec[key] = analysis.summarise(r)

    # the multisensory units: classes, and the RF shift at the fixed probe
    # and at the network's own +-sd
    cong = analysis.congruency(model, d_full["encoders"], c, eye=eye_mu)
    rf_fixed = analysis.rf_shift(model, d_full["encoders"], c, eye_positions=PROBE_EYE)
    rf_own = analysis.rf_shift(model, d_full["encoders"], c, eye_positions=eye_mu + np.linspace(-eye_sd, eye_sd, 5))

    # the observer's side of the lever
    sig2_eye = float(np.mean(gen["sigma2_eye_range"]))
    var_eye = 1.0 / (1.0 / sig2_eye + 1.0 / gen["eye_sigma_sq"])
    var_vis_body = float(np.mean(gen["sigma2_vis_range"])) + var_eye

    stats = {
        "param": param, "level": float(value),
        "eye_sd": eye_sd, "eye_sigma_sq": float(gen["eye_sigma_sq"]), "eye_mu": eye_mu, "seed": int(seed),
        "n_test": int(len(pred)), "best_val": float(history["best_val"]),
        "best_epoch": int(history.get("best_epoch", -1)) if isinstance(history, dict) else -1,
        "rejection_rate": float(np.asarray(d_full.get("rejection_rate", 0.0)).ravel()[0]),
        "realized_c1": float((d_full["C"] == 1).mean()),
        "var_eye_posterior": float(var_eye), "var_vis_body": var_vis_body,
        "eye_gain_k": float(var_eye / sig2_eye),
        "mean_post": float(post.mean()),
        "weight_vs_post_slope_vis": cmp_v["slope"], "weight_vs_post_r2_vis": cmp_v["r2"],
        "weight_vs_post_slope_prop": cmp_p["slope"], "weight_vs_post_r2_prop": cmp_p["r2"],
        "weight_reg_slope_vis": wr_v["slope"], "weight_reg_ci_vis": wr_v["slope_ci95"],
        "weight_reg_slope_prop": wr_p["slope"], "weight_reg_ci_prop": wr_p["slope_ci95"],
        "posreg_slope_vis": reg_v["slope"], "posreg_ci_vis": reg_v["slope_ci95"],
        "posreg_slope_prop": reg_p["slope"], "posreg_ci_prop": reg_p["slope_ci95"],
        "midpoint_net": float(net_mid), "midpoint_opt": float(opt_mid),
        "sharpness_net": float(net_k), "sharpness_opt": float(opt_k),
        "fusion_net": f_net, "segregation_net": s_net, "fusion_opt": f_opt, "segregation_opt": s_opt,
        "hump_net": float(vs.get("hump_net", np.nan)), "hump_opt": float(vs.get("hump_analytical", np.nan)),
        "reliability_slope": float(rel["combined_slope"]), "reliability_se": float(rel["combined_se"]),
        "best_strategy": mc["best"], "rmse_averaging": mc["overall"]["averaging"],
        "rmse_selection": mc["overall"]["selection"], "rmse_fixed": mc["overall"]["fixed"],
        "rmse_mu_vis": acc[iv]["rmse"], "rmse_mu_prop": acc[ip]["rmse"],
        "accuracy_r2": [r["r2"] for r in acc],
        "decode_post_sil": dec_post.get("layer0"), "decode_post_msl": dec_post.get("layer1"),
        **{f"decode_{k}_{lay}": dec[k].get(layer) for k in dec
           for lay, layer in (("sil", "layer0"), ("msl", "layer1"))},
        "n_congruent": int(cong["n_congruent"]), "n_opposite": int(cong["n_opposite"]),
        "n_mixed": int(cong["n_mixed"]), "n_untuned": int(cong["n_untuned"]),
        "rf_gain_median_fixed": float(rf_fixed["median_shift_gain"]),
        "rf_gain_median_own": float(rf_own["median_shift_gain"]),
        "rf_gain_mean_abs_fixed": float(np.nanmean(np.abs(rf_fixed["shift_gain"]))),
        "rf_gainfield_median_abs_fixed": float(np.nanmedian(np.abs(rf_fixed["gain_field"]))),
    }
    c_net, m_net, _, se_net = analysis.binned_weight(d["disparity"], w_vis, GRID,
                                                     min_count=MIN_COUNT, max_se=MAX_SE)
    c_opt, m_opt, _ = analysis.mean_by_bin(d["disparity"], post, GRID)
    curves = {"centres": GRID, "w_net": _on_grid(GRID, c_net, m_net), "se_net": _on_grid(GRID, c_net, se_net),
              "centres_opt": np.asarray(c_opt), "w_opt": np.asarray(m_opt),
              "t_disparity": d["disparity"], "t_post": post, "t_w_vis": w_vis,
              "rf_gain_fixed": rf_fixed["shift_gain"], "congruency_index": cong["index"]}
    return stats, curves


def twin_level(cfg, param, value, n_trials, epochs, seed):
    """The always-fuse twin of one level: the same trials (same seed), fused
    targets; its unit classes and RF-shift gain, so the causal-minus-twin
    excess of opposite units (B8) can be read at every level."""
    c = level_config(cfg, param, value, n_trials=int(n_trials), epochs=int(epochs), seed=int(seed), head="fused")
    d_full = make_dataset(c)
    model, history, _ = train(d_full, c, verbose=False)
    cong = analysis.congruency(model, d_full["encoders"], c, eye=float(c["generative"]["eye_mu"]))
    rf_fixed = analysis.rf_shift(model, d_full["encoders"], c, eye_positions=PROBE_EYE)
    return {"twin_n_congruent": int(cong["n_congruent"]), "twin_n_opposite": int(cong["n_opposite"]),
            "twin_n_mixed": int(cong["n_mixed"]), "twin_n_untuned": int(cong["n_untuned"]),
            "twin_rf_gain_median_fixed": float(rf_fixed["median_shift_gain"]),
            "twin_best_val": float(history["best_val"])}


SCALARS = ("rejection_rate", "realized_c1", "var_vis_body", "eye_gain_k", "mean_post",
           "weight_vs_post_slope_vis", "weight_vs_post_r2_vis", "weight_vs_post_slope_prop",
           "weight_reg_slope_vis", "weight_reg_slope_prop", "posreg_slope_vis", "posreg_slope_prop",
           "midpoint_net", "midpoint_opt", "sharpness_net", "sharpness_opt",
           "fusion_net", "segregation_net", "fusion_opt", "segregation_opt", "hump_net", "hump_opt",
           "reliability_slope", "rmse_averaging", "rmse_selection", "rmse_fixed", "rmse_mu_vis", "rmse_mu_prop",
           "decode_post_sil", "decode_post_msl", "decode_eye_sil", "decode_eye_msl",
           "decode_s_vis_sil", "decode_s_vis_msl", "decode_retinal_sil", "decode_retinal_msl",
           "n_congruent", "n_opposite", "n_mixed", "n_untuned",
           "rf_gain_median_fixed", "rf_gain_median_own", "rf_gain_mean_abs_fixed", "rf_gainfield_median_abs_fixed",
           "best_val", "best_epoch",
           "twin_n_congruent", "twin_n_opposite", "twin_n_mixed", "twin_n_untuned", "twin_rf_gain_median_fixed")


def _level(r):
    """The lever's value of a row (older sweep files carry it as eye_sd only)."""
    return float(r.get("level", r.get("eye_sd")))


def aggregate(rows):
    """One row per level: mean, sd and SEM over seeds of every scalar."""
    by = {}
    for r in rows:
        by.setdefault(_level(r), []).append(r)
    out = []
    for lv in sorted(by):
        g = by[lv]
        agg = {"level": lv, "param": g[0].get("param", "eye_sd"), "eye_sd": lv, "n_seeds": len(g),
               "seeds": sorted(r["seed"] for r in g)}
        for k in SCALARS:
            v = np.array([r[k] for r in g if r.get(k) is not None], float)
            v = v[np.isfinite(v)]
            agg[k] = float(v.mean()) if v.size else np.nan
            agg[k + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            agg[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        strategies = {r["best_strategy"] for r in g}
        agg["best_strategy"] = strategies.pop() if len(strategies) == 1 else "mixed:" + "/".join(sorted(strategies))
        out.append(agg)
    return out


def print_table(agg):
    print(f"\n{agg[0].get('param', 'eye_sd'):6s}  | w-on-post slope vis / prop | posreg vis / prop | midpoint net / opt | "
          "fusion net / opt | segreg net / opt | RF gain fixed / own | con / opp / mix | rmse mu_vis | strategy")
    for a in agg:
        print(f"  {_level(a):4.0f}  | {a['weight_vs_post_slope_vis']:.3f} / {a['weight_vs_post_slope_prop']:.3f}"
              f"            | {a['posreg_slope_vis']:.3f} / {a['posreg_slope_prop']:.3f}     "
              f"| {a['midpoint_net']:5.2f} / {a['midpoint_opt']:5.2f}      "
              f"| {a['fusion_net']:.3f} / {a['fusion_opt']:.3f}   | {a['segregation_net']:.3f} / {a['segregation_opt']:.3f}  "
              f"| {a['rf_gain_median_fixed']:+.3f} / {a['rf_gain_median_own']:+.3f}     "
              f"| {a['n_congruent']:4.1f} / {a['n_opposite']:4.1f} / {a['n_mixed']:4.1f} "
              + (f"(twin {a['twin_n_congruent']:4.1f} / {a['twin_n_opposite']:4.1f} / {a['twin_n_mixed']:4.1f}) "
                 if np.isfinite(a.get("twin_n_opposite", np.nan)) else "")
              + f"| {a['rmse_mu_vis']:.3f}   | {a['best_strategy']}")


def _write(out, args, rows, curves, agg):
    save_json({"config": args.config, "param": args.param, "levels": args.levels, "seeds": args.seeds,
               "n_trials": args.n, "epochs": args.epochs, "sigma_out_source": args.control,
               "probe_eye": PROBE_EYE.tolist(), "rows": rows, "aggregated": agg}, out / "sweep.json")
    np.savez_compressed(out / "curves.npz", **curves)


# --------------------------------------------------------------------------- #
# the figure
def draw(out, agg, curves, rows=None, param="eye_sd"):
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from cmsi.viz import apply_style, save_figures
    from cmsi.viz.manuscript import (PANEL_FONT, PANEL_LW, PANEL_MS, finish_panel, manuscript_dir,
                                     manuscript_grid, panel_axes)
    from cmsi.viz.style import COLORS
    apply_style()
    spec = PARAMS[param]
    OUT_, PAPER_ = spec["out"], spec["paper"]
    agg = sorted(agg, key=_level)
    sds = [_level(a) for a in agg]
    x = np.arange(len(sds))
    norm = Normalize(vmin=-0.5, vmax=len(sds) - 0.5)
    cmap = plt.get_cmap("viridis")
    col = {sd: cmap(norm(i)) for i, sd in enumerate(sds)}
    n_seeds = max(a["n_seeds"] for a in agg)
    label = f"{n_seeds} seeds" if n_seeds > 1 else "1 seed"

    def xaxis(ax, xlabel=spec["label"]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sd:g}" for sd in sds])
        ax.set_xlim(-0.5, len(sds) - 0.5)
        ax.set_xlabel(xlabel, fontsize=PANEL_FONT["label"])

    def series(ax, key, color, lab, marker="o", ls="-", offset=0.0):
        y = np.array([a[key] for a in agg], float)
        e = np.array([a.get(key + "_sem", 0.0) for a in agg], float)
        ax.errorbar(x + offset, y, yerr=e, fmt=marker, ls=ls, ms=PANEL_MS, lw=PANEL_LW, color=color,
                    capsize=2, label=lab)

    def pA(ax=None):
        ax = panel_axes(ax)
        for sd in sds:
            cn, wn, sn = (curves.get(f"sd{sd:g}_{k}") for k in ("centres", "w_net", "se_net"))
            co, wo = curves.get(f"sd{sd:g}_centres_opt"), curves.get(f"sd{sd:g}_w_opt")
            if co is not None:
                keep = np.abs(co) <= 30
                ax.plot(co[keep], wo[keep], "--", lw=0.9, color=col[sd], alpha=0.6)
            if cn is not None:
                ok = np.isfinite(wn)
                ax.errorbar(cn[ok], wn[ok], yerr=1.96 * sn[ok], fmt="none", ecolor=col[sd], elinewidth=0.7,
                            alpha=0.5)
                ax.plot(cn[ok], wn[ok], "o-", ms=PANEL_MS - 0.8, lw=PANEL_LW, color=col[sd],
                        label=spec["legend"].format(sd))
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        ax.set_xlim(-31, 31)
        ax.set_ylim(-0.06, 1.38)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fig = finish_panel(ax, "body-frame disparity (deg)", "implied fusion weight",
                           "weight vs disparity (seed 0)", None)
        ax.text(0.98, 0.42, "dashed:\nobserver p(C=1|x)", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="grey")
        ax.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.6,
                  borderaxespad=0.2, handletextpad=0.4)
        return fig

    def pB(ax=None):
        ax = panel_axes(ax)
        series(ax, "weight_vs_post_slope_vis", COLORS["visual"], "weight on p(C=1|x), vis")
        series(ax, "weight_vs_post_slope_prop", COLORS["prop"], "weight on p(C=1|x), hand", marker="s")
        series(ax, "posreg_slope_vis", COLORS["visual"], "position regr., vis", marker="^", ls="--")
        series(ax, "posreg_slope_prop", COLORS["prop"], "position regr., hand", marker="v", ls="--")
        ax.axhline(1.0, lw=0.6, ls=":", color="grey")
        ax.set_ylim(0.4, 1.3)
        xaxis(ax)
        fig = finish_panel(ax, spec["label"], "slope (Bayes-optimal: 1)",
                           f"Bayesian weighting ({label})", None)
        ax.legend(fontsize=6.5, loc="lower left", ncol=2, handlelength=1.6, columnspacing=0.5,
                  borderaxespad=0.2)
        return fig

    def pC(ax=None):
        ax = panel_axes(ax)
        series(ax, "midpoint_opt", "#111418", "observer", marker="s", ls="--")
        series(ax, "midpoint_net", COLORS["network"], "network")
        xaxis(ax)
        vals = np.array([a[k] for a in agg for k in ("midpoint_net", "midpoint_opt")], float)
        vals = vals[np.isfinite(vals)]
        lo, hi = (vals.min(), vals.max()) if vals.size else (0.0, 10.0)
        ax.set_ylim(max(0.0, lo - 1.5), hi + 2.0)
        fig = finish_panel(ax, spec["label"], "transition midpoint (deg)",
                           f"transition midpoint ({label})", "upper left")
        return fig

    def pD(ax=None):
        ax = panel_axes(ax)
        series(ax, "rf_gain_median_fixed", COLORS["network"], "probe 0 ± 9° (fixed)")
        series(ax, "rf_gain_median_own", COLORS["network"],
               "probe own mean ± sd" if param == "eye_mu" else "probe ±own sd", marker="s", ls="--")
        ax.axhline(0.0, lw=0.6, ls=":", color="grey")
        ax.axhline(1.0, lw=0.6, ls=":", color="grey")
        ax.text(len(sds) - 0.55, 1.03, "retinal code", ha="right", va="bottom", fontsize=7, color="grey")
        ax.text(len(sds) - 0.55, 0.03, "spatial code", ha="right", va="bottom", fontsize=7, color="grey")
        xaxis(ax)
        lo = np.nanmin([a["rf_gain_median_fixed"] for a in agg] + [a["rf_gain_median_own"] for a in agg] + [-0.3])
        ax.set_ylim(min(-0.3, lo - 0.1), 1.25)
        fig = finish_panel(ax, spec["label"], "median RF-shift gain, MSL",
                           f"RF transformation ({label})", "center left")
        return fig

    def pE(ax=None):
        ax = panel_axes(ax)
        from_common = {"congruent": "#1f77b4", "opposite": "#d62728", "mixed": "#ccbb44"}
        shown = {"congruent": "congruent", "opposite": "opposite", "mixed": "intermediate"}
        for key, c in (("n_congruent", "congruent"), ("n_opposite", "opposite"), ("n_mixed", "mixed")):
            series(ax, key, from_common[c], shown[c])
        if all(np.isfinite(a.get("twin_n_opposite", np.nan)) for a in agg):
            for key, c in (("twin_n_congruent", "congruent"), ("twin_n_opposite", "opposite"),
                           ("twin_n_mixed", "mixed")):
                series(ax, key, from_common[c], None, marker="s", ls="--")
            ax.plot([], [], "s--", color="#555555", ms=PANEL_MS, lw=PANEL_LW, label="always-fuse twin")
        xaxis(ax)
        ax.set_ylim(0, 76)
        ax.set_yticks([0, 20, 40, 60])
        fig = finish_panel(ax, spec["label"], "units of 64", f"MSL unit classes ({label})", None)
        ax.legend(fontsize=6.5, loc="upper center", ncol=2, handlelength=1.6, columnspacing=0.6,
                  borderaxespad=0.2)
        return fig

    def pF(ax=None):
        ax = panel_axes(ax)
        series(ax, "fusion_opt", "#111418", "fusion, observer", marker="s", ls="--")
        series(ax, "fusion_net", COLORS["network"], "fusion, network")
        series(ax, "segregation_opt", "#8c8c8c", "segr., observer", marker="s", ls="--")
        series(ax, "segregation_net", COLORS["hybrid"], "segr., network", marker="^")
        xaxis(ax)
        ax.set_ylim(0.2, 1.15)
        fig = finish_panel(ax, spec["label"], "fusion / segregation",
                           f"fusion and segregation ({label})", None)
        ax.legend(fontsize=6.5, loc="lower left", ncol=2, handlelength=1.6, columnspacing=0.5,
                  borderaxespad=0.2)
        return fig

    draws = [pA, pB, pC, pD, pE, pF]
    figs = {f"{OUT_}/{OUT_}": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_weight_curves", "B_slopes", "C_midpoint", "D_rf_shift", "E_unit_classes",
                        "F_fusion_segregation"), draws):
        figs[f"{OUT_}/{key}"] = fn()
    if out.name == OUT_:
        save_figures({f"{PAPER_}/{PAPER_}": figs[f"{OUT_}/{OUT_}"]}, manuscript_dir(), formats=("png", "tif", "svg"),
                     close=False)
    save_figures(figs, out / "figures", formats=("png", "tif", "svg"))
    print(f"wrote {out / 'figures' / OUT_}/" + (f" and {manuscript_dir() / PAPER_}/ (+ flat copies)" if out.name == OUT_ else ""))


def replot(out, param):
    sweep = load_json(out / "sweep.json")
    curves = dict(np.load(out / "curves.npz"))
    param = sweep.get("param", param)
    draw(out, sweep["aggregated"], curves, sweep["rows"], param=param)
    print_table(sweep["aggregated"])


def main(args):
    cfg = load_config(args.config)
    spec = PARAMS[args.param]
    if args.levels is None:
        args.levels = spec["default"]
    out = RESULTS / (args.out or spec["out"])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if args.replot:
        replot(out, args.param)
        return
    sig_out = None
    if args.control:
        cm = load_json(run_dir(args.control, create=False) / "metrics.json")
        if "residual_std" in cm:
            sig_out = np.asarray(cm["residual_std"])
            print(f"sigma_out from '{args.control}': " + ", ".join(f"{s:.3f}" for s in sig_out))
    rows, curves = [], {}
    if args.resume and (out / "sweep.json").exists():
        prev = load_json(out / "sweep.json")
        rows = [r for r in prev["rows"] if _level(r) in args.levels and r["seed"] in args.seeds]
        curves = dict(np.load(out / "curves.npz")) if (out / "curves.npz").exists() else {}
        print(f"resuming: {len(rows)} network(s) already done")
    done = {(_level(r), r["seed"]) for r in rows}
    n_nets = len(args.levels) * len(args.seeds)
    print(f"{args.param}: {len(args.levels)} levels x {len(args.seeds)} seed(s) = {n_nets} networks\n")
    for seed in args.seeds:
        for sd in args.levels:
            if (float(sd), int(seed)) in done:
                continue
            s, c = one_level(cfg, args.param, sd, sig_out, args.n, args.epochs, seed)
            rows.append(s)
            for k, v in c.items():
                if k.startswith("t_") or k in ("rf_gain_fixed", "congruency_index"):
                    curves[f"s{seed}_sd{sd:g}_{k}"] = v
                if seed == args.seeds[0]:
                    curves[f"sd{sd:g}_{k}"] = v
            print(f"  seed {seed}  {args.param} {sd:4.0f}: w-on-post slope {s['weight_vs_post_slope_vis']:.3f} | "
                  f"posreg {s['posreg_slope_vis']:.3f} | midpoint {s['midpoint_net']:5.2f} vs {s['midpoint_opt']:5.2f} "
                  f"| RF gain {s['rf_gain_median_fixed']:+.3f} | units {s['n_congruent']}/{s['n_opposite']}/{s['n_mixed']} "
                  f"| rejected {100 * s['rejection_rate']:.1f}%", flush=True)
            _write(out, args, rows, curves, aggregate(rows))
    if args.twin:
        todo = [r for r in rows if r.get("twin_n_opposite") is None]
        print(f"\nalways-fuse twins: {len(todo)} to train")
        for r in todo:
            r.update(twin_level(cfg, args.param, _level(r), args.n, args.epochs, r["seed"]))
            print(f"  seed {r['seed']}  {args.param} {_level(r):4.0f}: twin units "
                  f"{r['twin_n_congruent']}/{r['twin_n_opposite']}/{r['twin_n_mixed']} "
                  f"(causal {r['n_congruent']}/{r['n_opposite']}/{r['n_mixed']}); "
                  f"twin RF gain {r['twin_rf_gain_median_fixed']:+.3f}", flush=True)
            _write(out, args, rows, curves, aggregate(rows))
    agg = aggregate(rows)
    _write(out, args, rows, curves, agg)
    print(f"wrote {out / 'sweep.json'} and {out / 'curves.npz'}")
    draw(out, agg, curves, rows, param=args.param)
    print_table(agg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(CONFIGS / "flagship.yaml"), help="base config (default configs/flagship.yaml)")
    p.add_argument("--param", choices=sorted(PARAMS), default="eye_sd",
                   help="the lever: eye_sd (the spread of the eye position; default) or eye_mu (its mean)")
    p.add_argument("--levels", "--eye-sd", dest="levels", type=float, nargs="+", default=None,
                   help="the lever's values in deg (eye_sd: 1 3 9 18 27; eye_mu: 0 10 20 30 40)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--control", default="pcommon1", help="the run whose residuals are sigma_out")
    p.add_argument("--out", default=None, help="output folder under results/ (default eye_sweep)")
    p.add_argument("--resume", action="store_true", help="keep the networks already in sweep.json")
    p.add_argument("--twin", action="store_true",
                   help="also train the always-fuse twin of every network (same trials, fused targets) "
                        "for its unit classes")
    p.add_argument("--replot", action="store_true", help="redraw from the stored results")
    main(p.parse_args())
