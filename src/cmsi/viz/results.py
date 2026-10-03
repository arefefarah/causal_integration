"""Model results: the network against the analytical observer.

These are the figures that answer the research question, so they get their own
folder (results/<run>/figures/model) separate from the input and training checks.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import mean_by_bin, mixture_variance, weight_by_posterior
from cmsi.viz.manuscript import (  # noqa: F401  (re-exported for callers and tests)
    CELL_FULL,
    CELL_SQUARE,
    CELL_WIDE,
    PANEL_FONT,
    PANEL_LW,
    PANEL_MARGIN,
    PANEL_MS,
    PANEL_RECT,
    finish_panel,
    manuscript_grid,
    panel_axes,
    panel_rect,
)
from cmsi.viz.style import COLORS, SIZE, label_panels


def _grid(n, size, ncols=2):
    """n panels in a `ncols`-wide grid at a PLOS-sized figure; unused axes are
    hidden. Returns (fig, flat axes). Four outputs in one row would give each
    panel 1.8 in of the 7.5 in width, too little for readable ticks."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    axes = np.ravel(axes)
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def output_scatter(pred, target, names):
    """Network vs analytical, one panel per output, with R^2 and RMSE."""
    rows = accuracy(pred, target, names)
    fig, axes = _grid(len(names), SIZE["quad"])
    for i, (ax, row) in enumerate(zip(axes, rows, strict=False)):
        y, yhat = target[:, i], pred[:, i]
        # rasterized: the point cloud embeds as one 300-dpi image inside the
        # SVG/PDF instead of ~12,000 vector markers; axes and text stay vector
        ax.scatter(y, yhat, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
        lo, hi = np.percentile(y, [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"])
        ax.set(title=f"{row['output']}   R$^2$ = {row['r2']:.3f}   RMSE = {row['rmse']:.2f}",
               xlabel="analytical", ylabel="network")
    label_panels(axes[:len(names)])
    fig.tight_layout()
    return fig


def error_histograms(pred, target, names):
    fig, axes = _grid(len(names), SIZE["quad_short"])
    for i, (ax, name) in enumerate(zip(axes, names, strict=False)):
        err = pred[:, i] - target[:, i]
        ax.hist(err, bins=60, color=COLORS["network"])
        ax.axvline(0, lw=1, color=COLORS["analytical"])
        ax.set(title=f"{name}   bias = {err.mean():.2f}", xlabel="network - analytical",
               ylabel="trials")
    label_panels(axes[:len(names)])
    fig.tight_layout()
    return fig


def p_common_vs_disparity(disparity, p_common, grid):
    """The Kording-style curve: p(C=1) high near zero disparity, falling away."""
    centres, means, _ = mean_by_bin(disparity, p_common, grid)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot(centres, means, "o-", color=COLORS["analytical"])
    ax.set(xlabel="body-frame disparity (deg)", ylabel="p(C=1)", ylim=(0, 1),
           title="analytical common-cause posterior")
    fig.tight_layout()
    return fig


def fusion_weight_curve(disparity, w_network, p_analytical, grid, w_prop=None):
    """The key panel: the implied weight against the optimal weight p(C=1).

    `w_network` and `w_prop` are the hybrid reads of the visual and the
    proprioceptive channel (analysis.hybrid_weight): the mixture-variance
    root of the channel's variance output where that root is unique
    (Delta^2 <= c, including zero disparity) and the channel's own position
    ratio where the variance has two roots. Both exist on every trial, so
    the curve carries through zero disparity, where the two hypotheses
    coincide. The two channels encode the same weight, so the two curves
    are two readings of one quantity.
    """
    fig, ax = plt.subplots(figsize=SIZE["single"])
    c_opt, m_opt, _ = mean_by_bin(disparity, p_analytical, grid)
    ax.plot(c_opt, m_opt, "--o", color=COLORS["analytical"], label="analytical p(C=1)")

    c_net, m_net, _ = mean_by_bin(disparity, w_network, grid)
    ax.plot(c_net, m_net, "o-", color=COLORS["network"],
            label="implied, vis (var_vis root / mu_vis ratio)")

    if w_prop is not None:
        c_p, m_p, _ = mean_by_bin(disparity, w_prop, grid)
        ax.plot(c_p, m_p, "s-", color=COLORS["prop"],
                label="implied, prop (var_prop root / mu_prop ratio)")

    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="fusion -> segregation transition")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fusion_weight_by_reliability(curves):
    """curves: output of analysis.by_reliability -> {level: (centres, means, n)}."""
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for level, (centres, means, _) in sorted(curves.items()):
        ax.plot(centres, means, "o-", label=f"sigma2_vis ~ {level:g}")
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="shift by different cue reliability")
    ax.legend()
    fig.tight_layout()
    return fig


def weight_vs_posterior(res, output_name="vis"):
    """The implied weight against the analytical posterior, in posterior
    bins, every trial (analysis.weight_by_posterior on the hybrid read).

    A Bayes-optimal model-averaging observer puts every point on the
    identity line. The error bars are 1.96 standard errors of the bin mean.
    """
    x, w, se = res["centres"], res["w"], res["se"]
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot([0, 1], [0, 1], "--", lw=1.1, color=COLORS["analytical"],
            label="Bayes-optimal (w = posterior)")
    ax.errorbar(x, w, yerr=1.96 * np.nan_to_num(se), fmt="o-", ms=4, lw=1.7, capsize=2.5,
                color=COLORS["network"], label=f"network (implied weight, {output_name})")
    ax.set(xlabel="analytical posterior p(C=1|x)",
           ylabel="implied weight on the fused estimate",
           xlim=(0, 1), ylim=(-0.05, 1.05), title="implied weight by posterior, every trial")
    ax.text(0.03, 0.95, f"all {int(res['n'].sum())} trials, {len(x)} bins",
            transform=ax.transAxes, fontsize=8, color="0.35", va="top")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def decoding_bars(scores, title="p(C=1) decodable from"):
    """scores: {layer_name: held-out r2} -- i.e. analysis.summarise output."""
    names = list(scores)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.bar(names, [scores[n] for n in names], color=COLORS["network"])
    ax.set(ylabel="held-out R2", ylim=(0, 1), title=title)
    fig.tight_layout()
    return fig


def decoding_comparison(by_model, title="emergent vs imposed"):
    """by_model: {"causal": {layer: r2}, "fused twin": {layer: r2}}.

    If p(C=1) is decodable from the twin -- which was never asked for it -- the
    latent emerges from the integration task rather than from the objective.
    """
    layers = sorted({k for v in by_model.values() for k in v})
    width = 0.8 / len(by_model)
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=SIZE["single"])
    for i, (label, scores) in enumerate(by_model.items()):
        ax.bar(x + i * width, [scores.get(k, 0) for k in layers], width, label=label)
    ax.set_xticks(x + width * (len(by_model) - 1) / 2, layers)
    ax.set(ylabel="held-out R2 for p(C=1)", ylim=(0, 1), title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def variance_regression_scatter(var_out, fused_var, seg_var, delta, post, reg, output_name,
                                ax=None, title=None):
    """SS7.1 headline in the VARIANCE domain: (var_out - var_seg) against
    (v_opt - var_seg), the optimal mixture's variance reduction, with the
    fit (analysis.variance_regression). Slope 1 = Bayes-optimal; no per-trial
    division anywhere, every trial enters with its natural leverage."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=SIZE["single_tall"])
    x = mixture_variance(post, fused_var, seg_var, delta) - seg_var
    y = var_out - seg_var
    ax.scatter(x, y, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"],
            label="Bayes-optimal (slope 1)")
    ax.plot([lo, hi], [reg["slope"] * lo + reg["intercept"],
                       reg["slope"] * hi + reg["intercept"]],
            lw=1.5, color=COLORS["network"],
            label=f"fit: slope {reg['slope']:.2f} "
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}]")
    ax.set(xlabel="v_opt - var_seg  (deg^2)", ylabel=f"{output_name} output - var_seg  (deg^2)",
           title=f"variance-domain regression: {output_name}" if title is None else title)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    if own:
        fig.tight_layout()
        return fig
    return ax


def weight_regression_scatter(w, post, reg, flags=None, output_name="vis", ax=None,
                              n_show=6000, seed=0):
    """The implied weight (the hybrid read of one channel) against the
    analytical posterior, every trial, with its fit (analysis.weight_regression).

    `flags`, when given, are the hybrid's per-trial flags; the points are
    then coloured by the output each weight came from, and the legend gives
    each source's share of the trials. Trials are subsampled for drawing
    only (n_show)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=SIZE["single_tall"])
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isfinite(w))
    pick = rng.choice(ok, min(n_show, ok.size), replace=False)
    if flags is None:
        ax.scatter(post[pick], w[pick], s=2, alpha=0.15, color=COLORS["hybrid"],
                   rasterized=True)
    else:
        flags = np.asarray(flags)
        for value, (lab, col) in HYBRID_SOURCES.items():
            sel = pick[flags[pick] == value]
            share = np.mean(flags[ok] == value)
            ax.scatter(post[sel], w[sel], s=2, alpha=0.2, color=col, rasterized=True,
                       label=f"{lab}: {100 * share:.0f}%")
    ax.plot([0, 1], [0, 1], "--", lw=1, color=COLORS["analytical"], label="Bayes-optimal (slope 1)")
    ax.plot([0, 1], [reg["intercept"], reg["slope"] + reg["intercept"]], lw=1.5,
            color=COLORS["hybrid"],
            label=f"fit: slope {reg['slope']:.2f} "
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}], "
                  f"intercept {reg['intercept']:+.2f}")
    ax.set(xlabel="analytical posterior p(C=1|x)",
           ylabel=f"implied weight, {output_name} (var root / mu ratio)",
           xlim=(0, 1), ylim=(-0.25, 1.25),
           title=f"implied weight on the posterior, {output_name}")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    if own:
        fig.tight_layout()
        return fig
    return ax


HYBRID_SOURCES = {0: ("variance root (Delta^2 <= c)", COLORS["hybrid"]),
                  1: ("position ratio (Delta^2 > c)", COLORS["visual"]),
                  2: ("variance peak (no root)", "0.5")}


def variance_regression_figure(var_out, fused_var, seg_var, delta, post, reg_var,
                               w, reg_w, output_name="var_vis", flags=None):
    """08_position_regression's counterpart: A the headline regression in the
    VARIANCE domain, B the implied weight (the hybrid read of the same
    channel) against the posterior on every trial, its points coloured by
    the output each trial's weight came from, with its fit."""
    channel = output_name.split("_")[-1] if "_" in output_name else output_name
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    variance_regression_scatter(var_out, fused_var, seg_var, delta, post, reg_var,
                                output_name, ax=axes[0])
    weight_regression_scatter(w, post, reg_w, flags=flags, output_name=channel, ax=axes[1])
    label_panels(axes)
    fig.tight_layout()
    return fig


def hybrid_weight_distribution(w_hybrid_vis, w_hybrid_prop, post, flags_vis=None,
                               flags_prop=None, lim=(-1.0, 2.0)):
    """The implied weight on every trial (the hybrid read), one panel per
    channel, against the analytical posterior's own distribution.

    Nothing is filtered. The read is the variance root where Delta^2 <= c
    and the channel's position ratio where Delta^2 > c, so it can leave
    [0, 1] on the ratio trials; the axis is clipped to `lim` with the
    overflow piled into the edge bins, and the fraction outside is printed.
    With the flags, the share of trials read from the ratio is printed too."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    bins = np.linspace(lim[0], lim[1], 91)
    panels = (("vis", w_hybrid_vis, flags_vis), ("prop", w_hybrid_prop, flags_prop))
    for ax, (channel, w, flags) in zip(axes, panels, strict=False):
        if w is None:
            ax.set_visible(False)
            continue
        w = np.asarray(w, float)
        ok = np.isfinite(w)
        wf = w[ok]
        median, q1, q3 = np.median(wf), np.percentile(wf, 25), np.percentile(wf, 75)
        outside = np.mean((wf < lim[0]) | (wf > lim[1]))
        ax.hist(np.clip(wf, *lim), bins=bins, density=True, color=COLORS["hybrid"],
                alpha=0.75, label=f"network, all {wf.size} trials")
        ax.hist(post, bins=bins, density=True, histtype="step", lw=1.3,
                color=COLORS["analytical"], label="analytical posterior")
        ax.axvline(median, color=COLORS["hybrid"], lw=1.0)
        note = (f"{100 * outside:.1f}% of trials outside [{lim[0]:g}, {lim[1]:g}]"
                f"\n(piled into the edge bins)")
        if flags is not None:
            flags = np.asarray(flags)[ok]
            note += f"\n{100 * np.mean(flags == 1):.0f}% read from the mu_{channel} ratio"
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7.5, va="top",
                color="0.35")
        ax.set(xlabel=f"implied weight, {channel} (var_{channel} root / mu_{channel} ratio)",
               xlim=lim,
               title=f"{channel}: median {median:.2f} (IQR {q1:.2f} to {q3:.2f})")
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("density")
    label_panels(axes)
    fig.tight_layout()
    return fig


def position_regression_scatter(pred_col, seg, fused, post, reg, output_name):
    """SS7.1 headline: (estimate - seg) against w_opt * Delta, with the fit."""
    x = post * (fused - seg)
    y = pred_col - seg
    fig, ax = plt.subplots(figsize=SIZE["single_tall"])
    ax.scatter(x, y, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"],
            label="Bayes-optimal (slope 1)")
    ax.plot([lo, hi], [reg["slope"] * lo + reg["intercept"],
                       reg["slope"] * hi + reg["intercept"]],
            lw=1.5, color=COLORS["network"],
            label=f"fit: slope {reg['slope']:.2f} "
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}]")
    ax.set(xlabel="w_opt * (fused - seg)  (deg)",
           ylabel="network - seg  (deg)",
           title=f"position-domain regression: {output_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def variance_hump(sig, output_name):
    """SS7.3: network Var output vs posterior bin, with the analytical mixture
    variance, its between-component term, and the humpless fixed-weight line."""
    c = np.asarray(sig["centres"])
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot(c, sig["mixture"], "--o", color=COLORS["analytical"],
            label="analytical mixture variance")
    ax.plot(c, sig["net"], "o-", color=COLORS["network"], label="network output")
    ax.plot(c, sig["between"], ":", color=COLORS["analytical"],
            label="between-component term w(1-w)d^2")
    ax.plot(c, sig["fixed"], "-", lw=1, color=COLORS.get("prop", "gray"),
            label="best fixed-weight (no hump)")
    ax.set(xlabel="analytical p(C=1|x)", ylabel="variance (deg^2)",
           title=f"uncertainty vs causal ambiguity: {output_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def model_comparison_curves(mc, output_name):
    """SS7.4: per-posterior-decile implied weight, network vs the five strategies.

    The left panel is the discriminating one: model AVERAGING predicts a smooth
    sigmoid tracking the posterior, model SELECTION a step at p = 0.5. The
    per-bin weight is a least-squares slope, not a mean of signed biases --
    the latter cancels within a bin because the disparity is signed.
    """
    c = np.asarray(mc["bin_centres"])
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    axes[0].plot(c, mc["bin_weight_net"], "ko-", lw=2, label="network", zorder=5)
    for k in ("averaging", "integration", "segregation", "selection", "fixed"):
        axes[0].plot(c, mc["bin_weight"][k], "--", label=k)
    axes[0].set(xlabel="analytical p(C=1|x) (decile centres)",
                ylabel="implied weight on the fused estimate",
                ylim=(-0.15, 1.15),
                title=f"per-decile weight: {output_name}")
    axes[0].legend()
    for k in ("averaging", "integration", "segregation", "selection", "fixed"):
        axes[1].plot(c, mc["bin_rmse"][k], "o-", label=k)
    axes[1].set(xlabel="analytical p(C=1|x) (decile centres)",
                ylabel="RMSE vs network (deg)",
                title=f"which strategy explains the network\n(best: {mc['best']})")
    axes[1].legend()
    label_panels(axes)
    fig.tight_layout()
    return fig


def behavioral_bias(saved):
    """SS7.6: Kording Fig. 2e analog + conditioning on the inferred cause."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    axes[0].plot(saved["bias_centres"], saved["bias_net"], "o-",
                 color=COLORS["network"], label="network")
    if "bias_opt" in saved:
        axes[0].plot(saved["bias_centres_opt"], saved["bias_opt"], "--o",
                     color=COLORS["analytical"], label="Bayes-optimal")
    axes[0].axhline(0, lw=0.5, color="gray")
    axes[0].set(xlabel="body-frame disparity (deg)",
                ylabel="hand-report pull toward vision (deg)",
                title="bias vs disparity (Kording 2e analog)")
    axes[0].legend()
    for label, style in (("common", "o-"), ("separate", "s-")):
        c = saved.get(f"cond_bias_{label}_centres")
        b = saved.get(f"cond_bias_{label}")
        if c is not None and len(c):
            axes[1].plot(c, b, style, label=f"inferred {label} cause")
    axes[1].axhline(0, lw=0.5, color="gray")
    axes[1].set(xlabel="|disparity| (deg)", ylabel="bias (deg)",
                title="conditioned on the network's\ncausal judgment (3b-c)")
    axes[1].legend()
    label_panels(axes)
    fig.tight_layout()
    return fig


def congruency_panels(saved, balance_stats=None):
    """SS7.5: congruency-index distribution + balance-vs-posterior scatter."""
    idx = saved["congruency_index"]
    has_balance = "congruent_opposite_balance" in getattr(saved, "files", saved)
    fig, axes = plt.subplots(1, 2 if has_balance else 1,
                             figsize=SIZE["pair"] if has_balance else SIZE["single"])
    axes = np.atleast_1d(axes)
    axes[0].hist(idx, bins=30, color=COLORS["network"])
    axes[0].axvline(0.5, ls="--", lw=1, color=COLORS["analytical"])
    axes[0].axvline(-0.5, ls="--", lw=1, color=COLORS["analytical"])
    axes[0].set(xlabel="congruency index (corr of vis vs prop tuning)",
                ylabel="MSL units", title="congruent / opposite units")
    if has_balance:
        bal = saved["congruent_opposite_balance"]
        post = saved["post_c1"]
        axes[1].scatter(bal, post, s=3, alpha=0.2, color=COLORS["network"],
                        rasterized=True)
        title = "balance predicts p(C=1|x)"
        if balance_stats:
            title += f"  (corr {balance_stats.get('corr', float('nan')):.2f})"
        axes[1].set(xlabel="congruent - opposite mean activity",
                    ylabel="analytical p(C=1|x)", title=title)
        label_panels(axes)
    fig.tight_layout()
    return fig


def rf_shift_hist(saved):
    """Continuity analysis: distribution of RF shift gains and gain fields."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    g = saved["rf_shift_gain"]
    axes[0].hist(g[np.isfinite(g)], bins=30, color=COLORS["network"])
    axes[0].axvline(0, ls="--", lw=1, color=COLORS["analytical"], label="spatial code")
    axes[0].axvline(1, ls=":", lw=1, color=COLORS["analytical"], label="retinal code")
    axes[0].set(xlabel="RF shift gain (d preferred-spatial / d eye)",
                ylabel="MSL units", title="reference frame of MSL units")
    axes[0].legend()
    gf = saved["rf_gain_field"]
    axes[1].hist(gf[np.isfinite(gf)], bins=30, color=COLORS["network"])
    axes[1].set(xlabel="d peak response / d eye  (gain field slope)",
                ylabel="MSL units", title="eye-position gain fields")
    label_panels(axes)
    fig.tight_layout()
    return fig


def all_figures(pred, d, names, analysis_cfg, w=None, curves=None,
                decoding=None, twin_decoding=None, w_prop=None,
                saved=None, metrics=None):
    """Every model figure -> results/<run>/figures/model.

    `d` should already be restricted to the trials `pred` was computed on
    (use data.subset(d, splits["test"])). `saved` is the analysis.npz mapping,
    `metrics` the metrics.json dict -- both optional; figures that need them
    are skipped when absent.
    """
    metrics = metrics or {}
    target = np.stack([d[k] for k in names], axis=1)
    figs = {
        "01_output_scatter": output_scatter(pred, target, names),
        "02_error_histograms": error_histograms(pred, target, names),
        "03_post_c1_vs_disparity": p_common_vs_disparity(
            d["disparity"], d["post_c1"], analysis_cfg["disparity_grid"]),
    }
    if w is not None:
        figs["04_fusion_weight"] = fusion_weight_curve(
            d["disparity"], w, d["post_c1"], analysis_cfg["disparity_grid"],
            w_prop=w_prop)
    if curves:
        figs["05_fusion_weight_by_reliability"] = fusion_weight_by_reliability(curves)
    if decoding:
        figs["06_decoding"] = decoding_bars(decoding)
    if decoding and twin_decoding:
        figs["07_emergent_vs_imposed"] = decoding_comparison(
            {"causal model": decoding, "always-fuse twin": twin_decoding})

    if "position_regression_vis" in metrics:
        i = names.index("mu_vis")
        figs["08_position_regression"] = position_regression_scatter(
            pred[:, i], d["seg_vis_mu"], d["fused_mu"], d["post_c1"],
            metrics["position_regression_vis"], "mu_vis")
    files = getattr(saved, "files", saved) if saved is not None else ()

    def _saved(key):
        return saved[key] if saved is not None and key in files else None

    # the headline in the variance domain, with the implied weight (the
    # hybrid read, stored by 03_analyze.py) on the posterior beside it
    if ("variance_regression_vis" in metrics and "weight_regression_vis" in metrics
            and w is not None):
        i = names.index("var_vis")
        figs["08v_variance_regression"] = variance_regression_figure(
            pred[:, i], d["fused_var"], d["seg_vis_var"], d["fused_mu"] - d["seg_vis_mu"],
            d["post_c1"], metrics["variance_regression_vis"], w,
            metrics["weight_regression_vis"], "var_vis", flags=_saved("hybrid_flags_vis"))
    if "variance_signature_vis" in metrics:
        figs["09_variance_hump_vis"] = variance_hump(
            metrics["variance_signature_vis"], "var_vis")
        figs["10_variance_hump_prop"] = variance_hump(
            metrics["variance_signature_prop"], "var_prop")
    if "model_comparison_vis" in metrics:
        figs["11_model_comparison"] = model_comparison_curves(
            metrics["model_comparison_vis"], "mu_vis")
    if saved is not None and "bias_centres" in files:
        figs["12_behavioral_bias"] = behavioral_bias(saved)
    if saved is not None and "congruency_index" in files:
        figs["13_congruency"] = congruency_panels(
            saved, metrics.get("balance_vs_post"))
    if saved is not None and "rf_shift_gain" in files:
        figs["14_rf_shifts"] = rf_shift_hist(saved)
    if w is not None:
        # the implied weight in posterior bins, every trial, and its
        # distribution on every trial against the posterior's
        figs["15_weight_vs_posterior"] = weight_vs_posterior(
            weight_by_posterior(w, d["post_c1"]), "vis")
        figs["16_weight_distribution"] = hybrid_weight_distribution(
            w, w_prop, d["post_c1"], flags_vis=_saved("hybrid_flags_vis"),
            flags_prop=_saved("hybrid_flags_prop"))
    return figs


# --------------------------------------------------------------------------- #
# Manuscript panels, on the standard defined in cmsi.viz.manuscript
# --------------------------------------------------------------------------- #
# The constants are re-exported here so callers and tests can keep reading
# them from `results`; the definitions live in one place, `viz/manuscript.py`.
_panel_axes = panel_axes
_finish_panel = finish_panel


def panel_fusion_weight(disparity, w_network, p_analytical, grid, w_prop=None,
                        ax=None):
    """04_fusion_weight as a manuscript panel."""
    ax = _panel_axes(ax)
    c_opt, m_opt, _ = mean_by_bin(disparity, p_analytical, grid)
    ax.plot(c_opt, m_opt, "--o", lw=PANEL_LW, ms=PANEL_MS,
            color=COLORS["analytical"], label="analytical p(C=1)")
    c_net, m_net, _ = mean_by_bin(disparity, w_network, grid)
    ax.plot(c_net, m_net, "o-", lw=PANEL_LW, ms=PANEL_MS,
            color=COLORS["network"], label="implied weight, vis")
    if w_prop is not None:
        c_p, m_p, _ = mean_by_bin(disparity, w_prop, grid)
        ax.plot(c_p, m_p, "s-", lw=PANEL_LW, ms=PANEL_MS,
                color=COLORS["prop"], label="implied weight, prop")
    # headroom for the legend: at 2.5 in the three-entry legend is two-thirds
    # of the axes width, so it sits ABOVE the curve's peak rather than on it
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return _finish_panel(ax, "body-frame disparity (deg)",
                         "weight on fused estimate",
                         "fusion-segregation transition", "upper right")


def panel_position_regression(pred_col, seg, fused, post, reg, ax=None):
    """08_position_regression as a manuscript panel."""
    ax = _panel_axes(ax)
    x = post * (fused - seg)
    y = pred_col - seg
    ax.scatter(x, y, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"],
            label="Bayes-optimal (slope 1)")
    ax.plot([lo, hi], [reg["slope"] * lo + reg["intercept"],
                       reg["slope"] * hi + reg["intercept"]],
            lw=PANEL_LW, color=COLORS["network"],
            label=f"fit: slope {reg['slope']:.2f}\n"
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}]")
    # plain text, not a mathtext subscript: a subscript is set at 70 % of the
    # label size, which would put a 6.3-pt glyph on a panel whose fonts are
    # otherwise identical to its neighbours' (and below the PLOS 8-pt floor)
    return _finish_panel(ax, "w_opt (fused - seg)  (deg)",
                         "network - seg  (deg)",
                         "position-domain regression", "upper left")


def panel_weight_regression(w, post, reg, flags=None, output_name="vis", ax=None,
                            n_show=6000, seed=0):
    """Panel B of 08v_variance_regression as a manuscript panel: the implied
    weight of one channel (its hybrid read, analysis.hybrid_weight) against
    the analytical posterior on every trial, the points coloured by the
    output each weight came from, with the fit (analysis.weight_regression;
    slope 1, intercept 0 = Bayes-optimal). Trials are subsampled for drawing
    only (n_show); the shares in the legend count every trial.

    Two legends, because one of five entries would cover the cloud on a
    2.5-in panel: the sources in the lower right, where a good network has
    almost no points (high posterior, low weight), and the two lines in the
    upper left under the headroom the y limit leaves above the cloud.
    """
    from matplotlib.lines import Line2D
    ax = _panel_axes(ax)
    rng = np.random.default_rng(seed)
    w, post = np.asarray(w, float), np.asarray(post, float)
    ok = np.flatnonzero(np.isfinite(w))
    pick = rng.choice(ok, min(n_show, ok.size), replace=False)
    handles, labels = [], []
    if flags is None:
        ax.scatter(post[pick], w[pick], s=2, alpha=0.15, color=COLORS["hybrid"],
                   rasterized=True)
    else:
        flags = np.asarray(flags)
        names = {0: f"var_{output_name} root", 1: f"mu_{output_name} ratio",
                 2: "variance peak"}
        # the root trials form the tight band along the identity line and are
        # drawn last, on top of the ratio trials' wider cloud
        for value in (1, 2, 0):
            sel = pick[flags[pick] == value]
            if sel.size:
                ax.scatter(post[sel], w[sel], s=2, alpha=0.2,
                           color=HYBRID_SOURCES[value][1], rasterized=True)
        for value, (_, col) in HYBRID_SOURCES.items():
            share = np.mean(flags[ok] == value)
            if share > 0:
                handles.append(Line2D([], [], ls="", marker="o", ms=3.5, color=col))
                labels.append(f"{names[value]}: {100 * share:.0f}%")
    ax.plot([0, 1], [0, 1], "--", lw=1, color=COLORS["analytical"],
            label="Bayes-optimal (slope 1)")
    ax.plot([0, 1], [reg["intercept"], reg["slope"] + reg["intercept"]],
            lw=PANEL_LW, color=COLORS["hybrid"],
            label=f"fit: slope {reg['slope']:.2f}\n"
                  f"[{reg['slope_ci95'][0]:.2f}, {reg['slope_ci95'][1]:.2f}]")
    if handles:
        ax.add_artist(ax.legend(handles, labels, fontsize=PANEL_FONT["legend"],
                                loc="lower right", handlelength=1.0,
                                handletextpad=0.5, borderaxespad=0.4))
    # the same y range as panel A (04_fusion_weight), with its headroom for
    # the legend above the cloud; ticks stop at 1 so the headroom reads as such
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1.4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return _finish_panel(ax, "analytical posterior p(C=1|x)",
                         f"implied weight, {output_name}",
                         "implied weight vs posterior", "upper left")


def panel_bias_vs_disparity(saved, ax=None):
    """The left half of 12_behavioral_bias as a manuscript panel."""
    ax = _panel_axes(ax)
    ax.plot(saved["bias_centres"], saved["bias_net"], "o-", lw=PANEL_LW,
            ms=PANEL_MS, color=COLORS["network"], label="network")
    if "bias_opt" in saved:
        ax.plot(saved["bias_centres_opt"], saved["bias_opt"], "--o",
                lw=PANEL_LW, ms=PANEL_MS, color=COLORS["analytical"],
                label="Bayes-optimal")
    ax.axhline(0, lw=0.5, color="gray")
    return _finish_panel(ax, "body-frame disparity (deg)",
                         "pull toward vision (deg)",
                         "bias vs disparity", "upper left")


def panel_output_scatter(y, yhat, name, r2, rmse, ax=None):
    """One output of 01_output_scatter on the standard panel."""
    ax = _panel_axes(ax)
    ax.scatter(y, yhat, s=2, alpha=0.15, color=COLORS["network"], rasterized=True)
    lo, hi = np.percentile(y, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"])
    # U+00B2 rather than mathtext $^2$: a mathtext superscript is set at 70 %
    # of the title size, which would add a 6.3-pt glyph to a 9-pt title. No
    # "=": with them the title is 2.03 in wide and touches the 2.5-in cell edge;
    # without, 1.82 in, with 0.11 in to spare (measured, not estimated).
    return _finish_panel(ax, "analytical", "network",
                         f"{name}   R\u00b2 {r2:.3f}   RMSE {rmse:.2f}")


def panel_error_histogram(err, name, ax=None):
    """One output of 02_error_histograms on the standard panel."""
    ax = _panel_axes(ax)
    ax.hist(err, bins=60, color=COLORS["network"])
    ax.axvline(0, lw=1, color=COLORS["analytical"])
    return _finish_panel(ax, "network - analytical", "trials",
                         f"{name}   bias = {err.mean():.2f}")


def panel_variance_hump(sig, output_name, legend=True, ax=None):
    """09/10_variance_hump on the WIDE cell: the four curves of SS7.6."""
    ax = _panel_axes(ax, cell=CELL_WIDE)
    c = np.asarray(sig["centres"])
    ax.plot(c, sig["mixture"], "--o", lw=PANEL_LW, ms=PANEL_MS,
            color=COLORS["analytical"], label="analytical mixture")
    ax.plot(c, sig["net"], "o-", lw=PANEL_LW, ms=PANEL_MS,
            color=COLORS["network"], label="network")
    ax.plot(c, sig["between"], ":", lw=PANEL_LW, color=COLORS["analytical"],
            label="between term w(1\u2212w)d\u00b2")
    ax.plot(c, sig["fixed"], "-", lw=1, color=COLORS.get("prop", "gray"),
            label="fixed weight (no hump)")
    top = max(np.max(sig["mixture"]), np.max(sig["net"]))
    fig = _finish_panel(ax, "analytical p(C=1|x)", "variance (deg\u00b2)",
                        f"uncertainty vs causal ambiguity: {output_name}")
    if legend:
        # headroom so the four-entry legend sits above the curves, not on
        # them: 40 % above the highest point (measured: at 30 % the nearest
        # marker is 0.035 in from the legend box), and the legend packed tight
        ax.set_ylim(0, 1.4 * top)
        ax.legend(fontsize=PANEL_FONT["legend"], loc="upper right",
                  handlelength=1.4, labelspacing=0.25, borderaxespad=0.3)
    else:
        ax.set_ylim(0, 1.05 * top)
    return fig


def manuscript_row(draw):
    """Three standard panels in one 7.5 x 2.5 in figure, lettered A-C."""
    return manuscript_grid(draw, ncols=3)


# One folder per manuscript figure, so every format of one figure is together.
# Rename here and nowhere else.
F2 = "fig2_weight_regression_bias"
FSC = "output_scatter_2x2"                   # figure number not yet assigned
FEH = "error_histograms_2x2"                 # figure number not yet assigned
F4 = "fig4_variance_hump"


def manuscript_panels(pred, d, names, analysis_cfg, w, w_prop, saved, metrics):
    """-> {folder/name: Figure} for manuscript_dir(run)/<folder>/ (results/manuscript/).

    Three fixed-frame panels with identical size, axes rectangle and fonts,
    plus the composed row. Returns {} when any input is missing, and says so.
    """
    files = getattr(saved, "files", saved) if saved is not None else ()
    need = {"fusion weight": w is not None,
            "weight regression": "weight_regression_vis" in metrics,
            "bias curve": saved is not None and "bias_centres" in files}
    missing = [k for k, ok in need.items() if not ok]
    if missing:
        print(f"manuscript panels skipped, missing: {', '.join(missing)}")
        return {}
    grid = analysis_cfg["disparity_grid"]
    reg = metrics["weight_regression_vis"]
    flags = saved["hybrid_flags_vis"] if "hybrid_flags_vis" in files else None

    def fw(ax=None):
        return panel_fusion_weight(d["disparity"], w, d["post_c1"], grid,
                                   w_prop=w_prop, ax=ax)

    def wr(ax=None):
        return panel_weight_regression(w, d["post_c1"], reg, flags=flags,
                                       output_name="vis", ax=ax)

    def bd(ax=None):
        return panel_bias_vs_disparity(saved, ax=ax)

    # 2 x 2 versions of 01_output_scatter and 02_error_histograms: same panel
    # order (A B / C D) as the originals, every cell a standard panel, so each
    # figure is 5.0 x 5.0 in
    target = np.stack([d[k] for k in names], axis=1)
    acc = accuracy(pred, target, names)

    def sc(j):
        return lambda ax=None: panel_output_scatter(
            target[:, j], pred[:, j], names[j], acc[j]["r2"], acc[j]["rmse"], ax=ax)

    def eh(j):
        return lambda ax=None: panel_error_histogram(
            pred[:, j] - target[:, j], names[j], ax=ax)

    # figure 4: 09/10_variance_hump side by side, two WIDE cells -> 7.5 x 2.5 in,
    # the same outer size as figure 2; one legend, in A, since both panels
    # draw the same four series
    figs = {
        f"{F2}/A_fusion_weight": fw(),
        f"{F2}/B_weight_regression": wr(),
        f"{F2}/C_bias_vs_disparity": bd(),
        f"{F2}/row_ABC": manuscript_row([fw, wr, bd]),
        f"{FSC}/output_scatter_2x2": manuscript_grid([sc(j) for j in range(len(names))], ncols=2),
        f"{FEH}/error_histograms_2x2": manuscript_grid([eh(j) for j in range(len(names))], ncols=2),
    }
    if "variance_signature_vis" in metrics and "variance_signature_prop" in metrics:
        sv, sp = metrics["variance_signature_vis"], metrics["variance_signature_prop"]
        vh = [lambda ax=None: panel_variance_hump(sv, "var_vis", ax=ax),
              lambda ax=None: panel_variance_hump(sp, "var_prop", legend=False, ax=ax)]
        figs[f"{F4}/row_AB"] = manuscript_grid(vh, ncols=2, cell=CELL_WIDE)
    else:
        print("figure 4 skipped: no variance_signature in metrics.json")
    return figs
