"""Model results: the network against the analytical observer.

These are the figures that answer the research question, so they get their own
folder (results/<run>/figures/model) separate from the input and training checks.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import implied_weight_by_posterior, mean_by_bin
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
    """The key panel: implied network weight against the optimal weight p(C=1).

    Both output columns encode the same weight, so `w_network` (from mu_vis) and
    `w_prop` (from mu_prop) are two readings of one quantity. Their denominators
    are proportional to Pp and Pv respectively, so the one derived from the LESS
    reliable cue is the better conditioned of the two.

    Near zero disparity the two hypotheses coincide and no reading is possible
    from either -- fusion_weight returns NaN there, so the centre of the curve
    rests on few trials and swings wildly. That is a property of the estimator,
    not of the network.
    """
    fig, ax = plt.subplots(figsize=SIZE["single"])
    c_opt, m_opt, _ = mean_by_bin(disparity, p_analytical, grid)
    ax.plot(c_opt, m_opt, "--o", color=COLORS["analytical"], label="analytical p(C=1)")

    c_net, m_net, _ = mean_by_bin(disparity, w_network, grid)
    ax.plot(c_net, m_net, "o-", color=COLORS["network"], label="implied, from mu_vis")

    if w_prop is not None:
        c_p, m_p, _ = mean_by_bin(disparity, w_prop, grid)
        ax.plot(c_p, m_p, "s-", color=COLORS["prop"], label="implied, from mu_prop")

    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="fusion -> segregation transition")
    ax.legend()
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


def implied_weight_vs_posterior(res, method="least_squares"):
    """One estimator's implied weight against the analytical posterior.

    `res` is the output of analysis.implied_weight_by_posterior, which holds
    both estimators. `method` picks the one to draw, so calling this twice puts
    the two on identical axes and they can be compared directly:

        "least_squares"  w = sum(Delta*(est-seg))/sum(Delta^2), every trial
        "ratio"          mean of the sigma_w-filtered per-trial ratio, a subset

    A Bayes-optimal model-averaging observer puts every point on the identity
    line. For the filtered estimator the fraction of trials the filter retains
    is drawn on the right-hand axis, because that fraction is what explains
    where the two estimators part company: a high posterior means the cues
    agree, which means a small Delta, which is what the filter removes.
    """
    if method not in ("least_squares", "ratio"):
        raise ValueError(f"method must be 'least_squares' or 'ratio', got {method!r}")
    ls = method == "least_squares"
    w = res["w_ls"] if ls else res["w_filt"]
    se = res["se_ls"] if ls else res["se_filt"]
    x, good = res["centres"], np.isfinite(w)

    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot([0, 1], [0, 1], "--", lw=1.1, color=COLORS["analytical"],
            label="Bayes-optimal (w = posterior)")
    ax.errorbar(x[good], w[good], yerr=1.96 * np.nan_to_num(se[good]),
                fmt="o-", ms=4, lw=1.7, capsize=2.5, color=COLORS["network"],
                label="network (implied weight)")

    if ls:
        note = f"all {int(res['n'].sum())} trials used"
    else:
        ax2 = ax.twinx()
        ax2.plot(x, 100 * res["frac_kept"], ":", lw=1.2, color="0.45")
        ax2.set_ylabel("trials kept by the filter (%)", color="0.35", fontsize=9)
        ax2.tick_params(axis="y", colors="0.35", labelsize=8)
        ax2.set_ylim(-4, 104)
        lost = int(good.size - good.sum())
        note = (f"{int(res['n_kept'].sum())} of {int(res['n'].sum())} trials kept"
                + (f"; {lost} bins unmeasurable" if lost else ""))

    ax.set(xlabel="analytical posterior p(C=1|x)",
           ylabel="implied weight on the fused estimate",
           xlim=(0, 1), ylim=(-0.05, 1.05),
           title=("least squares, no division" if ls
                  else "per-trial ratio, sigma_w filtered"))
    ax.text(0.03, 0.95, note, transform=ax.transAxes, fontsize=8,
            color="0.35", va="top")
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
    if "variance_signature_vis" in metrics:
        figs["09_variance_hump_vis"] = variance_hump(
            metrics["variance_signature_vis"], "var_vis")
        figs["10_variance_hump_prop"] = variance_hump(
            metrics["variance_signature_prop"], "var_prop")
    if "model_comparison_vis" in metrics:
        figs["11_model_comparison"] = model_comparison_curves(
            metrics["model_comparison_vis"], "mu_vis")
    if saved is not None and "bias_centres" in getattr(saved, "files", saved):
        figs["12_behavioral_bias"] = behavioral_bias(saved)
    if saved is not None and "congruency_index" in getattr(saved, "files", saved):
        figs["13_congruency"] = congruency_panels(
            saved, metrics.get("balance_vs_post"))
    if saved is not None and "rf_shift_gain" in getattr(saved, "files", saved):
        figs["14_rf_shifts"] = rf_shift_hist(saved)
    if ("sigma_out" in metrics and "mu_vis" in names
            and "seg_vis_mu" in d and "fused_mu" in d):
        # the same quantity by the two estimators, on identical axes.
        # sigma_out is the read-out noise measured on the p_common=1 control
        # (03_analyze.py --control); this run's own residual_std must NOT be
        # used here, since on the flagship it also contains causal
        # misweighting and is ~3x too large, which starves the sigma_w filter.
        i = names.index("mu_vis")
        wbp = implied_weight_by_posterior(
            pred[:, i], d["seg_vis_mu"], d["fused_mu"], d["post_c1"],
            metrics["sigma_out"][i],
            sigma_w_criterion=analysis_cfg.get("sigma_w_criterion", 0.1),
            min_separation=analysis_cfg.get("min_separation", 1.0))
        figs["15_weight_vs_post_ratio"] = implied_weight_vs_posterior(wbp, "ratio")
        figs["16_weight_vs_post_leastsq"] = implied_weight_vs_posterior(
            wbp, "least_squares")
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
            color=COLORS["network"], label="implied, from mu_vis")
    if w_prop is not None:
        c_p, m_p, _ = mean_by_bin(disparity, w_prop, grid)
        ax.plot(c_p, m_p, "s-", lw=PANEL_LW, ms=PANEL_MS,
                color=COLORS["prop"], label="implied, from mu_prop")
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
    need = {"fusion weight": w is not None,
            "position regression": "position_regression_vis" in metrics,
            "bias curve": saved is not None
            and "bias_centres" in getattr(saved, "files", saved)}
    missing = [k for k, ok in need.items() if not ok]
    if missing:
        print(f"manuscript panels skipped, missing: {', '.join(missing)}")
        return {}
    i = names.index("mu_vis")
    grid = analysis_cfg["disparity_grid"]
    reg = metrics["position_regression_vis"]

    def fw(ax=None):
        return panel_fusion_weight(d["disparity"], w, d["post_c1"], grid,
                                   w_prop=w_prop, ax=ax)

    def pr(ax=None):
        return panel_position_regression(pred[:, i], d["seg_vis_mu"],
                                         d["fused_mu"], d["post_c1"], reg, ax=ax)

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
        f"{F2}/B_position_regression": pr(),
        f"{F2}/C_bias_vs_disparity": bd(),
        f"{F2}/row_ABC": manuscript_row([fw, pr, bd]),
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
