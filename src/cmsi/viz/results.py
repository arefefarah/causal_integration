"""Model results: the network against the analytical observer.

These are the figures that answer the research question, so they get their own
folder (results/<run>/figures/model) separate from the input and training checks.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import mean_by_bin
from cmsi.viz.style import COLORS


def output_scatter(pred, target, names):
    """Network vs analytical, one panel per output, with R^2 and RMSE."""
    rows = accuracy(pred, target, names)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    for i, (ax, row) in enumerate(zip(axes, rows)):
        y, yhat = target[:, i], pred[:, i]
        ax.scatter(y, yhat, s=2, alpha=0.15, color=COLORS["network"])
        lo, hi = np.percentile(y, [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "--", lw=1, color=COLORS["analytical"])
        ax.set(title=f"{row['output']}\nR2={row['r2']:.3f}  RMSE={row['rmse']:.2f}",
               xlabel="analytical", ylabel="network")
    fig.tight_layout()
    return fig


def error_histograms(pred, target, names):
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    axes = np.atleast_1d(axes)
    for i, (ax, name) in enumerate(zip(axes, names)):
        err = pred[:, i] - target[:, i]
        ax.hist(err, bins=60, color=COLORS["network"])
        ax.axvline(0, lw=1, color=COLORS["analytical"])
        ax.set(title=f"{name}   bias={err.mean():.2f}", xlabel="network - analytical")
    fig.tight_layout()
    return fig


def p_common_vs_disparity(disparity, p_common, grid):
    """The Kording-style curve: p(C=1) high near zero disparity, falling away."""
    centres, means, _ = mean_by_bin(disparity, p_common, grid)
    fig, ax = plt.subplots(figsize=(5.5, 4))
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
    fig, ax = plt.subplots(figsize=(5.8, 4))
    c_opt, m_opt, _ = mean_by_bin(disparity, p_analytical, grid)
    ax.plot(c_opt, m_opt, "--o", color=COLORS["analytical"], label="analytical p(C=1)")

    c_net, m_net, _ = mean_by_bin(disparity, w_network, grid)
    ax.plot(c_net, m_net, "o-", color=COLORS["network"], label="implied, from mu_vis")

    if w_prop is not None:
        c_p, m_p, _ = mean_by_bin(disparity, w_prop, grid)
        ax.plot(c_p, m_p, "s-", color=COLORS["prop"], label="implied, from mu_prop")

    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="fusion -> segregation transition")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fusion_weight_by_reliability(curves):
    """curves: output of analysis.by_reliability -> {level: (centres, means, n)}."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for level, (centres, means, _) in sorted(curves.items()):
        ax.plot(centres, means, "o-", label=f"sigma2_vis ~ {level:g}")
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="shift by different cue reliability")
    ax.legend()
    fig.tight_layout()
    return fig


def decoding_bars(scores, title="p(C=1) decodable from"):
    """scores: {layer_name: held-out r2} -- i.e. analysis.summarise output."""
    names = list(scores)
    fig, ax = plt.subplots(figsize=(4 + 0.7 * len(names), 4))
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

    fig, ax = plt.subplots(figsize=(6, 4))
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
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(x, y, s=2, alpha=0.15, color=COLORS["network"])
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
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def variance_hump(sig, output_name):
    """SS7.3: network Var output vs posterior bin, with the analytical mixture
    variance, its between-component term, and the humpless fixed-weight line."""
    c = np.asarray(sig["centres"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(c, sig["mixture"], "--o", color=COLORS["analytical"],
            label="analytical mixture variance")
    ax.plot(c, sig["net"], "o-", color=COLORS["network"], label="network output")
    ax.plot(c, sig["between"], ":", color=COLORS["analytical"],
            label="between-component term w(1-w)d^2")
    ax.plot(c, sig["fixed"], "-", lw=1, color=COLORS.get("prop", "gray"),
            label="best fixed-weight (no hump)")
    ax.set(xlabel="analytical p(C=1|x)", ylabel="variance (deg^2)",
           title=f"uncertainty vs causal ambiguity: {output_name}")
    ax.legend(fontsize=8)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(c, mc["bin_weight_net"], "ko-", lw=2, label="network", zorder=5)
    for k in ("averaging", "integration", "segregation", "selection", "fixed"):
        axes[0].plot(c, mc["bin_weight"][k], "--", label=k)
    axes[0].set(xlabel="analytical p(C=1|x) (decile centres)",
                ylabel="implied weight on the fused estimate",
                ylim=(-0.15, 1.15),
                title=f"per-decile weight: {output_name}")
    axes[0].legend(fontsize=8)
    for k in ("averaging", "integration", "segregation", "selection", "fixed"):
        axes[1].plot(c, mc["bin_rmse"][k], "o-", label=k)
    axes[1].set(xlabel="analytical p(C=1|x) (decile centres)",
                ylabel="RMSE vs network (deg)",
                title=f"which strategy explains the network "
                      f"(best: {mc['best']})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def behavioral_bias(saved):
    """SS7.6: Kording Fig. 2e analog + conditioning on the inferred cause."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(saved["bias_centres"], saved["bias_net"], "o-",
                 color=COLORS["network"], label="network")
    if "bias_opt" in saved:
        axes[0].plot(saved["bias_centres_opt"], saved["bias_opt"], "--o",
                     color=COLORS["analytical"], label="Bayes-optimal")
    axes[0].axhline(0, lw=0.5, color="gray")
    axes[0].set(xlabel="body-frame disparity (deg)",
                ylabel="hand-report pull toward vision (deg)",
                title="bias vs disparity (Kording 2e analog)")
    axes[0].legend(fontsize=8)
    for label, style in (("common", "o-"), ("separate", "s-")):
        c = saved.get(f"cond_bias_{label}_centres")
        b = saved.get(f"cond_bias_{label}")
        if c is not None and len(c):
            axes[1].plot(c, b, style, label=f"inferred {label} cause")
    axes[1].axhline(0, lw=0.5, color="gray")
    axes[1].set(xlabel="|disparity| (deg)", ylabel="bias (deg)",
                title="conditioned on the network's causal judgment (3b-c)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def congruency_panels(saved, balance_stats=None):
    """SS7.5: congruency-index distribution + balance-vs-posterior scatter."""
    idx = saved["congruency_index"]
    has_balance = "congruent_opposite_balance" in getattr(saved, "files", saved)
    fig, axes = plt.subplots(1, 2 if has_balance else 1,
                             figsize=(10.5 if has_balance else 5.5, 4.2))
    axes = np.atleast_1d(axes)
    axes[0].hist(idx, bins=30, color=COLORS["network"])
    axes[0].axvline(0.5, ls="--", lw=1, color=COLORS["analytical"])
    axes[0].axvline(-0.5, ls="--", lw=1, color=COLORS["analytical"])
    axes[0].set(xlabel="congruency index (corr of vis vs prop tuning)",
                ylabel="MSL units", title="congruent / opposite units")
    if has_balance:
        bal = saved["congruent_opposite_balance"]
        post = saved["post_c1"]
        axes[1].scatter(bal, post, s=3, alpha=0.2, color=COLORS["network"])
        title = "balance predicts p(C=1|x)"
        if balance_stats:
            title += f"  (corr {balance_stats.get('corr', float('nan')):.2f})"
        axes[1].set(xlabel="congruent - opposite mean activity",
                    ylabel="analytical p(C=1|x)", title=title)
    fig.tight_layout()
    return fig


def rf_shift_hist(saved):
    """Continuity analysis: distribution of RF shift gains and gain fields."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    g = saved["rf_shift_gain"]
    axes[0].hist(g[np.isfinite(g)], bins=30, color=COLORS["network"])
    axes[0].axvline(0, ls="--", lw=1, color=COLORS["analytical"], label="spatial code")
    axes[0].axvline(1, ls=":", lw=1, color=COLORS["analytical"], label="retinal code")
    axes[0].set(xlabel="RF shift gain (d preferred-spatial / d eye)",
                ylabel="MSL units", title="reference frame of MSL units")
    axes[0].legend(fontsize=8)
    gf = saved["rf_gain_field"]
    axes[1].hist(gf[np.isfinite(gf)], bins=30, color=COLORS["network"])
    axes[1].set(xlabel="d peak response / d eye  (gain field slope)",
                ylabel="MSL units", title="eye-position gain fields")
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
    return figs
