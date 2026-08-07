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


def fusion_weight_curve(disparity, w_network, p_analytical, grid):
    """The key panel: implied network weight against the optimal weight p(C=1).

    Trials near zero disparity are dropped by fusion_weight (fused ~= segregated),
    so the very centre of the curve rests on fewer trials and is noisier.
    """
    c_net, m_net, _ = mean_by_bin(disparity, w_network, grid)
    c_opt, m_opt, _ = mean_by_bin(disparity, p_analytical, grid)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(c_opt, m_opt, "--o", color=COLORS["analytical"], label="analytical p(C=1)")
    ax.plot(c_net, m_net, "o-", color=COLORS["network"], label="network implied weight")
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1), title="fusion -> segregation transition")
    ax.legend()
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


def all_figures(pred, d, names, analysis_cfg, w=None, curves=None,
                decoding=None, twin_decoding=None):
    """Every model figure -> results/<run>/figures/model.

    `d` should already be restricted to the trials `pred` was computed on
    (use data.subset(d, splits["test"])). `decoding` and `twin_decoding` are
    {layer: r2} dicts straight out of metrics.json.
    """
    target = np.stack([d[k] for k in names], axis=1)
    figs = {
        "01_output_scatter": output_scatter(pred, target, names),
        "02_error_histograms": error_histograms(pred, target, names),
        "03_p_common_vs_disparity": p_common_vs_disparity(
            d["disparity"], d["p_common"], analysis_cfg["disparity_grid"]),
    }
    if w is not None:
        figs["04_fusion_weight"] = fusion_weight_curve(
            d["disparity"], w, d["p_common"], analysis_cfg["disparity_grid"])
    if curves:
        figs["05_fusion_weight_by_reliability"] = fusion_weight_by_reliability(curves)
    if decoding:
        figs["06_decoding"] = decoding_bars(decoding)
    if decoding and twin_decoding:
        figs["07_emergent_vs_imposed"] = decoding_comparison(
            {"causal model": decoding, "always-fuse twin": twin_decoding})
    return figs
