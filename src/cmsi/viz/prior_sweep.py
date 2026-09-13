"""The cross-prior figure (design SS9.3).

One publication figure, four panels:

    A  implied fusion weight vs disparity, one curve per prior
    B  network transition midpoint against the analytical midpoint
    C  position-domain regression slope vs prior
    D  the variance hump vs prior

Colour encodes the prior, which is an ordered magnitude, so it uses a single
sequential hue ramp (light = low prior, dark = high) rather than categorical
hues. The analytical prediction is always the dashed/open series in the same
ramp position, so a reader compares like with like within each prior.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cmsi.viz.manuscript import (
    CELL_FULL,
    CELL_WIDE,
    PANEL_FONT,
    PANEL_LW,
    PANEL_MS,
    cell_axes,
    finish_panel,
    letter,
)
from cmsi.viz.style import SIZE, exact_frame

RAMP = "viridis"
NET = "#2F6FD0"
OPT = "#111418"


def _ramp(priors):
    norm = Normalize(vmin=min(priors) - 0.05, vmax=max(priors) + 0.05)
    cmap = plt.get_cmap(RAMP)
    return norm, cmap, {p: cmap(norm(p)) for p in priors}


def figure(rows, curves, figsize=SIZE["composite"]):
    """rows: list of per-prior stat dicts. curves: the npz mapping."""
    rows = sorted(rows, key=lambda r: r["p_common"])
    priors = [r["p_common"] for r in rows]
    norm, cmap, col = _ramp(priors)

    # constrained layout keeps the colourbar and every label INSIDE figsize, so
    # the saved file is never wider than the 7.5 in PLOS allows
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ---- A: implied weight vs disparity, one curve per prior ---------------
    for r in rows:
        p = r["p_common"]
        cn = curves.get(f"p{p:.2f}_centres_net")
        wn = curves.get(f"p{p:.2f}_w_net")
        sn = curves.get(f"p{p:.2f}_se_net")
        co = curves.get(f"p{p:.2f}_centres_opt")
        wo = curves.get(f"p{p:.2f}_w_opt")
        if co is not None:
            keep = np.abs(co) <= 30
            axA.plot(co[keep], wo[keep], "--", lw=1.1, color=col[p],
                     alpha=0.5, zorder=2)
        if cn is not None and len(cn):
            if sn is not None and len(sn) == len(cn):
                axA.errorbar(cn, wn, yerr=1.96 * sn, fmt="none",
                             ecolor=col[p], elinewidth=0.9, alpha=0.55, zorder=3)
            axA.plot(cn, wn, "o-", ms=3.2, lw=1.7, color=col[p], zorder=4)
    axA.set(xlabel="body-frame disparity (deg)",
            ylabel="implied weight on the fused estimate",
            ylim=(-0.06, 1.06), xlim=(-31, 31))
    axA.axhline(0.5, lw=0.6, ls=":", color="grey", zorder=1)
    axA.set_title("A   The fusion-to-segregation transition shifts with the prior",
                  loc="left", fontsize=11, fontweight="600")
    handles = [plt.Line2D([], [], color="grey", lw=1.7, marker="o", ms=3.4,
                          label="network (implied)"),
               plt.Line2D([], [], color="grey", lw=1.1, ls="--",
                          label="Bayes-optimal p(C=1|x)")]
    axA.legend(handles=handles, fontsize=8.5, loc="upper left", frameon=False,
               ncol=1, bbox_to_anchor=(0.005, 0.99))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axA, pad=0.012, fraction=0.028)
    cb.set_label("prior  $p_{\\mathrm{common}}$", fontsize=9)

    # ---- B: network midpoint vs analytical midpoint ------------------------
    mn = np.array([r["midpoint_net"] for r in rows])
    mo = np.array([r["midpoint_opt"] for r in rows])
    ok = np.isfinite(mn) & np.isfinite(mo)
    if not ok.any():
        # Every logistic fit was rejected (undertrained networks, or priors so
        # extreme there is no transition to find). Say so on the panel rather
        # than crashing -- the rest of the figure is still informative.
        axB.text(0.5, 0.5, "no usable transition fit\n(see sweep.json)",
                 ha="center", va="center", transform=axB.transAxes,
                 fontsize=10, color="#8B96A8")
        axB.set(xlabel="analytical transition midpoint (deg)",
                ylabel="network transition midpoint (deg)")
        axB.set_title("B   Quantitative match, no free parameters",
                      loc="left", fontsize=11, fontweight="600")
        axB.spines[["top", "right"]].set_visible(False)
        return _panel_c(fig, axA, axB, axC, rows, priors, col)
    lo = float(min(mn[ok].min(), mo[ok].min())) - 1.2
    hi = float(max(mn[ok].max(), mo[ok].max())) + 1.2
    axB.plot([lo, hi], [lo, hi], "--", lw=1, color=OPT, zorder=1,
             label="identity")
    sem = np.array([r.get("midpoint_net_sem", 0.0) or 0.0 for r in rows])
    if np.any(sem > 0):
        axB.errorbar(mo, mn, yerr=sem, fmt="none", ecolor="#8B96A8",
                     elinewidth=1.1, capsize=2.5, zorder=2)
    for r, a, b in zip(rows, mo, mn, strict=True):
        axB.scatter(a, b, s=52, color=col[r["p_common"]], zorder=3,
                    edgecolor="white", linewidth=0.7)
    if ok.sum() >= 2:
        s, i = np.polyfit(mo[ok], mn[ok], 1)
        rr = np.corrcoef(mo[ok], mn[ok])[0, 1] ** 2
        axB.plot([lo, hi], [s * lo + i, s * hi + i], lw=1.5, color=NET,
                 zorder=2, label=f"fit: slope {s:.2f}, $R^2$ = {rr:.3f}")
    axB.set(xlabel="analytical transition midpoint (deg)",
            ylabel="network transition midpoint (deg)",
            xlim=(lo, hi), ylim=(lo, hi))
    axB.set_title("B   Quantitative match,\n      no free parameters",
                  loc="left", fontsize=11, fontweight="600")
    axB.legend(fontsize=8.5, frameon=False, loc="upper left")

    return _panel_c(fig, axA, axB, axC, rows, priors, col)


def _panel_c(fig, axA, axB, axC, rows, priors, col):
    """Panel C, factored out so the figure still completes when panel B has
    nothing to plot."""
    sl = np.array([r["posreg_slope_vis"] for r in rows])
    ci = np.array([r["posreg_ci_vis"] for r in rows])
    err = np.vstack([sl - ci[:, 0], ci[:, 1] - sl])
    n_seeds = max((r.get("n_seeds", 1) for r in rows), default=1)
    bar_label = (f"mean $\\pm$ 95% CI across {n_seeds} seeds" if n_seeds > 1
                 else "within-run 95% CI")
    axC.axhline(1.0, ls="--", lw=1, color=OPT, zorder=1,
                label="Bayes-optimal (slope 1)")
    axC.errorbar(priors, sl, yerr=err, fmt="none", ecolor="#8B96A8",
                 elinewidth=1.1, capsize=2.5, zorder=2, label=bar_label)
    for r, y in zip(rows, sl, strict=True):
        axC.scatter(r["p_common"], y, s=52, color=col[r["p_common"]], zorder=3,
                    edgecolor="white", linewidth=0.7)
    axC.set(xlabel="prior  $p_{\\mathrm{common}}$",
            ylabel="position-regression slope", ylim=(0, 1.18))
    # two lines: at half the page width a one-line title overruns the figure
    # edge, and bbox_inches="tight" would then grow the file past 7.5 in
    axC.set_title("C   Model averaging at every prior,\n      slightly conservative",
                  loc="left", fontsize=11, fontweight="600")
    axC.legend(fontsize=8.5, frameon=False, loc="lower right")

    for ax in (axA, axB, axC):
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def hump_panel(rows, ax=None):
    """Optional extra: the variance hump, network vs analytical, per prior."""
    rows = sorted(rows, key=lambda r: r["p_common"])
    priors = [r["p_common"] for r in rows]
    _, _, col = _ramp(priors)
    if ax is None:
        _, ax = plt.subplots(figsize=SIZE["single"])
    hn = [r["hump_net"] for r in rows]
    ho = [r["hump_opt"] for r in rows]
    ax.plot(priors, ho, "--o", ms=4, lw=1.2, color=OPT, mfc="white",
            label="analytical mixture variance")
    ax.plot(priors, hn, "-", lw=1.5, color=NET, zorder=2, label="network output")
    hsem = np.array([r.get("hump_net_sem", 0.0) or 0.0 for r in rows])
    if np.any(hsem > 0):
        ax.errorbar(priors, hn, yerr=hsem, fmt="none", ecolor="#8B96A8",
                    elinewidth=1.1, capsize=2.5, zorder=2)
    for p, y in zip(priors, hn, strict=True):
        ax.scatter(p, y, s=48, color=col[p], zorder=3, edgecolor="white",
                   linewidth=0.7)
    ax.set(xlabel="prior  $p_{\\mathrm{common}}$",
           ylabel="mid-ambiguity variance elevation (deg$^2$)")
    ax.set_title("Variance hump vs prior", loc="left", fontsize=11,
                 fontweight="600")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# --------------------------------------------------------------------------- #
# The manuscript version, on the standard cells (viz/manuscript.py)
# --------------------------------------------------------------------------- #
FOLDER = "prior_sweep"                       # figure number not yet assigned
COLORBAR_RESERVE = 0.62                      # in, carved from A's axes width


def manuscript_figure(rows, curves):
    """A over B + C: one CELL_FULL row above two CELL_WIDE cells -> 7.5 x 5 in.

    Same content as `figure`, restyled to the standard: 8/9-pt text, 12-pt
    panel letters, absolute margins shared with every other manuscript figure.
    The colourbar lives inside A's cell, carved from the axes width, so the
    cell's outer margins are untouched. Labels avoid subscripts (a mathtext
    subscript would be set at 6.3 pt): the prior is "common-cause prior".
    """
    rows = sorted(rows, key=lambda r: r["p_common"])
    priors = [r["p_common"] for r in rows]
    norm, cmap, col = _ramp(priors)
    W, H = CELL_FULL[0], CELL_FULL[1] + CELL_WIDE[1]
    fig = plt.figure(figsize=(W, H))
    axA = cell_axes(fig, CELL_FULL, origin=(0, CELL_WIDE[1]),
                    reserve_right=COLORBAR_RESERVE)
    axB = cell_axes(fig, CELL_WIDE, origin=(0, 0))
    axC = cell_axes(fig, CELL_WIDE, origin=(CELL_WIDE[0], 0))

    # ---- A -------------------------------------------------------------------
    for r in rows:
        p = r["p_common"]
        cn, wn, sn = (curves.get(f"p{p:.2f}_{k}") for k in ("centres_net", "w_net", "se_net"))
        co, wo = curves.get(f"p{p:.2f}_centres_opt"), curves.get(f"p{p:.2f}_w_opt")
        if co is not None:
            keep = np.abs(co) <= 30
            axA.plot(co[keep], wo[keep], "--", lw=0.9, color=col[p], alpha=0.5, zorder=2)
        if cn is not None and len(cn):
            if sn is not None and len(sn) == len(cn):
                axA.errorbar(cn, wn, yerr=1.96 * sn, fmt="none", ecolor=col[p],
                             elinewidth=0.8, alpha=0.55, zorder=3)
            axA.plot(cn, wn, "o-", ms=PANEL_MS - 0.5, lw=PANEL_LW, color=col[p], zorder=4)
    axA.axhline(0.5, lw=0.6, ls=":", color="grey", zorder=1)
    axA.set(ylim=(-0.06, 1.06), xlim=(-31, 31))
    handles = [plt.Line2D([], [], color="grey", lw=PANEL_LW, marker="o", ms=PANEL_MS,
                          label="network (implied)"),
               plt.Line2D([], [], color="grey", lw=0.9, ls="--",
                          label="Bayes-optimal p(C=1|x)")]
    finish_panel(axA, "body-frame disparity (deg)", "implied fusion weight",
                 "The fusion-to-segregation transition shifts with the prior")
    axA.legend(handles=handles, fontsize=PANEL_FONT["legend"], loc="upper left",
               handlelength=1.6, borderaxespad=0.4)
    # colourbar: inside A's cell, to the right of the axes
    ax_pos = axA.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.06 / W, ax_pos.y0, 0.1 / W, ax_pos.height])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=PANEL_FONT["tick"])
    cb.set_label("common-cause prior", fontsize=PANEL_FONT["label"])
    cb.outline.set_linewidth(0.6)

    # ---- B -------------------------------------------------------------------
    mn = np.array([r["midpoint_net"] for r in rows])
    mo = np.array([r["midpoint_opt"] for r in rows])
    ok = np.isfinite(mn) & np.isfinite(mo)
    if ok.any():
        lo = float(min(mn[ok].min(), mo[ok].min())) - 1.2
        hi = float(max(mn[ok].max(), mo[ok].max())) + 1.2
        axB.plot([lo, hi], [lo, hi], "--", lw=1, color=OPT, zorder=1, label="identity")
        sem = np.array([r.get("midpoint_net_sem", 0.0) or 0.0 for r in rows])
        if np.any(sem > 0):
            axB.errorbar(mo, mn, yerr=sem, fmt="none", ecolor="#8B96A8",
                         elinewidth=0.9, capsize=2, zorder=2)
        for r, a, b in zip(rows, mo, mn, strict=True):
            axB.scatter(a, b, s=26, color=col[r["p_common"]], zorder=3,
                        edgecolor="white", linewidth=0.6)
        if ok.sum() >= 2:
            sl, ic = np.polyfit(mo[ok], mn[ok], 1)
            rr = np.corrcoef(mo[ok], mn[ok])[0, 1] ** 2
            axB.plot([lo, hi], [sl * lo + ic, sl * hi + ic], lw=PANEL_LW, color=NET,
                     zorder=2, label=f"fit: slope {sl:.2f}, R\u00b2 = {rr:.3f}")
        axB.set(xlim=(lo, hi), ylim=(lo, hi))
        finish_panel(axB, "analytical midpoint (deg)", "network midpoint (deg)",
                     "Quantitative match, no free parameters", "upper left")
    else:
        axB.text(0.5, 0.5, "no usable transition fit\n(see sweep.json)", ha="center",
                 va="center", transform=axB.transAxes, fontsize=PANEL_FONT["label"],
                 color="#8B96A8")
        finish_panel(axB, "analytical midpoint (deg)", "network midpoint (deg)",
                     "Quantitative match, no free parameters")

    # ---- C -------------------------------------------------------------------
    sl = np.array([r["posreg_slope_vis"] for r in rows])
    ci = np.array([r["posreg_ci_vis"] for r in rows])
    err = np.vstack([sl - ci[:, 0], ci[:, 1] - sl])
    n_seeds = max((r.get("n_seeds", 1) for r in rows), default=1)
    bar_label = (f"mean \u00b1 95% CI across {n_seeds} seeds" if n_seeds > 1
                 else "within-run 95% CI")
    axC.axhline(1.0, ls="--", lw=1, color=OPT, zorder=1, label="Bayes-optimal (slope 1)")
    axC.errorbar(priors, sl, yerr=err, fmt="none", ecolor="#8B96A8", elinewidth=0.9,
                 capsize=2, zorder=2, label=bar_label)
    for r, y in zip(rows, sl, strict=True):
        axC.scatter(r["p_common"], y, s=26, color=col[r["p_common"]], zorder=3,
                    edgecolor="white", linewidth=0.6)
    axC.set(ylim=(0, 1.18))
    finish_panel(axC, "common-cause prior", "position-regression slope",
                 "Model averaging at every prior, slightly conservative",
                 "lower right")

    letter([axA, axB, axC])
    exact_frame(fig)
    return fig
