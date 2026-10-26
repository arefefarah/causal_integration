"""The manuscript figures drawn from the transformation investigation,
composed on the manuscript's standard panel (cmsi.viz.manuscript) and saved
under results/manuscript/, where the paper's figures live:

    fig11_transformation_decision   2 x 3   the transformation's uncertainty (A-C) and error (D-F)
                                            enter the causal decision as Bayes prescribes
    figS_decision_frame             2 x 2   the decision is made after the transformation (A, B),
                                            with the Bayesian eye gain (C), by units in the
                                            observer's frame (D)

    python scripts/transformation/manuscript_figures.py

Reads results/transformation_investigation/{stored,levels} (run_stored.py,
run_levels.py, frames_by_k.py). Panel code follows figures.py; only the
selection, the labels and the destination differ. The names come from
cmsi.viz.manuscript (FIG_TRANSFORMATION, FIG_S_DECISION_FRAME), the one place
a figure is numbered.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figures import (  # noqa: E402
    C_NET,
    C_R0,
    C_R1,
    C_TWIN,
    C_VIS,
    CLS,
    _bins_mean,
    _levels,
    _load,
    _series,
    _xcat,
)
from inv_common import OUT  # noqa: E402  (stubs cmsi so the viz modules load without torch)

from cmsi.viz.manuscript import (  # noqa: E402
    FIG_S_DECISION_FRAME,
    FIG_TRANSFORMATION,
    PANEL_LW,
    PANEL_MS,
    finish_panel,
    manuscript_dir,
    manuscript_grid,
    panel_axes,
)
from cmsi.viz.style import COLORS, apply_style, save_figures  # noqa: E402

FIG_TRANSFORM = FIG_TRANSFORMATION
FIG_FRAME = FIG_S_DECISION_FRAME
L_R1, L_R0, L_NET = "observer, noise-free eye signal", "observer, noisy eye signal", "network"  # 20 Oct 2026 (was "true eye position", "own eye signal")
XL_NOISE = "eye-signal noise sd (deg)"


# --------------------------------------------------------------------------- #
def fig_transformation_decision(D):
    lv = _levels(D["agg"], "eye_noise")
    S, F = D["seeds"], D["flag"]
    agg = S["aggregated"] if S else {}

    def g(k):
        a = agg.get(k)
        return (a["mean"], a["sd"]) if a and a["mean"] is not None else (np.nan, np.nan)

    def pA(ax):  # fusion window along the eye-noise lever
        panel_axes(ax)
        _series(ax, lv, "R1.crossing_R1", C_R1, L_R1, marker="s", ls="--")
        _series(ax, lv, "R0.crossing_R0", C_R0, L_R0, marker="s", ls="--")
        _series(ax, lv, "R0.crossing_net", C_NET, L_NET)
        _xcat(ax, lv, XL_NOISE)
        ax.set_ylim(3, 15)                      # headroom for the longer legend labels (20 Oct 2026)
        finish_panel(ax, XL_NOISE, "fusion window (deg)", "fusion window", "upper left")

    def pB(ax):  # causal accuracy along the eye-noise lever
        panel_axes(ax)
        _series(ax, lv, "R1.accuracy_R1", C_R1, L_R1, marker="s", ls="--")
        _series(ax, lv, "R0.accuracy_R0", C_R0, L_R0, marker="s", ls="--")
        _series(ax, lv, "R0.accuracy_net", C_NET, L_NET)
        _xcat(ax, lv, XL_NOISE)
        ax.set_ylim(0.5, 0.95)
        finish_panel(ax, XL_NOISE, "causal decision correct", "causal accuracy", "lower left")

    def pC(ax):  # window by eye-noise tercile in the wide-range configuration (22 Oct 2026)
        panel_axes(ax)
        wide = [r for r in D["rows"] if r["param"] == "eye_wide"]
        cn = np.array([[t["crossing_net"] for t in json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["eyerel"]["by_sig2_eye_tercile"]] for r in wide])
        co = np.array([[t["crossing_R0"] for t in json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["eyerel"]["by_sig2_eye_tercile"]] for r in wide])
        q = json.loads((OUT / "levels" / "numbers" / f"{wide[0]['name']}.json").read_text())["eyerel"]["by_sig2_eye_tercile"]
        x = np.arange(3)
        ax.errorbar(x, co.mean(0), yerr=co.std(0), fmt="s", ls="--", ms=PANEL_MS, lw=PANEL_LW, color=C_R0, capsize=2, label=L_R0)
        ax.errorbar(x, cn.mean(0), yerr=cn.std(0), fmt="o", ls="-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, capsize=2, label=L_NET)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{np.sqrt(t['sig2_eye_lo']):.1f}–{np.sqrt(t['sig2_eye_hi']):.1f}" for t in q])
        ax.set_xlim(-0.5, 2.5); ax.set_ylim(4.5, 7.5)
        finish_panel(ax, "eye-signal noise sd, per trial (deg)", "fusion window (deg)", "trial-by-trial eye reliability", "upper left")

    def pD(ax):  # natural error of the eye estimate: fusion of true common-cause trials
        panel_axes(ax)
        rows = F["terr"]["fusion_true_by_abs_terr"][:4]
        x = np.arange(len(rows))
        for key, col, lab, mk, ls in (("fusion_true_R1", C_R1, L_R1, "s", "--"), ("fusion_true_R0", C_R0, L_R0, "s", "--"),
                                      ("fusion_true_net", C_NET, L_NET, "o", "-")):
            y = np.array([g(f"terr.{key}_{r['lo']:g}_{r['hi']:g}")[0] for r in rows])
            e = np.array([g(f"terr.{key}_{r['lo']:g}_{r['hi']:g}")[1] for r in rows])
            if np.all(np.isnan(y)):
                y = np.array([r[key] for r in rows]); e = np.zeros(len(rows))
            ax.errorbar(x, y, yerr=e, fmt=mk, ls=ls, ms=PANEL_MS, lw=PANEL_LW, color=col, capsize=2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['lo']:g}–{r['hi']:g}" for r in rows]); ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(0.4, 0.9)                   # room below the curves for the legend (20 Oct 2026)
        finish_panel(ax, "|error of the eye estimate| (deg)", "fusion of common-cause trials", "natural eye-signal error", "lower left")

    def pE(ax):  # injected error: the weight curve translates by -k*eps
        panel_axes(ax)
        cols = {-10.0: "#d95f02", 0.0: C_NET, 10.0: "#7570b3"}
        for r in F["eps"]["levels"]:
            if r["eps"] not in cols:
                continue
            cn = np.array([[0.5 * (q["lo"] + q["hi"]), q["mean"]] for q in r["curve_net"]], float)
            co = np.array([[0.5 * (q["lo"] + q["hi"]), q["mean"]] for q in r["curve_R0eps"]], float)
            ax.plot(co[:, 0], co[:, 1], "--", lw=0.9, color=cols[r["eps"]], alpha=0.7)
            ax.plot(cn[:, 0], cn[:, 1], "o-", ms=PANEL_MS - 1, lw=PANEL_LW, color=cols[r["eps"]], label=("ε = 0°" if r["eps"] == 0 else f"ε = {r['eps']:+g}°"))
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        ax.set_xlim(-31, 31); ax.set_ylim(-0.05, 1.2)
        finish_panel(ax, "true disparity s_v − s_p (deg)", "implied fusion weight", "injected eye-signal error", None)
        ax.legend(fontsize=7, loc="upper right", handlelength=1.4)
        ax.text(0.02, 0.93, "dashed: observer, same signal", transform=ax.transAxes, fontsize=6.5, color="grey")

    def pF(ax):  # location of maximal fusion against eps
        panel_axes(ax)
        lv_e = F["eps"]["levels"]
        x = np.array([r["eps"] for r in lv_e])
        pn = np.array([g(f"eps.peak_w_vs_true_disparity_net_{e:+g}")[0] for e in x])
        sn = np.array([g(f"eps.peak_w_vs_true_disparity_net_{e:+g}")[1] for e in x])
        po = np.array([g(f"eps.peak_w_vs_true_disparity_R0eps_{e:+g}")[0] for e in x])
        if np.all(np.isnan(pn)):
            pn = np.array([r["peak_w_vs_true_disparity_net"] for r in lv_e]); sn = np.zeros(len(x))
            po = np.array([r["peak_w_vs_true_disparity_R0eps"] for r in lv_e])
        ax.plot(x, -F["eps"]["k"] * x, ls=":", lw=0.9, color="grey", label=f"−k·ε (k = {F['eps']['k']:.2f})")
        ax.plot(x, po, "s--", ms=PANEL_MS, lw=PANEL_LW, color=C_R0, label="observer, same signal")
        ax.errorbar(x, pn, yerr=sn, fmt="o-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, capsize=2, label=L_NET)
        finish_panel(ax, "injected eye-signal error ε (deg)", "disparity of maximal fusion (deg)", "error read as displacement", "upper right")

    # panel C is the trial-by-trial reliability of the wide-range configuration since 22 Oct 2026
    # (it replaced the 45-network window regression, removed the same day at the author's request)
    return manuscript_grid([pA, pB, pC, pD, pE, pF], ncols=3)


# --------------------------------------------------------------------------- #
def fig_decision_frame(D):
    F, fc = D["flag"], D["fcurves"]
    lv = _levels(D["agg"], "eye_sd")
    rows = [r for r in D["rows"] if r["param"] in ("eye_sd", "eye_noise", "eye_wide", "eye_ecc")]

    def pA(ax):  # which disparity the decision is sharp against (flagship)
        panel_axes(ax)
        edges = np.arange(-33, 34, 3.0)
        w, post = fc["w"], fc["post"]
        for x, col, lab in ((fc["disparity"], C_NET, "vs body-frame disparity"), (fc["retinal_disparity"], C_VIS, "vs retinal disparity")):
            c, m, s = _bins_mean(x, w, edges)
            co, mo, _ = _bins_mean(x, post, edges)
            ax.plot(co, mo, "--", lw=0.9, color=col, alpha=0.6)
            ax.errorbar(c, m, yerr=1.96 * s, fmt="o-", ms=PANEL_MS - 0.8, lw=PANEL_LW, color=col, capsize=1.5, label=lab)
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        ax.set_xlim(-33, 33); ax.set_ylim(-0.05, 1.2)
        finish_panel(ax, "disparity (deg)", "implied fusion weight", "which disparity decides", None)
        ax.legend(fontsize=7, loc="upper right", handlelength=1.4)
        ax.text(0.98, 0.74, "dashed: observer", transform=ax.transAxes, ha="right", fontsize=7, color="grey")

    def pB(ax):  # R2 of the weight from each disparity, across eye spreads
        panel_axes(ax)
        _series(ax, lv, "frame.lookup_r2_body_net", C_NET, "from |body-frame disparity|")
        _series(ax, lv, "frame.lookup_r2_retinal_net", C_VIS, "from |retinal disparity|", marker="^")
        _series(ax, lv, "frame.lookup_r2_body_R0", C_R0, "observer, each disparity", marker="s", ls="--")
        _series(ax, lv, "frame.lookup_r2_retinal_R0", "#8c8c8c", None, marker="s", ls="--")
        _xcat(ax, lv, "eye-position sd (deg)")
        ax.set_ylim(0, 1.55)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        finish_panel(ax, "eye-position sd (deg)", "R² of the implied weight", "R² from each disparity", None)
        ax.legend(fontsize=6.5, loc="upper left", handlelength=1.4, borderaxespad=0.3)

    def pC(ax):  # effective eye gain against the observer's k
        panel_axes(ax)
        ax.plot([0, 1.05], [0, 1.05], ls=":", lw=0.8, color="grey", label="gain = k (Bayes)")
        ax.axhline(1.0, lw=0.8, ls="--", color="#8c8c8c", label="gain = 1 (raw sum)")
        for params, col, mk, lab in ((("eye_sd",), C_NET, "o", "eye-spread levels"), (("eye_noise",), COLORS["eye"], "^", "eye-noise levels"),
                                     (("eye_wide", "eye_ecc"), "#8c8c8c", "s", "wide-range and eccentricity")):
            rr = [r for r in rows if r["param"] in params]
            x = [r["partials.eye_gain_k_observer"] for r in rr]; y = [r["partials.eye_gain_net_median"] for r in rr]
            ax.plot(x, y, mk, ms=PANEL_MS + 0.5, color=col, alpha=0.8, label=lab)
        ax.set_xlim(-0.03, 1.08); ax.set_ylim(-0.03, 1.15)
        finish_panel(ax, "observer's eye gain k", "network's effective eye gain", "effective eye gain", "upper left")

    def pD(ax):  # the opposite units' displacement index against 1 - k, causal vs twin
        panel_axes(ax)
        fr = json.loads((OUT / "levels" / "frames_by_k.json").read_text())
        groups = {}
        for r in fr:
            groups.setdefault((r["param"], r["level"]), []).append(r)

        def mean(key, recs):
            v = [r.get(key) for r in recs if r.get(key) is not None]
            return float(np.nanmean(v)) if v else np.nan

        def sd(key, recs):
            v = [r.get(key) for r in recs if r.get(key) is not None]
            return float(np.nanstd(v)) if len(v) > 1 else 0.0

        pts = {"eye_noise": [], "eye_sd": []}
        for (param, level), recs in sorted(groups.items()):
            if param in pts:
                pts[param].append((mean("expected_di", recs), mean("di_median_opposite", recs), sd("di_median_opposite", recs),
                                   mean("twin_di_median", recs), sd("twin_di_median", recs)))
        ax.plot([-0.05, 1.05], [-0.05, 1.05], ls=":", lw=0.8, color="grey", label="observer's frame, 1 − k")
        ax.axhline(1.0, lw=0.8, ls="--", color="#8c8c8c", label="retinal, 1")
        for i, (param, mk) in enumerate((("eye_sd", "o"), ("eye_noise", "^"))):
            P = np.array(pts[param])
            ax.errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], fmt=mk, ms=PANEL_MS + 0.5, color=CLS["opposite"], capsize=1.5,
                        label="causal network" if i == 0 else None)
            ax.errorbar(P[:, 0], P[:, 3], yerr=P[:, 4], fmt=mk, ms=PANEL_MS + 0.5, color=C_TWIN, capsize=1.5, mfc="none",
                        label="always-fuse twin" if i == 0 else None)
        ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.1, 1.2)
        finish_panel(ax, "expected shift 1 − k", "displacement index (median)", "opposite units' frame", None)
        ax.legend(fontsize=6.5, loc="lower right", handlelength=1.4, borderaxespad=0.3)

    return manuscript_grid([pA, pB, pC, pD], ncols=2)


def main():
    apply_style()
    D = _load()
    out = manuscript_dir()
    figs = {f"{FIG_TRANSFORM}/{FIG_TRANSFORM}": fig_transformation_decision(D),
            f"{FIG_FRAME}/{FIG_FRAME}": fig_decision_frame(D)}
    paths = save_figures(figs, out, formats=("png", "tif", "svg"))
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
