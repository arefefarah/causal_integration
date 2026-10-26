"""The figures of the transformation -> causal inference investigation, on the
manuscript's standard panel (cmsi.viz.manuscript), saved as png / tif / svg
under results/transformation_investigation/figures/<name>/ (never to
results/manuscript/: the figures join the paper only if you decide so).

    python scripts/transformation/figures.py            # every figure the stored numbers allow

Figures (one per channel of the plan):
    figS_transformation_uncertainty   A: eye-noise sweep (sd 0.3 ... 1000): window, weights, accuracy, hump, slopes
    figS_eye_reliability              B: the wide-reliability network: the eye signal's reliability trial by trial
    figS_transformation_error         C: natural and injected eye-signal errors; gain errors
    figS_decision_frame               D: body-frame vs retinal disparity; leakage; shared error; across eye spreads
    figS_eye_gain                     E: effective eye gain vs the observer's k; the (1-k) eye bias
    figS_transformation_cost          F: the three gaps (net-R0, R0-R1, R2) and the unit classes across levels
    figS_eye_eccentricity             A (second lever): eccentricity-dependent eye noise, by |eye|
    figS_units_frame                  G: the opposite units' displacement index against the observer's 1 - k; Bayes-frame shares, causal vs twin
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import OUT  # noqa: E402  (also stubs cmsi so the viz modules load without torch)

from cmsi.viz.manuscript import (  # noqa: E402
    PANEL_FONT,
    PANEL_LW,
    PANEL_MS,
    finish_panel,
    manuscript_grid,
    panel_axes,
)
from cmsi.viz.style import COLORS, apply_style, save_figures  # noqa: E402

FIGS = OUT / "figures"
C_NET, C_R0, C_R1, C_R2 = COLORS["network"], "#111418", "#8c8c8c", "#2ca02c"
C_VIS, C_PROP, C_TWIN = COLORS["visual"], COLORS["prop"], COLORS["twin"]
CLS = {"congruent": "#1f77b4", "opposite": "#d62728", "mixed": "#ccbb44"}


def _load():
    L = OUT / "levels"
    S = OUT / "stored"
    out = {"agg": [], "rows": [], "curves": {}, "seeds": None, "flag": None, "fcurves": None}
    if (L / "aggregated.json").exists():
        out["agg"] = json.loads((L / "aggregated.json").read_text())
        out["rows"] = json.loads((L / "rows.json").read_text())
        if (L / "curves.npz").exists():
            out["curves"] = dict(np.load(L / "curves.npz"))
    if (S / "seeds.json").exists():
        out["seeds"] = json.loads((S / "seeds.json").read_text())
    if (S / "numbers_flagship.json").exists():
        out["flag"] = json.loads((S / "numbers_flagship.json").read_text())
    if (S / "curves_flagship.npz").exists():
        out["fcurves"] = dict(np.load(S / "curves_flagship.npz"))
    return out


def _levels(agg, param):
    return sorted([a for a in agg if a["param"] == param], key=lambda a: a["level"])


def _series(ax, lv, key, color, label, marker="o", ls="-", x=None, offset=0.0):
    y = np.array([a.get(key, np.nan) if a.get(key) is not None else np.nan for a in lv], float)
    e = np.array([a.get(key + "_sem", 0.0) or 0.0 for a in lv], float)
    xx = np.arange(len(lv)) if x is None else np.asarray(x, float)
    ax.errorbar(xx + offset, y, yerr=e, fmt=marker, ls=ls, ms=PANEL_MS, lw=PANEL_LW, color=color, capsize=2, label=label)


def _xcat(ax, lv, label, fmt="{:g}"):
    ax.set_xticks(np.arange(len(lv)))
    ax.set_xticklabels([fmt.format(a["level"]) for a in lv])
    ax.set_xlim(-0.5, len(lv) - 0.5)
    ax.set_xlabel(label, fontsize=PANEL_FONT["label"])


def _bins_mean(x, y, edges, min_n=30):
    c, m, s = [], [], []
    ok = np.isfinite(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (x >= lo) & (x < hi)
        if sel.sum() >= min_n:
            c.append(0.5 * (lo + hi)); m.append(y[sel].mean()); s.append(y[sel].std() / np.sqrt(sel.sum()))
    return np.array(c), np.array(m), np.array(s)


# --------------------------------------------------------------------------- #
def fig_uncertainty(D):
    lv = _levels(D["agg"], "eye_noise")
    if not lv:
        return {}
    xl = "eye-signal noise sd (deg)"

    def pA(ax):
        panel_axes(ax)
        _series(ax, lv, "R1.crossing_R1", C_R1, "true-eye observer R1", marker="s", ls="--")
        _series(ax, lv, "R0.crossing_R0", C_R0, "own observer R0", marker="s", ls="--")
        _series(ax, lv, "R0.crossing_net", C_NET, "network")
        _xcat(ax, lv, xl)
        finish_panel(ax, xl, "fusion window (deg)", "fusion window", "upper left")

    def pB(ax):
        panel_axes(ax)
        _series(ax, lv, "R1.fused_visual_weight_R1", C_R1, "R1, weight on vision in fusion", marker="s", ls="--")
        _series(ax, lv, "R1.fused_visual_weight_R0", C_R0, "R0, weight on vision in fusion", marker="s", ls="--")
        _series(ax, lv, "R0.fusion_R0", "#555555", "R0 fusion (|d| ≤ 2°)", marker="^", ls=":")
        _series(ax, lv, "R0.fusion_net", C_NET, "network fusion (|d| ≤ 2°)", marker="^")
        ax.set_ylim(0, 1.3)
        _xcat(ax, lv, xl)
        finish_panel(ax, xl, "weight / fusion", "visual weight, fusion", None)
        ax.legend(fontsize=5.5, loc="upper left", ncol=2, handlelength=1.1, columnspacing=0.5)

    def pC(ax):
        panel_axes(ax)
        _series(ax, lv, "R1.accuracy_R1", C_R1, "R1 (true eye)", marker="s", ls="--")
        _series(ax, lv, "R0.accuracy_R0", C_R0, "R0 (own observer)", marker="s", ls="--")
        _series(ax, lv, "R0.accuracy_net", C_NET, "network")
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        _xcat(ax, lv, xl)
        ax.set_ylim(0.45, 0.9)
        finish_panel(ax, xl, "causal decision correct", "causal accuracy", "lower left")

    def pD(ax):
        panel_axes(ax)
        _series(ax, lv, "R0.hump_R0", C_R0, "observer", marker="s", ls="--")
        _series(ax, lv, "R0.hump_net", C_NET, "network")
        _xcat(ax, lv, xl)
        finish_panel(ax, xl, "variance hump (deg²)", "variance hump", "upper left")

    def pE(ax):
        panel_axes(ax)
        _series(ax, lv, "R0.weight_vs_post_slope", C_VIS, "weight on p(C=1|x)")
        _series(ax, lv, "R0.posreg_slope_vis", C_VIS, "position regr., vis", marker="^", ls="--")
        _series(ax, lv, "R0.posreg_slope_prop", C_PROP, "position regr., hand", marker="v", ls="--")
        ax.axhline(1.0, lw=0.6, ls=":", color="grey")
        ax.set_ylim(0.4, 1.3)
        _xcat(ax, lv, xl)
        finish_panel(ax, xl, "slope (Bayes-optimal: 1)", "Bayesian weighting", "lower left")

    def pF(ax):
        panel_axes(ax)
        rows = [r for r in D["rows"] if r["param"] in ("eye_noise", "eye_sd", "eye_wide", "eye_ecc")]
        x = np.array([r["R0.crossing_R0"] for r in rows], float)
        y = np.array([r["R0.crossing_net"] for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[ok], y[ok], "o", ms=PANEL_MS, color=C_NET, alpha=0.7, label="one network")
        lim = [0, max(1.0, np.nanmax(np.concatenate([x[ok], y[ok]])) * 1.1)]
        ax.plot(lim, lim, ls=":", lw=0.8, color="grey", label="identity")
        if ok.sum() > 2:
            b = np.polyfit(x[ok], y[ok], 1)
            ax.plot(lim, np.polyval(b, lim), "-", lw=PANEL_LW, color=C_R0, label=f"fit slope {b[0]:.2f}")
        ax.set_xlim(lim); ax.set_ylim(lim)
        finish_panel(ax, "observer window (deg)", "network window (deg)", "windows, all networks", "upper left")

    figs = {"figS_transformation_uncertainty/figS_transformation_uncertainty": manuscript_grid([pA, pB, pC, pD, pE, pF], ncols=3)}
    return figs


# --------------------------------------------------------------------------- #
def fig_eye_reliability(D):
    rows = [r for r in D["rows"] if r["param"] == "eye_wide"]
    if not rows:
        return {}
    cur = D["curves"]
    key = [k for k in cur if k.startswith("eye_wide_") and k.endswith("_w")]
    pref = key[0][:-2] if key else None

    def pA(ax):
        panel_axes(ax)
        if pref is None:
            return
        w, post, disp, s2e = cur[pref + "_w"], cur[pref + "_post"], cur[pref + "_disparity"], cur[pref + "_sig2_eye"]
        q = np.quantile(s2e, [0, 1 / 3, 2 / 3, 1])
        cols = ["#1b9e77", "#d95f02", "#7570b3"]
        edges = np.arange(-31, 32, 2.0)
        for j in range(3):
            m = (s2e >= q[j]) & (s2e <= q[j + 1])
            c, mn, _ = _bins_mean(disp[m], w[m], edges)
            co, mo, _ = _bins_mean(disp[m], post[m], edges)
            ax.plot(co, mo, "--", lw=0.9, color=cols[j], alpha=0.7)
            ax.plot(c, mn, "o-", ms=PANEL_MS - 0.8, lw=PANEL_LW, color=cols[j], label=f"eye noise sd {np.sqrt(q[j]):.1f}–{np.sqrt(q[j + 1]):.1f}°")
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        ax.set_xlim(-31, 31); ax.set_ylim(-0.05, 1.2)
        finish_panel(ax, "body-frame disparity (deg)", "implied fusion weight", "weight by eye reliability", None)
        ax.legend(fontsize=6.5, loc="upper right", handlelength=1.4)
        ax.text(0.02, 0.05, "dashed: observer", transform=ax.transAxes, fontsize=7, color="grey")

    def pB(ax):
        panel_axes(ax)
        x = np.arange(3)
        for i, r in enumerate(rows):
            nb = json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["eyerel"]["by_sig2_eye_tercile"]
            cn = [t["crossing_net"] for t in nb]; co = [t["crossing_R0"] for t in nb]
            ax.plot(x + 0.08 * (i - 1), cn, "o", ms=PANEL_MS, color=C_NET, label="network" if i == 0 else None)
            ax.plot(x + 0.08 * (i - 1), co, "s", ms=PANEL_MS, color=C_R0, label="observer" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(["low", "mid", "high"]); ax.set_xlim(-0.5, 2.5)
        finish_panel(ax, "eye-signal noise tercile", "fusion window (deg)", "window vs eye reliability", "upper left")

    def pC(ax):
        panel_axes(ax)
        for i, r in enumerate(rows):
            bins = json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["eyerel"]["bins"]
            x = np.array([0.5 * (q["d_lo"] + q["d_hi"]) for q in bins])
            ax.plot(x, [q["slope_post_on_sig2eye"] for q in bins], "s--", ms=PANEL_MS - 0.5, lw=0.9, color=C_R0, alpha=0.8, label="observer" if i == 0 else None)
            ax.plot(x, [q["slope_w_on_sig2eye"] for q in bins], "o-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, alpha=0.8, label="network" if i == 0 else None)
        ax.axhline(0.0, lw=0.6, color="grey")
        ax.set_xlim(0, 18)
        finish_panel(ax, "|body-frame disparity| (deg)", "∂w / ∂σ²_eye (per deg²)", "weight vs eye noise, by disparity", "lower right")

    return {"figS_eye_reliability/figS_eye_reliability": manuscript_grid([pA, pB, pC], ncols=3)}


# --------------------------------------------------------------------------- #
def fig_error(D):
    S, F, fc = D["seeds"], D["flag"], D["fcurves"]
    if F is None:
        return {}
    agg = S["aggregated"] if S else {}

    def g(k):
        a = agg.get(k)
        return (a["mean"], a["sd"]) if a and a["mean"] is not None else (np.nan, np.nan)

    def pA(ax):
        panel_axes(ax)
        rows = F["terr"]["fusion_true_by_abs_terr"][:4]
        x = np.arange(len(rows))
        for key, col, lab, mk in (("fusion_true_R1", C_R1, "true-eye observer R1", "s"), ("fusion_true_R0", C_R0, "own observer R0", "s"), ("fusion_true_net", C_NET, "network", "o")):
            y = np.array([g(f"terr.{key.replace('fusion_true', 'fusion_true')}_{r['lo']:g}_{r['hi']:g}")[0] for r in rows])
            e = np.array([g(f"terr.{key}_{r['lo']:g}_{r['hi']:g}")[1] for r in rows])
            if np.all(np.isnan(y)):
                y = np.array([r[key] for r in rows]); e = np.zeros(len(rows))
            ax.errorbar(x, y, yerr=e, fmt=mk, ls="--" if key != "fusion_true_net" else "-", ms=PANEL_MS, lw=PANEL_LW, color=col, capsize=2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['lo']:g}–{r['hi']:g}" for r in rows]); ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(0.4, 0.9)
        finish_panel(ax, "|eye-estimate error| (deg)", "fusion of true common-cause trials", "natural eye-estimate error", "lower left")

    def pB(ax):
        panel_axes(ax)
        lv = F["eps"]["levels"]
        x = np.array([r["eps"] for r in lv])
        for key, col, lab, mk, ls in (("fusion_true_R0eps", C_R0, "observer, same error", "s", "--"), ("fusion_true_net", C_NET, "network", "o", "-")):
            y = np.array([g(f"eps.{key}_{e:+g}")[0] for e in x]); sd = np.array([g(f"eps.{key}_{e:+g}")[1] for e in x])
            if np.all(np.isnan(y)):
                y = np.array([r[key] for r in lv]); sd = np.zeros(len(x))
            ax.errorbar(x, y, yerr=sd, fmt=mk, ls=ls, ms=PANEL_MS, lw=PANEL_LW, color=col, capsize=2, label=lab)
        for key, col, mk in (("accuracy_R0eps", C_R0, "s"), ("accuracy_net", C_NET, "o")):
            y = np.array([g(f"eps.{key}_{e:+g}")[0] for e in x])
            if np.all(np.isnan(y)):
                y = np.array([r[key] for r in lv])
            ax.plot(x, y, mk + (":" if "R0" in key else "-."), ms=PANEL_MS - 0.6, lw=0.9, color=col, alpha=0.8, label="accuracy, " + ("observer" if "R0" in key else "network"))
        ax.set_ylim(0, 1.0)
        finish_panel(ax, "injected eye-signal error ε (deg)", "fusion (common-cause trials) / accuracy", "injected eye error", None)
        ax.legend(fontsize=6, loc="upper right", ncol=1, handlelength=1.2, borderaxespad=0.2)

    def pC(ax):
        panel_axes(ax)
        cols = {-10.0: "#d95f02", 0.0: C_NET, 10.0: "#7570b3"}
        for r in F["eps"]["levels"]:
            if r["eps"] not in cols:
                continue
            cn = np.array([[0.5 * (q["lo"] + q["hi"]), q["mean"]] for q in r["curve_net"]], float)
            co = np.array([[0.5 * (q["lo"] + q["hi"]), q["mean"]] for q in r["curve_R0eps"]], float)
            ax.plot(co[:, 0], co[:, 1], "--", lw=0.9, color=cols[r["eps"]], alpha=0.7)
            ax.plot(cn[:, 0], cn[:, 1], "o-", ms=PANEL_MS - 1, lw=PANEL_LW, color=cols[r["eps"]], label=f"ε = {r['eps']:+g}°")
        ax.axhline(0.5, lw=0.6, ls=":", color="grey")
        ax.set_xlim(-31, 31); ax.set_ylim(-0.05, 1.2)
        finish_panel(ax, "true disparity s_vis − s_prop (deg)", "implied fusion weight", "fusion moves to −k·ε", None)
        ax.legend(fontsize=7, loc="upper right", handlelength=1.4)
        ax.text(0.02, 0.93, "dashed: observer, same signal", transform=ax.transAxes, fontsize=6.5, color="grey")

    def pD(ax):
        panel_axes(ax)
        lv = F["eps"]["levels"]
        x = np.array([r["eps"] for r in lv])
        pn = np.array([g(f"eps.peak_w_vs_true_disparity_net_{e:+g}")[0] for e in x]); sn = np.array([g(f"eps.peak_w_vs_true_disparity_net_{e:+g}")[1] for e in x])
        po = np.array([g(f"eps.peak_w_vs_true_disparity_R0eps_{e:+g}")[0] for e in x])
        if np.all(np.isnan(pn)):
            pn = np.array([r["peak_w_vs_true_disparity_net"] for r in lv]); sn = np.zeros(len(x)); po = np.array([r["peak_w_vs_true_disparity_R0eps"] for r in lv])
        ax.plot(x, -F["eps"]["k"] * x, ls=":", lw=0.9, color="grey", label=f"−k·ε (k = {F['eps']['k']:.2f})")
        ax.plot(x, po, "s--", ms=PANEL_MS, lw=PANEL_LW, color=C_R0, label="observer, same error")
        ax.errorbar(x, pn, yerr=sn, fmt="o-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, capsize=2, label="network")
        finish_panel(ax, "injected eye-signal error ε (deg)", "location of maximal fusion (deg)", "error read as displacement", "upper right")

    def pE(ax):
        panel_axes(ax)
        rows = F["beta"]["levels"]
        cols = {0.5: "#d95f02", 0.75: "#e6ab02", 1.0: C_NET, 1.25: "#7570b3"}
        for r in rows:
            ecc = r["by_eye_eccentricity"]
            x = np.arange(len(ecc))
            ax.plot(x, [q["accuracy_R0beta"] for q in ecc], "s--", ms=PANEL_MS - 0.5, lw=0.9, color=cols[r["beta"]], alpha=0.7)
            ax.plot(x, [q["accuracy_net"] for q in ecc], "o-", ms=PANEL_MS, lw=PANEL_LW, color=cols[r["beta"]], label=f"gain β = {r['beta']:g}")
        ecc = rows[0]["by_eye_eccentricity"]
        ax.set_xticks(np.arange(len(ecc))); ax.set_xticklabels([f"{q['lo']:g}–{min(q['hi'], 30):g}" for q in ecc], fontsize=7)
        ax.set_ylim(0.45, 0.9)
        finish_panel(ax, "|eye position| (deg)", "causal decision correct", "mis-gained eye signal", None)
        ax.legend(fontsize=6.5, loc="lower left", ncol=2, handlelength=1.4, columnspacing=0.6)

    def pF(ax):
        panel_axes(ax)
        lv = _levels(D["agg"], "eye_noise")
        if lv:
            x = [a["basic.k"] for a in lv]
            sl_n, sl_o = [], []
            for a in lv:
                eps = np.array([-15, -10, -5, -2, 0, 2, 5, 10, 15], float)
                pn = np.array([a.get(f"eps.peak_w_vs_true_disparity_net_{e:+g}") or np.nan for e in eps])
                po = np.array([a.get(f"eps.peak_w_vs_true_disparity_R0eps_{e:+g}") or np.nan for e in eps])
                ok = np.isfinite(pn) & (np.abs(eps) <= 10)
                sl_n.append(-np.polyfit(eps[ok], pn[ok], 1)[0] if ok.sum() > 2 else np.nan)
                ok = np.isfinite(po) & (np.abs(eps) <= 10)
                sl_o.append(-np.polyfit(eps[ok], po[ok], 1)[0] if ok.sum() > 2 else np.nan)
            ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="grey", label="identity")
            ax.plot(x, sl_o, "s--", ms=PANEL_MS, lw=PANEL_LW, color=C_R0, label="observer")
            ax.plot(x, sl_n, "o-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, label="network")
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.1)
        finish_panel(ax, "observer's eye gain k", "displacement per deg of error", "impact scales with k", "upper left")

    return {"figS_transformation_error/figS_transformation_error": manuscript_grid([pA, pB, pC, pD, pE, pF], ncols=3)}


# --------------------------------------------------------------------------- #
def fig_frame(D):
    F, fc = D["flag"], D["fcurves"]
    if F is None or fc is None:
        return {}
    lv = _levels(D["agg"], "eye_sd")
    S = D["seeds"]

    def pA(ax):
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
        ax.text(0.02, 0.93, "dashed: observer", transform=ax.transAxes, fontsize=7, color="grey")

    def pB(ax):
        panel_axes(ax)
        if lv:
            _series(ax, lv, "frame.crossing_body_net", C_NET, "body frame, network")
            _series(ax, lv, "frame.crossing_retinal_net", C_VIS, "retinal, network", marker="^")
            _series(ax, lv, "frame.crossing_body_R0", C_R0, None, marker="s", ls="--")
            _series(ax, lv, "frame.crossing_retinal_R0", "#8c8c8c", None, marker="s", ls="--")
            _xcat(ax, lv, "eye-position sd (deg)")
        finish_panel(ax, "eye-position sd (deg)", "fusion window (deg)", "window in each frame", "upper left")
        ax.text(0.98, 0.05, "dashed: observer; no point = no crossing", transform=ax.transAxes, ha="right", fontsize=6, color="grey")

    def pC(ax):
        panel_axes(ax)
        if lv:
            _series(ax, lv, "frame.lookup_r2_body_net", C_NET, "from |body disparity|")
            _series(ax, lv, "frame.lookup_r2_retinal_net", C_VIS, "from |retinal disparity|", marker="^")
            _series(ax, lv, "frame.lookup_r2_body_R0", C_R0, None, marker="s", ls="--")
            _series(ax, lv, "frame.lookup_r2_retinal_R0", "#8c8c8c", None, marker="s", ls="--")
            _xcat(ax, lv, "eye-position sd (deg)")
        ax.set_ylim(0, 1.05)
        finish_panel(ax, "eye-position sd (deg)", "R² of the weight", "R² from each disparity", "lower left")
        ax.text(0.98, 0.5, "dashed: observer", transform=ax.transAxes, ha="right", fontsize=7, color="grey")

    def pD(ax):
        panel_axes(ax)
        if lv:
            _series(ax, lv, "partials.leak_d_muprop_dxeye_segregated_median", COLORS["eye"], "∂μ_hand/∂x_eye, segregated (observer: 0)")
            _series(ax, lv, "partials.shared_error_muprop_on_terr_segregated", C_PROP, "hand error on eye error (observer: 0)", marker="v")
            _series(ax, lv, "partials.shared_error_muvis_on_terr_segregated", C_VIS, "visual error on eye error", marker="^")
            _series(ax, lv, "partials.shared_error_muvis_on_terr_observer", "#8c8c8c", "observer's visual value", marker="s", ls="--")
            _xcat(ax, lv, "eye-position sd (deg)")
        ax.axhline(0.0, lw=0.6, ls=":", color="grey")
        ax.set_ylim(-0.3, 1.15)
        finish_panel(ax, "eye-position sd (deg)", "deg per deg", "where the eye signal goes", None)
        ax.legend(fontsize=6, loc="center left", handlelength=1.4)

    return {"figS_decision_frame/figS_decision_frame": manuscript_grid([pA, pB, pC, pD], ncols=2)}


# --------------------------------------------------------------------------- #
def fig_eye_gain(D):
    rows = D["rows"]
    if not rows:
        return {}
    sel = [r for r in rows if r["param"] in ("eye_sd", "eye_noise")]

    def pA(ax):
        panel_axes(ax)
        ax.plot([0, 1.05], [0, 1.05], ls=":", lw=0.8, color="grey", label="gain = k (Bayes)")
        ax.axhline(1.0, lw=0.8, ls="--", color="#8c8c8c", label="gain = 1 (raw sum)")
        for param, col, mk, lab in (("eye_sd", C_NET, "o", "eye-spread levels"), ("eye_noise", COLORS["eye"], "^", "eye-noise levels")):
            rr = [r for r in sel if r["param"] == param]
            x = [r["partials.eye_gain_k_observer"] for r in rr]; y = [r["partials.eye_gain_net_median"] for r in rr]
            ax.plot(x, y, mk, ms=PANEL_MS + 0.5, color=col, alpha=0.8, label=lab)
        ax.set_xlim(-0.03, 1.08); ax.set_ylim(-0.03, 1.15)
        finish_panel(ax, "observer's eye gain k", "network's effective eye gain", "effective eye gain vs k", "upper left")

    def pB(ax):
        panel_axes(ax)
        for param, col, mk, lab in (("eye_sd", C_NET, "o", "eye-spread levels"), ("eye_noise", COLORS["eye"], "^", "eye-noise levels")):
            rr = [r for r in sel if r["param"] == param]
            x = [r["partials.eye_gain_k_observer"] for r in rr]; y = [r["terr.residual_on_xeye"] for r in rr]
            ax.plot(x, y, mk, ms=PANEL_MS + 0.5, color=col, alpha=0.8, label=lab)
        ax.axhline(0.0, lw=0.6, ls=":", color="grey")
        finish_panel(ax, "observer's eye gain k", "coefficient of (w − p) on x_eye", "residual weight on x_eye", "upper right")

    def pC(ax):
        panel_axes(ax)
        lv = _levels(D["agg"], "eye_sd")
        if lv:
            _series(ax, lv, "beta.bias_muvis_on_eye_R0beta_1", C_R0, "observer (−(1−k)·weight)", marker="s", ls="--")
            _series(ax, lv, "beta.bias_muvis_on_eye_net_1", C_NET, "network")
            _xcat(ax, lv, "eye-position sd (deg)")
        ax.axhline(0.0, lw=0.6, ls=":", color="grey")
        finish_panel(ax, "eye-position sd (deg)", "bias of μ_vis per deg of eye", "gaze-dependent bias", "lower right")

    return {"figS_eye_gain/figS_eye_gain": manuscript_grid([pA, pB, pC], ncols=3)}


# --------------------------------------------------------------------------- #
def fig_cost(D):
    agg = D["agg"]
    if not agg:
        return {}
    ln, ls_ = _levels(agg, "eye_noise"), _levels(agg, "eye_sd")

    def gaps(ax, lv, xl):
        panel_axes(ax)
        if lv:
            d1 = [a["R0.accuracy_R0"] - a["R0.accuracy_net"] for a in lv]
            d2 = [a["R1.accuracy_R1"] - a["R0.accuracy_R0"] for a in lv]
            x = np.arange(len(lv))
            ax.bar(x - 0.18, d2, width=0.36, color=C_R0, label="R1 − R0 (Bayes)")
            ax.bar(x + 0.18, d1, width=0.36, color=C_NET, label="R0 − network")
            _xcat(ax, lv, xl)
        ax.axhline(0, lw=0.6, color="grey")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
        finish_panel(ax, xl, "loss of causal accuracy", "loss of causal accuracy", "upper left")

    def pA(ax):
        gaps(ax, ln, "eye-signal noise sd (deg)")

    def pB(ax):
        gaps(ax, ls_, "eye-position sd (deg)")

    def pC(ax):
        panel_axes(ax)
        if ls_:
            _series(ax, ls_, "R0.rmse_mu_vis", C_VIS, "RMSE μ_vis (deg)")
            _series(ax, ls_, "R0.rmse_mu_prop", C_PROP, "RMSE μ_hand (deg)", marker="s")
            _series(ax, ls_, "R0.weight_vs_post_slope", C_NET, "weight-on-posterior slope", marker="^", ls="--")
            _xcat(ax, ls_, "eye-position sd (deg)")
        ax.set_ylim(0.3, 1.1)
        finish_panel(ax, "eye-position sd (deg)", "deg / slope", "precision vs inference", None)
        ax.legend(fontsize=6, loc="lower right", handlelength=1.2)

    def pD(ax):
        panel_axes(ax)
        if ls_:
            for c in ("congruent", "opposite", "mixed"):
                _series(ax, ls_, f"units.n_{c}", CLS[c], c)
                _series(ax, ls_, f"twin.n_{c}", CLS[c], None, marker="s", ls="--")
            ax.plot([], [], "s--", color="#555555", ms=PANEL_MS, lw=PANEL_LW, label="always-fuse twin")
            _xcat(ax, ls_, "eye-position sd (deg)")
        ax.set_ylim(0, 70)
        finish_panel(ax, "eye-position sd (deg)", "units of 64", "MSL unit classes", None)
        ax.legend(fontsize=6.5, loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.6)

    def pE(ax):
        panel_axes(ax)
        if ls_:
            _series(ax, ls_, "units.di_share_body", C_NET, "causal network")
            _series(ax, ls_, "twin.di_share_body", C_TWIN, "always-fuse twin", marker="s", ls="--")
            _xcat(ax, ls_, "eye-position sd (deg)")
        ax.set_ylim(0, 1)
        finish_panel(ax, "eye-position sd (deg)", "share of body-frame MSL units", "body-frame MSL units", "upper left")

    def pF(ax):
        panel_axes(ax)
        if ln:
            for c in ("congruent", "opposite", "mixed"):
                _series(ax, ln, f"units.n_{c}", CLS[c], c)
                _series(ax, ln, f"twin.n_{c}", CLS[c], None, marker="s", ls="--")
            _xcat(ax, ln, "eye-signal noise sd (deg)")
        ax.set_ylim(0, 70)
        finish_panel(ax, "eye-signal noise sd (deg)", "units of 64", "unit classes vs eye noise", None)

    return {"figS_transformation_cost/figS_transformation_cost": manuscript_grid([pA, pB, pC, pD, pE, pF], ncols=3)}


# --------------------------------------------------------------------------- #
def fig_eccentricity(D):
    """Channel A, second lever: eccentricity-dependent eye noise (sig2_eye + c*eye^2)."""
    rows = [r for r in D["rows"] if r["param"] in ("eye_ecc", "eye_sd")]
    ecc = {}
    for r in rows:
        c = r.get("ecc_c", 0.0) if r["param"] == "eye_ecc" else (0.0 if r["level"] == 9.0 else None)
        if c is None:
            continue
        ecc.setdefault(c, []).append(json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["ecc"]["by_abs_eye"])
    if not ecc:
        return {}
    cols = {0.0: C_NET, 0.01: "#1b9e77", 0.03: "#d95f02", 0.1: "#7570b3"}

    def mean_over_seeds(lists, key):
        n = min(len(l) for l in lists)
        vals = [[np.nan if l[i][key] is None else l[i][key] for l in lists] for i in range(n)]
        return np.array([np.nanmean(v) for v in vals]), np.array([np.nanstd(v) for v in vals])

    def panel(ax, key_net, key_obs, ylabel, title, ylim=None):
        panel_axes(ax)
        for c in sorted(ecc):
            lists = ecc[c]
            n = min(len(l) for l in lists)
            x = np.arange(n)
            mn, sn = mean_over_seeds(lists, key_net)
            mo, _ = mean_over_seeds(lists, key_obs)
            ax.plot(x, mo, "s--", ms=PANEL_MS - 0.5, lw=0.9, color=cols.get(c, "k"), alpha=0.7)
            ax.errorbar(x, mn, yerr=sn, fmt="o-", ms=PANEL_MS, lw=PANEL_LW, color=cols.get(c, "k"), capsize=1.5, label=f"c = {c:g}")
        l0 = ecc[sorted(ecc)[0]][0]
        ax.set_xticks(np.arange(len(l0))); ax.set_xticklabels([f"{q['lo']:g}–{min(q['hi'], 30):g}" for q in l0], fontsize=7)
        if ylim:
            ax.set_ylim(*ylim)
        finish_panel(ax, "|eye position| (deg)", ylabel, title, None)
        ax.legend(fontsize=6.5, loc="best", handlelength=1.4)

    def pA(ax):
        panel(ax, "crossing_net", "crossing_R0", "fusion window (deg)", "window vs eccentricity")
        ax.text(0.02, 0.93, "dashed: observer", transform=ax.transAxes, fontsize=7, color="grey")

    def pB(ax):
        panel(ax, "fusion_net", "fusion_R0", "fusion (|d| ≤ 2°)", "fusion vs eccentricity", (0.3, 1.0))

    def pC(ax):
        panel(ax, "accuracy_net", "accuracy_R0", "causal decision correct", "accuracy vs eccentricity", (0.5, 0.95))

    return {"figS_eye_eccentricity/figS_eye_eccentricity": manuscript_grid([pA, pB, pC], ncols=3)}


# --------------------------------------------------------------------------- #
def fig_units_frame(D):
    """Finding 8: the opposite units hold vision in the observer's frame. A unit
    tuned to x_vis + k x_eye shifts with the eye by 1 - k, so the expected
    displacement index of a unit in the observer's frame is 1 - k, not 0."""
    FR = OUT / "levels" / "frames_by_k.json"
    if not FR.exists():
        return {}
    fr = json.loads(FR.read_text())
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
            pts[param].append((level, mean("expected_di", recs), mean("di_median_opposite", recs), sd("di_median_opposite", recs),
                               mean("twin_di_median", recs), sd("twin_di_median", recs),
                               mean("bayes_frame_share", recs), sd("bayes_frame_share", recs),
                               mean("twin_bayes_frame_share", recs), sd("twin_bayes_frame_share", recs)))

    def pA(ax):
        panel_axes(ax)
        ax.plot([-0.05, 1.05], [-0.05, 1.05], ls=":", lw=0.8, color="grey", label="observer's frame (DI = 1 − k)")
        ax.axhline(1.0, lw=0.8, ls="--", color="#8c8c8c", label="retinal (DI = 1)")
        for param, mk, lab in (("eye_noise", "^", "eye-noise levels"), ("eye_sd", "o", "eye-spread levels")):
            P = np.array(pts[param])
            ax.errorbar(P[:, 1], P[:, 2], yerr=P[:, 3], fmt=mk, ms=PANEL_MS + 0.5, color=CLS["opposite"], capsize=1.5, label=f"opposite units, {lab}")
            ax.errorbar(P[:, 1], P[:, 4], yerr=P[:, 5], fmt=mk, ms=PANEL_MS + 0.5, color=C_TWIN, capsize=1.5, mfc="none", label=f"always-fuse twin, {lab}")
        ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.1, 1.2)
        finish_panel(ax, "expected shift 1 − k", "displacement index (median)", "opposite units vs observer's frame", None)
        ax.legend(fontsize=5.5, loc="lower right", handlelength=1.2, borderaxespad=0.2)

    def series_levels(ax, param, xl):
        P = np.array(pts[param])
        x = np.arange(len(P))
        ax.errorbar(x - 0.08, P[:, 6], yerr=P[:, 7], fmt="o-", ms=PANEL_MS, lw=PANEL_LW, color=C_NET, capsize=1.5, label="causal network")
        ax.errorbar(x + 0.08, P[:, 8], yerr=P[:, 9], fmt="s--", ms=PANEL_MS, lw=PANEL_LW, color=C_TWIN, capsize=1.5, label="always-fuse twin")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in P[:, 0]]); ax.set_xlim(-0.5, len(P) - 0.5)
        ax.set_ylim(0, 1.05)
        finish_panel(ax, xl, "share within 0.2 of 1 − k", "MSL units in the observer's frame", "upper left")

    def pB(ax):
        panel_axes(ax)
        series_levels(ax, "eye_noise", "eye-signal noise sd (deg)")

    def pC(ax):
        panel_axes(ax)
        series_levels(ax, "eye_sd", "eye-position sd (deg)")

    return {"figS_units_frame/figS_units_frame": manuscript_grid([pA, pB, pC], ncols=3)}


def main():
    apply_style()
    D = _load()
    figs = {}
    for fn in (fig_uncertainty, fig_eye_reliability, fig_error, fig_frame, fig_eye_gain, fig_cost, fig_eccentricity, fig_units_frame):
        try:
            figs.update(fn(D))
        except Exception as ex:  # a figure whose data are not there yet
            print(f"  {fn.__name__}: skipped ({type(ex).__name__}: {ex})")
    FIGS.mkdir(parents=True, exist_ok=True)
    save_figures(figs, FIGS, formats=("png", "tif", "svg"))
    print(f"wrote {len(figs)} figures to {FIGS}")


if __name__ == "__main__":
    main()
