"""Neuronal-level analyses for the manuscript's layers-and-units subsection,
as figures under results/neuronal_level_analysis/<folder>/ with a numbers.json
each. Torch-free (npmodel.py); house style (cmsi.viz.style / manuscript);
png, tif and svg, no pdf.

    python scripts/neuronal/run.py units    Fig. units: example tuning curves (R1) and unit
                                            counts against the prior and the targets (R11), 5 panels
    python scripts/neuronal/run.py r6       decision-conditioned bias at the midpoint (Rideaux 2E-G)
    python scripts/neuronal/run.py r5       lesion on causal behaviour, four designs compared
    python scripts/neuronal/run.py all

Networks: pcommon0 (0), pcommon03 (0.3), flagship (0.5), pcommon07 (0.7),
pcommon1 (1), and the always-fuse twins of the three causal worlds. The
p = 0.28 run is not used.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_COLORS, CLASSES, COLORS, OUT, PANEL_FONT, PANEL_LW,   # noqa: E402
                    PANEL_MS, RUN_LABEL, RUN_PRIOR, TWIN_RUNS, Bundle, apply_style,
                    behaviour_stats, binned_weight, bootstrap_ci, congruency_np,
                    decode, finish_panel, fusion_segregation, hybrid_reads, load_run,
                    manuscript_grid, mean_by_bin, panel_axes, save_figures, save_json)
from cmsi.viz.style import FORMATS  # noqa: E402  (png, tif, svg)

_BUNDLES, _TWINS = {}, {}


def bundle(run):
    if run not in _BUNDLES:
        _BUNDLES[run] = Bundle(run)
    return _BUNDLES[run]


def twin(run):
    """(net, cfg, encoders, congruency, test split) of the always-fuse twin of `run`."""
    if run not in _TWINS:
        net, ckpt, cfg, d, d_full = load_run(f"{run}_twin")
        cg = congruency_np(net, d_full["encoders"], cfg)
        _TWINS[run] = {"net": net, "cfg": cfg, "cong": cg, "d": d,
                       "counts": {k: cg[k] for k in ("n_congruent", "n_opposite", "n_mixed", "n_untuned")}}
    return _TWINS[run]


def class_sets(classes):
    return {c: np.flatnonzero(classes == c) for c in CLASSES}


# --------------------------------------------------------------------------- #
# Fig. units  (R1 + R11): A, B example units; C class distribution by network;
#                         D counts against the prior; E opposite units, causal vs twin
# --------------------------------------------------------------------------- #
def units():
    b = bundle("flagship")
    c = b.cong
    idx = c["index"]
    u_con = int(np.argmax(np.where(c["classes"] == "congruent", idx, -np.inf)))
    u_opp = int(np.argmin(np.where(c["classes"] == "opposite", idx, np.inf)))

    rows = []
    for run, p in sorted(RUN_PRIOR.items(), key=lambda kv: kv[1]):
        bb = bundle(run)
        rows.append({"run": run, "p_common": p, **bb.counts, "counts_match_metrics": bb.counts_match})
    twins = [{"run": f"{run}_twin", "p_common": RUN_PRIOR[run], **twin(run)["counts"]} for run in TWIN_RUNS]

    def example(u, title):
        def draw(ax=None):
            ax = panel_axes(ax)
            ax.plot(c["positions"], c["resp_vis"][:, u], "-", lw=PANEL_LW, color=COLORS["visual"],
                    label="visual sweep")
            ax.plot(c["positions"], c["resp_prop"][:, u], "-", lw=PANEL_LW, color=COLORS["prop"],
                    label="hand sweep")
            ax.set_ylim(-0.05, 1.3)
            ax.set_yticks([0, 0.5, 1.0])
            return finish_panel(ax, "cue position (deg)", "MSL unit activation",
                                f"{title} (unit {u}, index {idx[u]:+.2f})", "upper right")
        return draw

    def pC(ax=None):
        ax = panel_axes(ax)
        order = [("flagship", bundle("flagship").counts, "main"),
                 ("flagship_twin", twin("flagship")["counts"], "twin"),
                 ("pcommon1", bundle("pcommon1").counts, "fuse ctrl"),
                 ("pcommon0", bundle("pcommon0").counts, "segregate ctrl")]
        x = np.arange(len(order))
        wdt = 0.26
        for i, cl in enumerate(CLASSES):
            ax.bar(x + (i - 1) * wdt, [o[1][f"n_{cl}"] for o in order], wdt,
                   color=CLASS_COLORS[cl], alpha=0.85, label=cl)
        ax.set_xticks(x)
        ax.set_xticklabels(["main", "twin", "fuse\nctrl", "segregate\nctrl"])
        ax.set_ylim(0, 58)
        return finish_panel(ax, "network", "units of 64", "unit classes by network", "upper left")

    def pD(ax=None):
        ax = panel_axes(ax)
        priors = [r["p_common"] for r in rows]
        for cl in CLASSES:
            ax.plot(priors, [r[f"n_{cl}"] for r in rows], "o",
                    ms=PANEL_MS + 1, color=CLASS_COLORS[cl], label=cl)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(priors)
        ax.set_xticklabels([f"{p:g}" for p in priors])
        ax.set_ylim(0, 66)
        fig = finish_panel(ax, "prior on a common cause", "units of 64",
                           "unit classes against the prior", None)
        ax.legend(fontsize=PANEL_FONT["legend"], loc="upper left", ncol=3, handlelength=1.2,
                  columnspacing=0.8, borderaxespad=0.2)
        return fig

    def pE(ax=None):
        ax = panel_axes(ax)
        pairs = [(r, t) for r in rows for t in twins if np.isclose(r["p_common"], t["p_common"])]
        x = np.arange(len(pairs))
        wdt = 0.36
        ax.bar(x - wdt / 2, [r["n_opposite"] for r, _ in pairs], wdt, color=CLASS_COLORS["opposite"],
               label="causal network")
        ax.bar(x + wdt / 2, [t["n_opposite"] for _, t in pairs], wdt, color=CLASS_COLORS["opposite"],
               alpha=0.35, label="always-fuse twin")
        ctrl1 = [r for r in rows if r["p_common"] == 1.0]
        if ctrl1:
            ax.axhline(ctrl1[0]["n_opposite"], ls=":", lw=0.8, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['p_common']:g}" for r, _ in pairs])
        ax.set_ylim(0, 40)
        return finish_panel(ax, "prior on a common cause", "opposite units of 64",
                            "opposite units: causal vs fused targets", "upper right")

    draws = [example(u_con, "congruent unit"), example(u_opp, "opposite unit"), pC, pD, pE]
    figs = {"Fig_units/row_ABCDE": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_congruent_unit", "B_opposite_unit", "C_classes_by_network",
                        "D_counts_vs_prior", "E_opposite_causal_vs_twin"), draws):
        figs[f"Fig_units/{key}"] = fn()
    save_figures(figs, OUT, formats=FORMATS)
    save_json({"example_units": {"congruent": u_con, "opposite": u_opp},
               "index": idx.tolist(), "classes": c["classes"].tolist(),
               "runs": rows, "twins": twins}, OUT / "Fig_units" / "numbers.json")
    print(f"units: example congruent unit {u_con} ({idx[u_con]:+.2f}), opposite unit {u_opp} ({idx[u_opp]:+.2f})")
    for r in rows + twins:
        print(f"   {r['run']:16s} p={r['p_common']:<4g} con {r['n_congruent']:2d} opp {r['n_opposite']:2d} "
              f"mixed {r['n_mixed']:2d} untuned {r['n_untuned']}")


# --------------------------------------------------------------------------- #
# R6  decision-conditioned bias at the transition midpoint
# --------------------------------------------------------------------------- #
def r6(run="flagship", n=4000, seed=0):
    from cmsi.data.generative import observer
    from common import encode, hybrid_weight, mid_sigmas
    b = bundle(run)
    gen, enc, encoders = b.cfg["generative"], b.cfg["encoding"], b.d_full["encoders"]
    d0 = b.metrics["transition"]["network"]["midpoint_deg"]
    D_list = [2.5, round(d0, 2), 7.5]
    sig2 = mid_sigmas(gen)
    rng = np.random.default_rng(seed)
    i_prop, i_vis = b.idx("mu_prop"), b.idx("mu_vis")
    i_vp, i_vv = b.idx("var_prop"), b.idx("var_vis")
    res = {"run": run, "midpoint": d0, "conditions": []}
    hist_data = None
    for D in D_list:
        pool = {"pull": [], "bayes_pull": [], "meas_disp": [], "one": [], "post": [], "w": []}
        for sign in (1.0, -1.0):
            d = {"x_vis": rng.normal(sign * D, np.sqrt(sig2["vis"]), n),
                 "x_eye": rng.normal(0.0, np.sqrt(sig2["eye"]), n),
                 "x_prop": rng.normal(0.0, np.sqrt(sig2["prop"]), n),
                 "sig2_vis": np.full(n, sig2["vis"]), "sig2_prop": np.full(n, sig2["prop"]),
                 "sig2_eye": np.full(n, sig2["eye"])}
            obs = observer(d, gen)
            X = encode(d, encoders, enc, rng)
            pred = b.net.predict(X)
            dv = obs["fused_mu"] - obs["seg_vis_mu"]
            w, _, _ = hybrid_weight(pred[:, i_vv], obs["fused_var"], obs["seg_vis_var"], dv,
                                    pred[:, i_vis], obs["seg_vis_mu"], b.sig_out[i_vis], b.sig_out[i_vv])
            pool["pull"].append(sign * (pred[:, i_prop] - obs["seg_prop_mu"]))
            pool["bayes_pull"].append(sign * obs["post_c1"] * (obs["fused_mu"] - obs["seg_prop_mu"]))
            pool["meas_disp"].append(sign * obs["disparity"])
            pool["one"].append(np.where(np.isfinite(w), w, obs["post_c1"]) > 0.5)
            pool["post"].append(obs["post_c1"])
            pool["w"].append(w)
        pool = {k: np.concatenate(v) for k, v in pool.items()}
        one = pool["one"]
        cond = {"D": float(D), "n": int(len(one)), "frac_one": float(one.mean())}
        for g, mask in (("one", one), ("two", ~one)):
            cond[g] = {"n": int(mask.sum()),
                       "pull": float(pool["pull"][mask].mean()), "pull_ci": bootstrap_ci(pool["pull"][mask]),
                       "bayes_pull": float(pool["bayes_pull"][mask].mean()),
                       "meas_disp": float(pool["meas_disp"][mask].mean()),
                       "meas_disp_ci": bootstrap_ci(pool["meas_disp"][mask]),
                       "post": float(pool["post"][mask].mean())}
        res["conditions"].append(cond)
        if D == D_list[1]:
            hist_data = pool
        print(f"R6 D={D:4.2f}: one-cause {cond['frac_one']:.2f}; pull one {cond['one']['pull']:+.2f} "
              f"[{cond['one']['pull_ci'][0]:+.2f},{cond['one']['pull_ci'][1]:+.2f}] two {cond['two']['pull']:+.2f} "
              f"[{cond['two']['pull_ci'][0]:+.2f},{cond['two']['pull_ci'][1]:+.2f}]; measured |d| one "
              f"{cond['one']['meas_disp']:.2f} two {cond['two']['meas_disp']:.2f}; Bayes pull one "
              f"{cond['one']['bayes_pull']:+.2f} two {cond['two']['bayes_pull']:+.2f}")
    save_json(res, OUT / "R6_decision_bias" / "numbers.json")

    def pA(ax=None):
        ax = panel_axes(ax)
        one = hist_data["one"]
        bins = np.linspace(-2, 3, 41)
        ax.hist(hist_data["pull"][one], bins, color=COLORS["network"], alpha=0.6, label="inferred one cause")
        ax.hist(hist_data["pull"][~one], bins, color=COLORS["visual"], alpha=0.6, label="inferred two causes")
        ax.axvline(0, lw=0.8, color="k", ls=":")
        return finish_panel(ax, "pull of hand estimate toward vision (deg)", "trials",
                            f"nominal disparity {d0:.2f}°", "upper right")

    def pB(ax=None):
        ax = panel_axes(ax)
        Ds = [c["D"] for c in res["conditions"]]
        for g, col, lab in (("one", COLORS["network"], "inferred one cause"),
                            ("two", COLORS["visual"], "inferred two causes")):
            m = [c[g]["meas_disp"] for c in res["conditions"]]
            lo = [c[g]["meas_disp_ci"][0] for c in res["conditions"]]
            hi = [c[g]["meas_disp_ci"][1] for c in res["conditions"]]
            ax.errorbar(Ds, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o-", lw=PANEL_LW,
                        ms=PANEL_MS, color=col, capsize=2, label=lab)
        ax.plot(Ds, Ds, ":", lw=0.8, color="gray", label="nominal")
        return finish_panel(ax, "nominal disparity (deg)", "measured disparity (deg)",
                            "the input behind each decision", "upper left")

    def pC(ax=None):
        ax = panel_axes(ax)
        Ds = [c["D"] for c in res["conditions"]]
        for g, col, lab in (("one", COLORS["network"], "one cause"), ("two", COLORS["visual"], "two causes")):
            m = [c[g]["pull"] for c in res["conditions"]]
            lo = [c[g]["pull_ci"][0] for c in res["conditions"]]
            hi = [c[g]["pull_ci"][1] for c in res["conditions"]]
            ax.errorbar(Ds, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o-", lw=PANEL_LW,
                        ms=PANEL_MS, color=col, capsize=2, label=f"network, {lab}")
            ax.plot(Ds, [c[g]["bayes_pull"] for c in res["conditions"]], "o--", lw=0.9, ms=PANEL_MS,
                    mfc="none", color=col, label=f"Bayes, {lab}")
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_ylim(0, 1.05)
        fig = finish_panel(ax, "nominal disparity (deg)", "pull toward vision (deg)",
                           "bias by inferred cause", None)
        ax.legend(fontsize=7, loc="lower left", ncol=1, handlelength=1.4, borderaxespad=0.2,
                  labelspacing=0.3)
        return fig

    figs = {"R6_decision_bias/row_ABC": manuscript_grid([pA, pB, pC], ncols=3)}
    save_figures(figs, OUT, formats=FORMATS)


# --------------------------------------------------------------------------- #
# R5  lesion on causal behaviour: four designs
#   v1  whole-class clamp, frozen read-out, seven statistics vs random null
#   v2  single-unit lesions ranked by congruency index (fusion / segregation)
#   v3  cumulative lesions, class order vs random orders (dose-response)
#   v4  whole-class clamp with the read-out refitted on the remaining units
# --------------------------------------------------------------------------- #
R5_STATS = ("rmse", "midpoint", "posreg_vis", "posreg_prop", "hump_vis", "hump_prop", "decode_msl")


def _fs(b, pred):
    w, _ = hybrid_reads(b, pred)
    return fusion_segregation(b.d["disparity"], w)


def r5_v1(b, n_random=100, seed=0):
    """Whole-class clamp against the manuscript's size-matched random draws."""
    X, ref = b.d["X"], b.msl.mean(0)
    grid = np.unique(np.abs(np.asarray(b.cfg["analysis"]["disparity_grid"], float)))
    ad = np.abs(b.d["disparity"])

    def curve(w):
        c, m, n, se = binned_weight(ad, w, grid, min_count=25)
        return {"c": c.tolist(), "m": m.tolist()}

    intact = behaviour_stats(b, b.pred, b.msl)
    f0, s0 = fusion_segregation(b.d["disparity"], intact["w"])
    out = {"intact": {**{k: intact[k] for k in R5_STATS}, "fusion": f0, "segregation": s0},
           "lesions": {}, "curves": {"intact": curve(intact["w"]), "grid": grid.tolist(), "random": {}}}
    rng = np.random.default_rng(seed)
    n_units, cache = len(b.classes), {}
    for label in CLASSES:
        idx = np.flatnonzero(b.classes == label)
        k = len(idx)
        if k == 0:
            continue
        keep = np.ones(n_units, bool)
        keep[idx] = False
        pred_l, h_l = b.net.lesion_predict(X, keep, ref)
        st = behaviour_stats(b, pred_l, h_l)
        fl, sl = fusion_segregation(b.d["disparity"], st["w"])
        out["curves"][label] = curve(st["w"])
        if k not in cache:
            draws, dcurves = [], []
            for _ in range(n_random):
                rkeep = np.ones(n_units, bool)
                rkeep[rng.choice(n_units, size=k, replace=False)] = False
                p_r, h_r = b.net.lesion_predict(X, rkeep, ref)
                s_r = behaviour_stats(b, p_r, h_r)
                fr, sr = fusion_segregation(b.d["disparity"], s_r["w"])
                draws.append({**{kk: s_r[kk] for kk in R5_STATS}, "fusion": fr, "segregation": sr})
                dcurves.append(curve(s_r["w"])["m"])
            cache[k] = (draws, dcurves)
        draws, dcurves = cache[k]
        les = {}
        for kk in R5_STATS + ("fusion", "segregation"):
            v = {**{s_: st[s_] for s_ in R5_STATS}, "fusion": fl, "segregation": sl}[kk]
            nv = np.array([d_[kk] for d_ in draws], float)
            ok = np.isfinite(nv)
            mu, sd = float(np.nanmean(nv)), float(np.nanstd(nv))
            les[kk] = {"value": float(v), "null_mean": mu, "null_sd": sd,
                       "z": float((v - mu) / sd) if sd > 0 and np.isfinite(v) else float("nan"),
                       "percentile": float((nv[ok] < v).mean()) if np.isfinite(v) else float("nan"),
                       "null_nan": int((~ok).sum())}
        out["lesions"][label] = {"k": int(k), **les}
        arr = np.array(dcurves)
        out["curves"]["random"][str(k)] = {"mean": np.nanmean(arr, 0).tolist(), "sd": np.nanstd(arr, 0).tolist(),
                                           "c": out["curves"][label]["c"]}
        print(f"R5 v1 {b.run} no_{label:9s} k={k:2d} " + " ".join(
            f"{kk} {les[kk]['value']:+.3f} (z {les[kk]['z']:+.2f})" for kk in ("posreg_vis", "posreg_prop", "fusion", "segregation")))
    return out


def r5_v2(b):
    """Every unit clamped alone: change in fusion and in segregation, against
    the unit's congruency index; permutation test on class means."""
    X, ref = b.d["X"], b.msl.mean(0)
    f0, s0 = _fs(b, b.pred)
    dF, dS = np.zeros(64), np.zeros(64)
    for u in range(64):
        keep = np.ones(64, bool)
        keep[u] = False
        pred_u, _ = b.net.lesion_predict(X, keep, ref)
        f, s = _fs(b, pred_u)
        dF[u], dS[u] = f - f0, s - s0
    cs = class_sets(b.classes)
    rng = np.random.default_rng(0)
    out = {"intact": {"fusion": f0, "segregation": s0}, "dF": dF.tolist(), "dS": dS.tolist(),
           "index": b.cong["index"].tolist(), "classes": b.classes.tolist(), "by_class": {}}
    for c, ix in cs.items():
        if len(ix) == 0:
            continue
        rec = {"n": int(len(ix)), "dF_mean": float(dF[ix].mean()), "dS_mean": float(dS[ix].mean()),
               "dF_ci": bootstrap_ci(dF[ix]), "dS_ci": bootstrap_ci(dS[ix])}
        # permutation: is this class's mean effect unusual for a random set of the same size?
        for name, arr in (("dF", dF), ("dS", dS)):
            obs = arr[ix].mean()
            perm = np.array([arr[rng.choice(64, len(ix), replace=False)].mean() for _ in range(5000)])
            rec[f"{name}_perm_p_two_sided"] = float(np.mean(np.abs(perm - arr.mean()) >= abs(obs - arr.mean())))
        out["by_class"][c] = rec
    out["corr_index_dS"] = float(np.corrcoef(b.cong["index"], dS)[0, 1])
    out["corr_index_dF"] = float(np.corrcoef(b.cong["index"], dF)[0, 1])
    print(f"R5 v2 {b.run}: corr(index, dSeg) {out['corr_index_dS']:+.2f}, corr(index, dFus) {out['corr_index_dF']:+.2f}")
    for c, rec in out["by_class"].items():
        print(f"      {c:9s} n={rec['n']:2d} dFus {rec['dF_mean']:+.4f} (p {rec['dF_perm_p_two_sided']:.3f}) "
              f"dSeg {rec['dS_mean']:+.4f} (p {rec['dS_perm_p_two_sided']:.3f})")
    return out


def r5_v3(b, n_random=30, seed=0):
    """Cumulative lesions: units of a class removed one at a time in order of
    |congruency index| (most typical first), fusion and segregation after each
    step, against random orders over all 64 units."""
    X, ref = b.d["X"], b.msl.mean(0)
    idx_all = np.abs(b.cong["index"])
    cs = class_sets(b.classes)
    kmax = max(len(ix) for ix in cs.values())
    out = {"k": list(range(1, kmax + 1)), "classes": {}, "random": {}}
    for c, ix in cs.items():
        if len(ix) == 0:
            continue
        order = ix[np.argsort(-idx_all[ix])]
        F, S = [], []
        keep = np.ones(64, bool)
        for u in order:
            keep[u] = False
            pred_l, _ = b.net.lesion_predict(X, keep, ref)
            f, s = _fs(b, pred_l)
            F.append(f)
            S.append(s)
        out["classes"][c] = {"order": order.tolist(), "fusion": F, "segregation": S}
    rng = np.random.default_rng(seed)
    RF, RS = [], []
    for _ in range(n_random):
        order = rng.permutation(64)[:kmax]
        keep = np.ones(64, bool)
        F, S = [], []
        for u in order:
            keep[u] = False
            pred_l, _ = b.net.lesion_predict(X, keep, ref)
            f, s = _fs(b, pred_l)
            F.append(f)
            S.append(s)
        RF.append(F)
        RS.append(S)
    RF, RS = np.array(RF), np.array(RS)
    out["random"] = {"n": n_random, "fusion_mean": RF.mean(0).tolist(), "fusion_sd": RF.std(0).tolist(),
                     "segregation_mean": RS.mean(0).tolist(), "segregation_sd": RS.std(0).tolist()}
    for c, rec in out["classes"].items():
        k = len(rec["fusion"])
        print(f"R5 v3 {b.run} {c:9s}: after all {k:2d}: fusion {rec['fusion'][-1]:.3f} "
              f"(random {RF[:, k - 1].mean():.3f} ± {RF[:, k - 1].std():.3f}), segregation {rec['segregation'][-1]:.3f} "
              f"(random {RS[:, k - 1].mean():.3f} ± {RS[:, k - 1].std():.3f})")
    return out


def r5_v4(b, n_random=20, seed=0, alpha=1e-3):
    """Whole-class clamp with the read-out refitted (ridge) on the training
    split from the remaining units: does the behaviour recover? If it does,
    the class's information was redundant; if not, it was not represented
    elsewhere."""
    from sklearn.linear_model import Ridge
    tr = np.asarray(b.ckpt["splits"]["train"])
    X_tr, X_te = b.d_full["X"][tr], b.d["X"]
    Y_tr = np.stack([b.d_full[k][tr] for k in b.names], axis=1)
    h_tr_full = b.net.hidden(X_tr)[-1]
    h_te_full = b.msl
    ref = h_te_full.mean(0)
    var_cols = b.net.var_cols

    def refit_predict(keep):
        h_tr = h_tr_full.copy()
        h_te = h_te_full.copy()
        h_tr[:, ~keep] = ref[~keep]
        h_te[:, ~keep] = ref[~keep]
        model = Ridge(alpha=alpha).fit(h_tr, Y_tr)
        pred = model.predict(h_te)
        for c in var_cols:
            pred[:, c] = np.clip(pred[:, c], 1e-3, None)
        return pred

    def stats(pred):
        w, wp = hybrid_reads(b, pred)
        from common import position_regression
        f, s = fusion_segregation(b.d["disparity"], w)
        d, n = b.d, b.names
        pv = position_regression(pred[:, n.index("mu_vis")], d["seg_vis_mu"], d["fused_mu"], d["post_c1"])["slope"]
        pp = position_regression(pred[:, n.index("mu_prop")], d["seg_prop_mu"], d["fused_mu"], d["post_c1"])["slope"]
        target = np.stack([d[k] for k in n], axis=1)
        return {"fusion": f, "segregation": s, "posreg_vis": float(pv), "posreg_prop": float(pp),
                "rmse": float(np.sqrt(((pred - target) ** 2).mean()))}

    keep_all = np.ones(64, bool)
    out = {"alpha": alpha, "intact_refit": stats(refit_predict(keep_all)),
           "intact_frozen": stats(b.pred), "lesions": {}}
    rng = np.random.default_rng(seed)
    cs = class_sets(b.classes)
    cache = {}
    for c, ix in cs.items():
        if len(ix) == 0:
            continue
        k = len(ix)
        keep = np.ones(64, bool)
        keep[ix] = False
        st = stats(refit_predict(keep))
        if k not in cache:
            draws = []
            for _ in range(n_random):
                rkeep = np.ones(64, bool)
                rkeep[rng.choice(64, k, replace=False)] = False
                draws.append(stats(refit_predict(rkeep)))
            cache[k] = draws
        null = cache[k]
        rec = {"k": int(k)}
        for kk in st:
            nv = np.array([d_[kk] for d_ in null])
            rec[kk] = {"value": st[kk], "null_mean": float(nv.mean()), "null_sd": float(nv.std()),
                       "z": float((st[kk] - nv.mean()) / nv.std()) if nv.std() > 0 else float("nan")}
        out["lesions"][c] = rec
        print(f"R5 v4 {b.run} no_{c:9s} k={k:2d} refit: " + " ".join(
            f"{kk} {rec[kk]['value']:+.3f} (random {rec[kk]['null_mean']:+.3f}, z {rec[kk]['z']:+.2f})"
            for kk in ("fusion", "segregation", "posreg_vis", "posreg_prop")))
    print(f"R5 v4 {b.run} intact refit: " + " ".join(f"{kk} {v:+.3f}" for kk, v in out["intact_refit"].items()))
    return out


def r5(runs=("flagship", "pcommon03")):
    import json
    results = {}
    for run in runs:
        path = OUT / "R5_lesion_behaviour" / f"numbers_{run}.json"
        if "--redraw" in sys.argv and path.exists():
            results[run] = json.loads(path.read_text())
            continue
        b = bundle(run)
        results[run] = {"v1": r5_v1(b), "v2": r5_v2(b), "v3": r5_v3(b), "v4": r5_v4(b), "counts": b.counts}
        save_json(results[run], path)

    # ---- figure: one row per network, four panels (v1 curves, v2, v3, v4) ----
    def p_v1(run):
        def draw(ax=None):
            ax = panel_axes(ax)
            r = results[run]["v1"]
            b = bundle(run)
            cv = r["curves"]
            grid = np.array(cv["grid"])
            c_o, m_o, _ = mean_by_bin(np.abs(b.d["disparity"]), b.d["post_c1"], grid)
            ax.plot(c_o, m_o, "--", lw=PANEL_LW, color=COLORS["analytical"], label="analytical")
            k_opp = str(r["lesions"]["opposite"]["k"])
            band = cv["random"][k_opp]
            bc, bm, bs = np.array(band["c"]), np.array(band["mean"]), np.array(band["sd"])
            ax.fill_between(bc, bm - 2 * bs, bm + 2 * bs, color="gray", alpha=0.25, lw=0,
                            label=f"random {k_opp}, ±2 sd")
            ax.plot(cv["intact"]["c"], cv["intact"]["m"], "-", lw=PANEL_LW + 0.4, color="k", label="intact")
            for c in CLASSES:
                if c in cv:
                    ax.plot(cv[c]["c"], cv[c]["m"], "-", lw=PANEL_LW, color=CLASS_COLORS[c],
                            label=f"−{c} ({r['lesions'][c]['k']})")
            ax.set_ylim(-0.1, 1.8)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            fig = finish_panel(ax, "|disparity| (deg)", "weight on fused estimate",
                               f"{RUN_LABEL[run]}: whole-class clamp", None)
            ax.legend(fontsize=7, loc="upper center", ncol=2, handlelength=1.3,
                      columnspacing=0.7, borderaxespad=0.2, handletextpad=0.5)
            return fig
        return draw

    def p_v2(run):
        def draw(ax=None):
            ax = panel_axes(ax)
            r = results[run]["v2"]
            idx, dS, dF, cl = (np.array(r[k]) for k in ("index", "dS", "dF", "classes"))
            for c in CLASSES:
                m = cl == c
                if m.any():
                    ax.plot(idx[m], dS[m], "o", ms=PANEL_MS + 0.5, color=CLASS_COLORS[c], alpha=0.8,
                            label=f"{c} ({m.sum()})")
            ax.axhline(0, lw=0.5, color="gray")
            ax.axvline(0.5, ls=":", lw=0.6, color="gray")
            ax.axvline(-0.5, ls=":", lw=0.6, color="gray")
            # every one of the 64 units is drawn; the legend sits in headroom
            # above the data so that its markers cannot be read as points
            top = float(np.abs(dS).max())
            ax.set_ylim(-1.15 * top, 1.7 * top)
            ax.set_yticks([t for t in ax.get_yticks() if abs(t) <= 1.15 * top])
            ax.set_xlim(-1.08, 1.08)
            fig = finish_panel(ax, "congruency index", "Δ segregation, one unit clamped",
                               f"single-unit lesions (r = {r['corr_index_dS']:+.2f})", None)
            ax.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8,
                      borderaxespad=0.2, handletextpad=0.4, frameon=False, title="all 64 units, one at a time",
                      title_fontsize=7)
            return fig
        return draw

    def p_v3(run, stat="segregation"):
        ylab = {"segregation": "segregation at |d| ≥ 15°", "fusion": "fusion at |d| ≤ 2°"}[stat]

        def draw(ax=None):
            ax = panel_axes(ax)
            r = results[run]["v3"]
            k = np.array(r["k"])
            rm, rs = np.array(r["random"][f"{stat}_mean"]), np.array(r["random"][f"{stat}_sd"])
            ax.fill_between(k, rm - 2 * rs, rm + 2 * rs, color="gray", alpha=0.25, lw=0, label="random order ±2 sd")
            ax.plot(k, rm, "-", lw=0.8, color="gray")
            for c in CLASSES:
                if c in r["classes"]:
                    S = np.array(r["classes"][c][stat])
                    ax.plot(k[:len(S)], S, "-", lw=PANEL_LW, color=CLASS_COLORS[c], label=f"{c} first")
            ax.set_ylim(-0.05, 1.05)
            return finish_panel(ax, "units removed (most typical first)", ylab,
                                f"{RUN_LABEL[run]}: cumulative lesion", "lower left")
        return draw

    def p_v4(run):
        def draw(ax=None):
            ax = panel_axes(ax)
            r = results[run]["v4"]
            stats_ = ("fusion", "segregation", "posreg_vis", "posreg_prop")
            labels = ("fusion", "segre-\ngation", "slope\nvis", "slope\nhand")
            x = np.arange(len(stats_))
            wdt = 0.2
            ax.bar(x - 1.5 * wdt, [r["intact_refit"][s] for s in stats_], wdt, color="k", alpha=0.6, label="intact")
            for i, c in enumerate(CLASSES):
                if c not in r["lesions"]:
                    continue
                vals = [r["lesions"][c][s]["value"] for s in stats_]
                nul = [r["lesions"][c][s]["null_mean"] for s in stats_]
                ax.bar(x + (i - 0.5) * wdt, vals, wdt, color=CLASS_COLORS[c], alpha=0.85,
                       label=f"−{c}, refit")
                ax.plot(x + (i - 0.5) * wdt, nul, "_", ms=8, mew=1.4, color="k")
            ax.axhline(1, ls=":", lw=0.6, color="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.55)
            fig = finish_panel(ax, "", "value, read-out refitted",
                               "refit; ticks = random lesions", None)
            ax.legend(fontsize=7, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.7,
                      borderaxespad=0.2)
            return fig
        return draw

    draws, figs = [], {}
    for run in runs:
        draws += [p_v1(run), p_v2(run), p_v3(run), p_v4(run)]
    figs["R5_lesion_behaviour/row_ABCD_EFGH"] = manuscript_grid(draws, ncols=4)
    for run in runs:
        for key, fn in (("v1_whole_class", p_v1), ("v2_single_unit", p_v2), ("v3_cumulative", p_v3),
                        ("v4_refit", p_v4)):
            figs[f"R5_lesion_behaviour/{run}_{key}"] = fn(run)()
        figs[f"R5_lesion_behaviour/{run}_v3_cumulative_fusion"] = p_v3(run, "fusion")()
    # recommended manuscript figure: cumulative lesion (fusion, segregation) + refitted read-out, main network
    figs["R5_lesion_behaviour/recommended_ABC"] = manuscript_grid(
        [p_v3("flagship", "fusion"), p_v3("flagship", "segregation"), p_v4("flagship")], ncols=3)
    save_figures(figs, OUT, formats=FORMATS)


ANALYSES = {"units": units, "r6": r6, "r5": r5}

if __name__ == "__main__":
    apply_style()
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["all"]
    if which == ["all"]:
        which = list(ANALYSES)
    for w in which:
        print(f"=== {w} ===")
        ANALYSES[w]()
