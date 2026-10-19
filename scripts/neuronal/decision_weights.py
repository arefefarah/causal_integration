"""Which units does a read-out of the network's own causal decision lean on?

The network has no decision unit, so we give it one after the fact, in the
manner of the causal-inference units of Rideaux et al. (2021, Fig. 4B): every
test trial is labelled by the network's own behaviour, as in the
decision-bias analysis (Fig. 5) — 'fused' when its implied visual weight is
above 0.5 (the observer's posterior stands in where the weight cannot be
computed), 'segregated' otherwise — and a logistic read-out is fitted from the
multisensory layer (MSL) to that label. Each unit's activity is z-scored over
the test trials first, so the read-out weights are comparable across units.
The weight of a unit towards 'fused' is its coefficient; towards
'segregated' it is the same number with the opposite sign (two classes).

Two fits: on every test trial, and on the trials of the transition band only
(|disparity| between BAND[0] and BAND[1] degrees), where both labels occur at
the same disparities, so that the read-out cannot rely on disparity alone.

    python scripts/neuronal/decision_weights.py                 # main network
    python scripts/neuronal/decision_weights.py --runs flagship flagship_s1 ...   # several

Output: results/<run>/figures/model/neuronal_level_analysis/decision_weights/
(figure, panels, numbers.json; with several runs, the pooled figure too).
Torch-free (npmodel.py).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_COLORS, CLASS_LABEL, CLASSES, Bundle, apply_style,  # noqa: E402
                    finish_panel, hybrid_reads, manuscript_grid, out_dir,
                    panel_axes, save_figures, save_json)
from cmsi.viz.style import FORMATS  # noqa: E402

K = "decision_weights"
BAND = (2.0, 10.0)        # the transition band, degrees of |disparity|
C_REG = 1.0               # inverse L2 strength of the logistic read-out


def trial_labels(b):
    """1 = fused (implied visual weight > 0.5), 0 = segregated, as in Fig. 5."""
    w, _ = hybrid_reads(b, b.pred)
    return (np.where(np.isfinite(w), w, b.d["post_c1"]) > 0.5).astype(int), w


def fit(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=5000)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    clf.fit(X, y)
    return clf.coef_[0], float(acc.mean())


def analyse(run, control="pcommon1"):
    b = Bundle(run, control=control)
    y, w = trial_labels(b)
    msl = np.asarray(b.msl, float)
    sd = msl.std(0)
    ok = sd > 1e-6
    Z = np.zeros_like(msl)
    Z[:, ok] = (msl[:, ok] - msl[:, ok].mean(0)) / sd[ok]
    ad = np.abs(b.d["disparity"])
    band = (ad >= BAND[0]) & (ad <= BAND[1])
    coef_all, acc_all = fit(Z[:, ok], y)
    coef_band, acc_band = fit(Z[band][:, ok], y[band])
    # decoder weights of correlated units are not interpretable one by one
    # (Haufe et al., 2014): the activation pattern A = Cov(Z) w says how each
    # unit's activity co-varies with the read-out; the univariate difference
    # is the unit's mean z-activity on fused minus segregated trials
    pat_all = np.cov(Z[:, ok].T) @ coef_all
    pat_band = np.cov(Z[band][:, ok].T) @ coef_band
    diff_all = Z[y == 1][:, ok].mean(0) - Z[y == 0][:, ok].mean(0)
    diff_band = Z[band & (y == 1)][:, ok].mean(0) - Z[band & (y == 0)][:, ok].mean(0)
    full = lambda c: _expand(c, ok)  # noqa: E731
    rec = {"run": run, "n_trials": int(len(y)), "share_fused": float(y.mean()),
           "band": list(BAND), "n_band": int(band.sum()), "share_fused_band": float(y[band].mean()),
           "acc_all": acc_all, "acc_band": acc_band,
           "classes": [str(c) for c in b.classes], "index": [float(v) for v in b.cong["index"]],
           "w_fused_all": [None if not np.isfinite(v) else float(v) for v in full(coef_all)],
           "w_fused_band": [None if not np.isfinite(v) else float(v) for v in full(coef_band)],
           "pattern_fused_all": [None if not np.isfinite(v) else float(v) for v in full(pat_all)],
           "pattern_fused_band": [None if not np.isfinite(v) else float(v) for v in full(pat_band)],
           "diff_fused_all": [None if not np.isfinite(v) else float(v) for v in full(diff_all)],
           "diff_fused_band": [None if not np.isfinite(v) else float(v) for v in full(diff_band)]}
    rec["summary"] = summarise(rec)
    return rec


def _expand(c, ok):
    out = np.full(len(ok), np.nan)
    out[ok] = c
    return out


def summarise(rec):
    cl = np.array(rec["classes"])
    s = {}
    for key in ("w_fused_all", "w_fused_band", "pattern_fused_all", "pattern_fused_band",
                "diff_fused_all", "diff_fused_band"):
        v = np.array([np.nan if x is None else x for x in rec[key]], float)
        s[key] = {c: {"mean": float(np.nanmean(v[cl == c])), "n": int((cl == c).sum())} for c in CLASSES}
        s[key]["corr_with_index"] = float(np.corrcoef(np.array(rec["index"])[np.isfinite(v)], v[np.isfinite(v)])[0, 1])
    return s


def figure(recs, out, label):
    """A, B: the decoder weights towards 'fused' and towards 'segregated' (all
    trials), per class, one point per unit; C: the weight towards 'segregated'
    against the congruency index; D: the activation pattern towards
    'segregated'; E: mean z-activity on segregated minus fused trials; F: the
    activation pattern of the transition-band read-out."""
    from scipy import stats
    cl = np.concatenate([np.array(r["classes"]) for r in recs])
    idx = np.concatenate([np.array(r["index"], float) for r in recs])
    get = lambda key: np.concatenate([np.array([np.nan if x is None else x for x in r[key]], float)  # noqa: E731
                                      for r in recs])
    wa, wb = get("w_fused_all"), get("w_fused_band")
    n_net = len(recs)
    splits = np.cumsum([len(r["classes"]) for r in recs])[:-1]

    def strip(ax, vals, title, ylabel):
        allv = []
        for i, c in enumerate(CLASSES):
            v = vals[(cl == c) & np.isfinite(vals)]
            allv.extend(v.tolist())
            if len(v) > 1:
                ax.boxplot([v], positions=[i], widths=0.6, showfliers=False,
                           medianprops=dict(color="k", lw=1.0), boxprops=dict(color=CLASS_COLORS[c], lw=1.0),
                           whiskerprops=dict(color=CLASS_COLORS[c], lw=0.8), capprops=dict(color=CLASS_COLORS[c], lw=0.8))
            jit = np.random.default_rng(5 + i).uniform(-0.2, 0.2, len(v))
            ax.plot(i + jit, v, "o", ms=2.2 if n_net > 1 else 3.0, color=CLASS_COLORS[c],
                    alpha=0.35 if n_net > 1 else 0.75, mew=0, rasterized=n_net > 1)
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES], rotation=15)
        ax.set_xlim(-0.7, len(CLASSES) - 0.3)
        lim = 1.1 * np.nanmax(np.abs(allv)) if allv else 1
        ax.set_ylim(-lim, lim)
        if n_net > 1:
            # units are nested in networks: compare the class means of each
            # network, paired over networks
            m = {c: np.array([np.nanmean(v[k]) for v, k in zip(np.split(vals, splits), np.split(cl == c, splits))])
                 for c in ("congruent", "opposite")}
            t = stats.ttest_rel(m["opposite"], m["congruent"])
            ax.text(-0.6, -0.93 * lim, f"opp − con, paired over networks: t = {t.statistic:+.1f}, p = {t.pvalue:.1g}",
                    ha="left", va="bottom", fontsize=6)
        else:
            con, opp = vals[(cl == "congruent") & np.isfinite(vals)], vals[(cl == "opposite") & np.isfinite(vals)]
            if len(con) > 2 and len(opp) > 2:
                p = stats.mannwhitneyu(opp, con).pvalue
                ax.text(0.5, -0.93 * lim, f"opp vs con (units): p = {p:.1g}", ha="center", va="bottom", fontsize=6.5)
        return finish_panel(ax, "", ylabel, title, None)

    acc_a = np.mean([r["acc_all"] for r in recs])
    acc_b = np.mean([r["acc_band"] for r in recs])
    pa, pb, da = get("pattern_fused_all"), get("pattern_fused_band"), get("diff_fused_all")
    YL = "towards 'segregated'"

    def pA(ax=None):
        return strip(panel_axes(ax), wa, f"decoder weight, 'fused' (acc {acc_a:.2f})", "weight (z-scored units)")

    def pB(ax=None):
        return strip(panel_axes(ax), -wa, "decoder weight, 'segregated'", "weight (z-scored units)")

    def pC(ax=None):
        ax = panel_axes(ax)
        for c in CLASSES:
            m = (cl == c) & np.isfinite(wa)
            ax.plot(idx[m], -wa[m], "o", ms=2.2 if n_net > 1 else 3.0, color=CLASS_COLORS[c],
                    alpha=0.35 if n_net > 1 else 0.8, mew=0, label=CLASS_LABEL[c], rasterized=n_net > 1)
        ok = np.isfinite(wa)
        r = np.corrcoef(idx[ok], -wa[ok])[0, 1]
        for x in (-0.5, 0.5):
            ax.axvline(x, ls=":", lw=0.6, color="gray")
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xlim(-1.05, 1.05)
        fig = finish_panel(ax, "congruency index", "weight " + YL, f"weight vs index (r = {r:+.2f})", None)
        ax.legend(fontsize=6.5, loc="upper left", handlelength=0.6, handletextpad=0.2, borderaxespad=0.2,
                  markerscale=1.5)
        return fig

    def pD(ax=None):
        return strip(panel_axes(ax), -pa, "activation pattern, all trials", "pattern " + YL)

    def pE(ax=None):
        return strip(panel_axes(ax), -da, "mean activity, segr. − fused", "Δ z-activity")

    def pF(ax=None):
        return strip(panel_axes(ax), -pb, f"pattern, {BAND[0]:.0f}–{BAND[1]:.0f}° only (acc {acc_b:.2f})",
                     "pattern " + YL)

    draws = [pA, pB, pC, pD, pE, pF]
    name = K if n_net == 1 else f"{K}_{n_net}_networks"
    figs = {f"{K}/{name}": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_weight_fused", "B_weight_segregated", "C_weight_vs_index", "D_pattern",
                        "E_activity_difference", "F_pattern_band"), draws):
        figs[f"{K}/{key}" + ("" if n_net == 1 else f"_{n_net}")] = fn()
    save_figures(figs, out, formats=FORMATS, flat=n_net == 1)


def main(argv=None):
    import argparse
    import warnings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=["flagship"])
    ap.add_argument("--control", default="pcommon1")
    ap.add_argument("--redraw", action="store_true", help="re-plot from the saved numbers.json")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    apply_style()
    run0 = args.runs[0]
    out = out_dir(run0)
    folder = out / K
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "per_network"
    cache.mkdir(exist_ok=True)
    recs = []
    for run in args.runs:
        path = cache / f"{run}.json"
        if path.exists() and args.redraw:
            recs.append(json.loads(path.read_text()))
            continue
        rec = analyse(run, args.control)
        save_json(rec, path)
        recs.append(rec)
        s = rec["summary"]["w_fused_all"]
        print(f"{run}: fused {rec['share_fused']:.2f}, acc {rec['acc_all']:.3f} (band {rec['acc_band']:.3f}); "
              "mean weight towards fused " + ", ".join(f"{CLASS_LABEL[c]} {s[c]['mean']:+.3f}" for c in CLASSES)
              + f"; r(index) {s['corr_with_index']:+.2f}")
    label = "main network" if args.runs == ["flagship"] else f"{len(recs)} networks"
    save_json({"runs": args.runs, "label": label, "per_network": [r["summary"] for r in recs],
               "acc_all": [r["acc_all"] for r in recs], "acc_band": [r["acc_band"] for r in recs]},
              folder / ("numbers.json" if len(recs) == 1 else f"numbers_{len(recs)}.json"))
    figure(recs, out, label)
    print(f"   -> {folder}/")


if __name__ == "__main__":
    main()
