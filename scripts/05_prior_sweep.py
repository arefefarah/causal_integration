"""Stage 5 -- the cross-prior experiment (design SS9.3, satellites).

    python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    python scripts/05_prior_sweep.py --control pcommon1        # sigma_out source

Trains one network per value of p_common, changing ONLY the Bernoulli constant
-- same seed, same encoders, same sigma ranges, same architecture -- and asks
whether the causal weighting each network implements shifts with its prior in
the way Bayes says it must.

This is a parametric test of the Bayesian interpretation, not a demonstration
that the prior is discovered from nothing: the prior is of course present in
each network's training targets. What is non-trivial is that a network trained
only on four model-averaged numbers, with no explicit causal read-out, ends up
implementing a fusion-vs-segregation trade-off whose *quantitative* dependence
on the prior matches the analytical posterior. A disparity heuristic or any
fixed-weight scheme would not produce that dependence.

Writes:
    results/prior_sweep/sweep.json    per-prior statistics
    results/prior_sweep/curves.npz    per-prior weight-vs-disparity curves
    results/prior_sweep/figures/      the manuscript figure
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi import analysis
from cmsi.data import make_dataset, subset
from cmsi.models import hidden_activations, predict, train
from cmsi.utils import load_config, load_json, run_dir, save_json, tweak
from cmsi.utils.paths import RESULTS

# Bin centres for the weight-vs-disparity curve. Deliberately COARSE near
# zero: there the two hypotheses nearly coincide, Delta collapses, and a narrow
# bin cannot identify a weight -- a fine grid there produces a spike that is an
# artefact of the estimator rather than anything the network does.
GRID = np.array([-30, -24, -19, -15, -12, -9, -6, -3.5, -1.5,
                 1.5, 3.5, 6, 9, 12, 15, 19, 24, 30], float)


def _on_grid(grid, centres, values):
    """Map per-bin values onto the full grid, NaN where the bin was dropped.

    Plotting on the full grid is what makes a dropped bin read as a GAP rather
    than as a straight line joining the two bins either side of it -- at high
    priors most far-disparity bins are too sparse to keep, and connecting
    across them invents a transition that was never measured.
    """
    import numpy as _np
    out = _np.full(len(grid), _np.nan)
    for c, v in zip(centres, values, strict=True):
        j = int(_np.argmin(_np.abs(_np.asarray(grid) - c)))
        out[j] = v
    return out


def one_prior(cfg, p, sig_out, n_trials, epochs, seed):
    """Train one network at prior p and return its cross-prior statistics."""
    c = tweak(cfg, p_common=float(p), n_trials=int(n_trials), epochs=int(epochs),
              seed=int(seed))
    d_full = make_dataset(c)
    model, history, splits = train(d_full, c, verbose=False)
    d = subset(d_full, splits["test"])
    names = d_full["target_names"]
    iv, ip = names.index("mu_vis"), names.index("mu_prop")
    pred = predict(model, d["X"])
    target = np.stack([d[k] for k in names], axis=1)

    dv = d["fused_mu"] - d["seg_vis_mu"]
    so = np.asarray(sig_out) if sig_out is not None \
        else analysis.sigma_out(pred, target)

    sw_v = analysis.sigma_w(so[iv], dv)
    w_vis = analysis.fusion_weight(pred[:, iv], d["seg_vis_mu"], d["fused_mu"], 1.0)
    w_f = np.where(sw_v < 0.1, w_vis, np.nan)

    net_mid, net_k = analysis.transition_fit(np.abs(d["disparity"]), w_f)
    opt_mid, opt_k = analysis.transition_fit(np.abs(d["disparity"]), d["post_c1"])

    reg_v = analysis.position_regression(pred[:, iv], d["seg_vis_mu"],
                                         d["fused_mu"], d["post_c1"])
    reg_p = analysis.position_regression(pred[:, ip], d["seg_prop_mu"],
                                         d["fused_mu"], d["post_c1"])
    vs = analysis.variance_signature(
        pred[:, names.index("var_vis")], d["post_c1"], d["fused_mu"],
        d["fused_var"], d["seg_vis_mu"], d["seg_vis_var"])
    rel = analysis.reliability_within_disparity(w_f, d["post_c1"],
                                                np.abs(d["disparity"]))
    mc = analysis.model_comparison(pred[:, iv], d["post_c1"], d["fused_mu"],
                                   d["seg_vis_mu"])
    acts = hidden_activations(model, d["X"])
    dec = analysis.summarise(analysis.decode_by_layer(acts, d["post_c1"]))

    # curves for the figure: per-bin LEAST-SQUARES implied weight (unbiased;
    # see analysis.binned_implied_weight for why the filtered ratio is not).
    c_net, m_net, n_net, se_net = analysis.binned_implied_weight(
        pred[:, iv], d["seg_vis_mu"], d["fused_mu"], d["disparity"], GRID,
        min_count=80, sigma_out=so[iv], max_se=0.05)
    c_opt, m_opt, _ = analysis.mean_by_bin(d["disparity"], d["post_c1"], GRID)
    keep = np.ones(len(c_net), bool)
    stats = {
        "p_common": float(p),
        "n_test": int(len(pred)),
        "best_val": float(history["best_val"]),
        "realized_c1": float((d_full["C"] == 1).mean()),
        "midpoint_net": float(net_mid), "midpoint_opt": float(opt_mid),
        "sharpness_net": float(net_k), "sharpness_opt": float(opt_k),
        "posreg_slope_vis": reg_v["slope"], "posreg_ci_vis": reg_v["slope_ci95"],
        "posreg_slope_prop": reg_p["slope"], "posreg_ci_prop": reg_p["slope_ci95"],
        "hump_net": float(vs.get("hump_net", np.nan)),
        "hump_opt": float(vs.get("hump_analytical", np.nan)),
        "reliability_slope": float(rel["combined_slope"]),
        "reliability_se": float(rel["combined_se"]),
        "best_strategy": mc["best"],
        "rmse_averaging": mc["overall"]["averaging"],
        "rmse_selection": mc["overall"]["selection"],
        "rmse_fixed": mc["overall"]["fixed"],
        "decode_sil": dec.get("layer0"), "decode_msl": dec.get("layer1"),
        "accuracy_r2": [r["r2"] for r in analysis.accuracy(pred, target, names)],
    }
    curves = {"centres_net": GRID,
              "w_net": _on_grid(GRID, c_net[keep], m_net[keep]),
              "se_net": _on_grid(GRID, c_net[keep], se_net[keep]),
              "centres_opt": c_opt, "w_opt": m_opt,
              # per-trial arrays, so the figure can be redrawn without retraining
              "t_disparity": d["disparity"], "t_post": d["post_c1"],
              "t_fused": d["fused_mu"], "t_seg_vis": d["seg_vis_mu"],
              "t_pred_vis": pred[:, iv]}
    return stats, curves


SUMMARY_KEYS = ("midpoint_net", "midpoint_opt", "posreg_slope_vis",
                "posreg_slope_prop", "hump_net", "hump_opt",
                "reliability_slope", "decode_msl", "decode_sil",
                "rmse_averaging", "sharpness_net", "sharpness_opt")


def aggregate(rows):
    """Collapse per-(prior, seed) rows to one row per prior, mean +- SEM.

    With several seeds the across-seed SEM is the uncertainty that matters for
    the figure: the within-run confidence interval on a regression slope says
    how well that ONE network's behaviour is pinned down, not how much the
    result would move if the network were retrained. Only the second answers
    "is the deviation from Bayes-optimal a property of the model or of the
    run?", which is the question the error bars on the figure are there for.
    """
    by_prior = {}
    for r in rows:
        by_prior.setdefault(r["p_common"], []).append(r)

    out = []
    for p in sorted(by_prior):
        group = by_prior[p]
        agg = {"p_common": p, "n_seeds": len(group),
               "seeds": sorted(r.get("seed", 0) for r in group)}
        for k in SUMMARY_KEYS:
            vals = np.array([r[k] for r in group if r.get(k) is not None], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                agg[k], agg[k + "_sem"] = np.nan, np.nan
                continue
            agg[k] = float(vals.mean())
            agg[k + "_sem"] = (float(vals.std(ddof=1) / np.sqrt(vals.size))
                               if vals.size > 1 else 0.0)
        # keep the single-run CI when there is only one seed, so the figure can
        # still show an honest error bar in that case
        if len(group) == 1:
            agg["posreg_ci_vis"] = group[0]["posreg_ci_vis"]
            agg["posreg_ci_prop"] = group[0]["posreg_ci_prop"]
        else:
            s = agg["posreg_slope_vis"], agg["posreg_slope_vis_sem"]
            agg["posreg_ci_vis"] = [s[0] - 1.96 * s[1], s[0] + 1.96 * s[1]]
            s = agg["posreg_slope_prop"], agg["posreg_slope_prop_sem"]
            agg["posreg_ci_prop"] = [s[0] - 1.96 * s[1], s[0] + 1.96 * s[1]]
        strategies = {r["best_strategy"] for r in group}
        agg["best_strategy"] = (strategies.pop() if len(strategies) == 1
                                else "mixed:" + "/".join(sorted(strategies)))
        agg["accuracy_r2"] = np.mean([r["accuracy_r2"] for r in group],
                                     axis=0).tolist()
        out.append(agg)
    return out


def _write(out, args, rows, curves, agg):
    save_json({"config": args.config, "seeds": args.seeds, "n_trials": args.n,
               "epochs": args.epochs, "sigma_out_source": args.control,
               "rows": rows, "aggregated": agg}, out / "sweep.json")
    np.savez_compressed(out / "curves.npz", **curves)


def main(args):
    cfg = load_config(args.config)
    out = RESULTS / "prior_sweep"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    sig_out = None
    if args.control:
        cm = load_json(run_dir(args.control, create=False) / "metrics.json")
        if "residual_std" in cm:
            sig_out = np.asarray(cm["residual_std"])
            print(f"sigma_out from '{args.control}': "
                  + ", ".join(f"{s:.3f}" for s in sig_out))
        else:
            print(f"warning: control '{args.control}' has no residual_std; "
                  f"each run will use its own residuals.")

    n_nets = len(args.priors) * len(args.seeds)
    print(f"{len(args.priors)} priors x {len(args.seeds)} seed(s) = "
          f"{n_nets} networks\n")

    rows, curves = [], {}
    for seed in args.seeds:
        print(f"--- seed {seed} ---", flush=True)
        for p in args.priors:
            s, c = one_prior(cfg, p, sig_out, args.n, args.epochs, seed)
            s["seed"] = int(seed)
            rows.append(s)
            # per-trial arrays only for the first seed -- they are the bulk of
            # the file and the curves panel needs one representative run
            keep_arrays = seed == args.seeds[0]
            for k, v in c.items():
                if k.startswith("t_") and not keep_arrays:
                    continue
                curves[f"s{seed}_p{p:.2f}_{k}"] = v
                if keep_arrays:
                    curves[f"p{p:.2f}_{k}"] = v          # back-compat alias
            print(f"  p_common {p:.2f}:  midpoint net {s['midpoint_net']:6.2f} "
                  f"vs opt {s['midpoint_opt']:6.2f} | posreg slope "
                  f"{s['posreg_slope_vis']:.3f} | hump {s['hump_net']:5.2f}",
                  flush=True)
        # write after every seed, so a long multi-seed run is never lost
        _write(out, args, rows, curves, aggregate(rows))
        print(f"  (saved {len(rows)} of {n_nets} networks)\n", flush=True)

    agg = aggregate(rows)
    _write(out, args, rows, curves, agg)
    print(f"wrote {out/'sweep.json'} and {out/'curves.npz'}")

    from cmsi.viz import apply_style
    from cmsi.viz.prior_sweep import figure, hump_panel
    apply_style()
    fig = figure(agg, curves)
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"prior_sweep.{ext}", dpi=300,
                    bbox_inches="tight")
    ax = hump_panel(agg)
    for ext in ("png", "pdf"):
        ax.figure.savefig(out / "figures" / f"variance_hump_vs_prior.{ext}",
                          dpi=300, bbox_inches="tight")
    print(f"wrote {out/'figures'}/prior_sweep.png, variance_hump_vs_prior.png "
          f"(+ .pdf)")
    if len(args.seeds) > 1:
        print(f"error bars are across-seed SEM over {len(args.seeds)} seeds")
    else:
        print("single seed: error bars are the within-run 95% CI. Run with "
              "--seeds 0 1 2 to get across-seed error bars.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="base config (default flagship)")
    p.add_argument("--priors", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one network per (prior, seed); >1 seed gives "
                        "across-seed error bars on the figure")
    p.add_argument("--seed", type=int, default=None,
                   help="deprecated alias for --seeds with a single value")
    p.add_argument("--control", default="pcommon1",
                   help="run whose residual_std supplies sigma_out")
    a = p.parse_args()
    if a.seed is not None:
        a.seeds = [a.seed]
    if a.config is None:
        from cmsi.utils.paths import CONFIGS
        a.config = str(CONFIGS / "flagship.yaml")
    main(a)
