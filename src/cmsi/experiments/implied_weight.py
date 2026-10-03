"""The implied-weight experiment.

The network never outputs a weight. Every weight in this project is *implied*:
recovered from the network's outputs on each trial by the hybrid read
(analysis.hybrid_weight) -- per channel, the mixture-variance root of the
variance output where that root is unique (Delta^2 <= c, which includes zero
disparity) and the channel's own position ratio where the variance has two
roots (Delta^2 > c). It exists on every trial and nothing is filtered.

This module applies that read to a network's stored test split, exactly as
stage 3 does, but writes under results/experiments/implied_weight/<name>/ so
that nothing here can disturb results/<run>/:

    weight_analysis        the weight per channel with its summaries, the
                           position- and variance-domain regressions, the
                           transition midpoints, the channel consistency
    weight_by_reliability  the weight against disparity at many levels of
                           one input's reliability, per channel
    baseline_figures       the pipeline's weight figures (04, 05, 08v, 15,
                           16) for one run

Entry points (scripts/06_implied_weight.py): figures, reliability, and train,
which takes a new configuration through the whole pipeline (stages 0-4) into
results/<name>/ with a p_common = 1 control of the same configuration.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import (
    binned_weight,
    by_reliability,
    compare,
    hybrid_weight,
    mean_by_bin,
    position_regression,
    transition_fit,
    variance_regression,
    weight_by_posterior,
    weight_consistency,
    weight_regression,
)
from cmsi.viz import results as viz_results
from cmsi.viz.style import COLORS, SIZE, label_panels

EXPERIMENT = "implied_weight"

# The disparity grid of the manuscript's prior-sweep figure (18 bins from -30
# to +30 deg, coarse near zero) and the guards on a bin's mean. The
# pipeline's own figures keep the config's disparity_grid.
GRID = [-30, -24, -19, -15, -12, -9, -6, -3.5, -1.5,
        1.5, 3.5, 6, 9, 12, 15, 19, 24, 30]
MIN_COUNT = 25
MAX_SE = 0.05
# Per reliability level a curve rests on a fifth of the trials or less, so the
# guards are relaxed there; a bin's error bar still says how good it is.
LEVEL_MIN_COUNT = 10
LEVEL_MAX_SE = 0.1

# one channel = (key, position output, variance output, segregated mean,
# segregated variance)
CHANNELS = (("vis", "mu_vis", "var_vis", "seg_vis_mu", "seg_vis_var"),
            ("prop", "mu_prop", "var_prop", "seg_prop_mu", "seg_prop_var"))
INPUTS = {"vis": "sig2_vis", "prop": "sig2_prop", "eye": "sig2_eye"}
INPUT_LABEL = {"vis": "visual noise sigma2_vis",
               "prop": "proprioceptive noise sigma2_prop",
               "eye": "eye-position noise sigma2_eye"}


# --------------------------------------------------------------------------- #
# where things go, and loading an existing run
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/implied_weight/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def load_run(run, config=None):
    """(pred, d_test, names, cfg) for results/<run>/, on its stored test split.

    The analysis block of `cfg` follows utils.analysis_block: the yaml at
    `config` when given, else the run's config.yaml as stage 3 left it, else
    the checkpoint's."""
    from cmsi.data import subset
    from cmsi.models import predict
    from cmsi.utils import (
        analysis_block,
        dataset_path,
        load_checkpoint,
        load_dataset,
        run_dir,
    )

    out = run_dir(run, create=False)
    model, cfg, _, splits = load_checkpoint(out / "model.pt")
    cfg, _ = analysis_block(cfg, out, config)
    d_full, _ = load_dataset(dataset_path((out / "dataset.txt").read_text().strip()))
    d = subset(d_full, splits["test"])
    return predict(model, d["X"]), d, list(d_full["target_names"]), cfg


def control_residuals(control):
    """Per-trial residuals (network - analytical) of a p_common = 1 control on
    its test split, per output. Their std is sigma_out (SS9.1)."""
    pred, d, names, cfg = load_run(control)
    if cfg["generative"]["p_common"] < 1.0:
        print(f"warning: control '{control}' has p_common = "
              f"{cfg['generative']['p_common']}, not 1 -- its residuals contain "
              f"causal misweighting, so sigma_out will be too large")
    target = np.stack([d[k] for k in names], axis=1)
    return pred - target, names


# --------------------------------------------------------------------------- #
# the per-run analysis
# --------------------------------------------------------------------------- #
def implied_weights(pred, d, names, sig_out):
    """The hybrid read of each channel on every trial ->
    {channel: (w, nominal sd, flags)} (analysis.hybrid_weight)."""
    out = {}
    for key, mu, var, seg_mu, seg_var in CHANNELS:
        i_mu, i_var = names.index(mu), names.index(var)
        delta = d["fused_mu"] - d[seg_mu]
        out[key] = hybrid_weight(pred[:, i_var], d["fused_var"], d[seg_var], delta,
                                 pred[:, i_mu], d[seg_mu], float(sig_out[i_mu]),
                                 float(sig_out[i_var]))
    return out


def weight_analysis(pred, d, names, sig_out, acfg, grid=GRID, min_count=MIN_COUNT,
                    max_se=MAX_SE):
    """Everything about the implied weight on one run's test split.

    `sig_out` is the read-out noise per output, measured on a p_common = 1
    control (control_residuals). Returns (metrics, arrays); per channel
    (`reads`), on every trial:

        w, sigma_w, flags     the hybrid read, its nominal sd and the output
                              each trial's weight came from (arrays)
        outside, sd_vs_posterior (and its root / ratio parts), corr, the
        shares of trials from the variance root, the position ratio and the
        variance peak
        compare               slope / R^2 / RMSE against the posterior
        weight_regression     the weight regressed on the posterior (slope 1,
                              intercept 0 for an optimal observer)
        position_regression, variance_regression
                              the two no-division headlines (SS7.1)
        curve, midpoint_deg   the weight against disparity on `grid` (mean
                              and standard error per bin) and the transition
                              midpoint fitted to it, next to the analytical
                              posterior's
        by_posterior          the weight in posterior bins

    plus `consistency`, the agreement of the two channels' reads trial by
    trial, and `analytical`, the posterior's own midpoint and sharpness.
    """
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    target = np.stack([d[k] for k in names], axis=1)
    disp, post = d["disparity"], d["post_c1"]

    metrics = {
        "n_test": int(len(pred)), "grid": grid.tolist(),
        "sigma_out": {names[i]: float(sig_out[i]) for i in range(len(names))},
        "residual_std_self": {names[i]: float((pred[:, i] - target[:, i]).std())
                              for i in range(len(names))},
        "accuracy": accuracy(pred, target, names),
        "reads": {},
    }
    arrays = {"disparity": disp, "post_c1": post}
    c_a, m_a, _ = mean_by_bin(disp, post, grid)
    arrays["analytical_curve"] = (c_a, m_a)
    mid_a, k_a = transition_fit(np.abs(disp), post)
    metrics["analytical"] = {"midpoint_deg": mid_a, "sharpness": k_a}

    reads = implied_weights(pred, d, names, sig_out)
    for key, mu, var, seg_mu, seg_var in CHANNELS:
        i_mu, i_var = names.index(mu), names.index(var)
        w, sw, flags = reads[key]
        delta = d["fused_mu"] - d[seg_mu]
        ok = np.isfinite(w)
        err = w - post
        curve = binned_weight(disp, w, grid, min_count=min_count, max_se=max_se)
        mid, k = transition_fit(np.abs(curve[0]), curve[1])
        metrics["reads"][key] = {
            "output": f"{var} root / {mu} ratio",
            "sigma_out": {"mu": float(sig_out[i_mu]), "var": float(sig_out[i_var])},
            "n": int(ok.sum()),
            "outside": float(np.mean(ok & ((w < -1) | (w > 2)))),
            "sd_vs_posterior": float(np.std(err[ok])),
            "sd_vs_posterior_root": float(np.std(err[ok & (flags != 1)])),
            "sd_vs_posterior_ratio": float(np.std(err[ok & (flags == 1)])),
            "corr_with_posterior": float(np.corrcoef(w[ok], post[ok])[0, 1]),
            "frac_variance_root": float(np.mean(flags == 0)),
            "frac_position_ratio": float(np.mean(flags == 1)),
            "frac_variance_peak": float(np.mean(flags == 2)),
            "sigma_w_median": float(np.nanmedian(sw)),
            "compare": compare(post, w),
            "weight_regression": weight_regression(w, post),
            "position_regression": position_regression(pred[:, i_mu], d[seg_mu],
                                                       d["fused_mu"], post),
            "variance_regression": variance_regression(pred[:, i_var], d["fused_var"],
                                                       d[seg_var], delta, post),
            "midpoint_deg": mid, "sharpness": k,
            "curve": {"centres": curve[0].tolist(), "w": curve[1].tolist(),
                      "n": curve[2].tolist(), "se": curve[3].tolist()},
        }
        bp = weight_by_posterior(w, post)
        metrics["reads"][key]["by_posterior"] = {kk: v.tolist() for kk, v in bp.items()}
        arrays[key] = {"w": w, "sigma_w": sw, "flags": flags, "delta": delta,
                       "curve": curve, "by_posterior": bp}
    metrics["consistency"] = weight_consistency(reads["vis"][0], reads["prop"][0])
    return metrics, arrays


def print_summary(metrics):
    """The per-run numbers, for the console."""
    a = metrics["analytical"]
    print(f"analytical midpoint {a['midpoint_deg']:.2f} deg, sharpness {a['sharpness']:.3f}")
    for key, r in metrics["reads"].items():
        wr, pr, vr = r["weight_regression"], r["position_regression"], r["variance_regression"]
        print(f"  {key}: {r['output']}; sigma_out mu {r['sigma_out']['mu']:.3f}, "
              f"var {r['sigma_out']['var']:.4f}")
        print(f"      {100 * r['frac_position_ratio']:.0f}% of trials from the position ratio, "
              f"{100 * r['outside']:.1f}% outside [-1, 2], sd vs posterior "
              f"{r['sd_vs_posterior']:.3f} (root {r['sd_vs_posterior_root']:.3f}, "
              f"ratio {r['sd_vs_posterior_ratio']:.3f}), corr {r['corr_with_posterior']:.3f}")
        print(f"      weight on the posterior: slope {wr['slope']:.3f} "
              f"[{wr['slope_ci95'][0]:.3f}, {wr['slope_ci95'][1]:.3f}], "
              f"intercept {wr['intercept']:+.3f}; position regression {pr['slope']:.3f}; "
              f"variance regression {vr['slope']:.3f}; midpoint {r['midpoint_deg']:.2f} deg")
    c = metrics["consistency"]
    print(f"  vis-prop consistency: corr {c['corr']:.3f}, mean|diff| {c['mean_abs_diff']:.3f} "
          f"on {c['n']} trials")


# --------------------------------------------------------------------------- #
# reliability levels
# --------------------------------------------------------------------------- #
def assign_levels(values, levels):
    """Split trials by a reliability variable.

    `levels` is either an int -- that many quantile bins, so every trial lands
    in exactly one bin and the bins hold equal numbers of trials -- or a list
    of centres, in which case each trial goes to the nearest centre (the
    convention of analysis.by_reliability and the config's reliability_levels).

    Returns (bin index per trial, list of (label, representative value)).
    """
    values = np.asarray(values, float)
    if np.isscalar(levels) or isinstance(levels, (int, np.integer)):
        k = int(levels)
        edges = np.quantile(values, np.linspace(0, 1, k + 1))
        idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, k - 1)
        info = [(f"{edges[b]:.2g}-{edges[b + 1]:.2g}",
                 float(np.median(values[idx == b]))) for b in range(k)]
    else:
        centres = np.asarray(levels, float)
        idx = np.abs(values[:, None] - centres[None, :]).argmin(1)
        info = [(f"~{c:g}", float(c)) for c in centres]
    return idx, info


def weight_by_reliability(pred, d, names, sig_out, acfg, input_key, levels,
                          grid=GRID, min_count=LEVEL_MIN_COUNT, max_se=LEVEL_MAX_SE):
    """Per reliability level of one input, the implied weight against
    disparity per channel (mean and standard error per bin), and the
    analytical posterior on the same bins.

    The transition midpoint is fitted to the binned curve for the network and
    for the analytical posterior alike, so the two are comparable; NaN when a
    level keeps too few bins for the logistic to be identifiable.

    Returns {level_label: {"value": v, "n": n, "analytical": (c, m),
                           "vis": (c, w, n, se), "prop": (...),
                           "midpoints": {...}}}
    """
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    reads = implied_weights(pred, d, names, sig_out)
    idx, info = assign_levels(d[INPUTS[input_key]], levels)
    out = {}
    for b, (label, value) in enumerate(info):
        m = idx == b
        if m.sum() < min_count:
            continue
        disp, post = d["disparity"][m], d["post_c1"][m]
        c_a, m_a, _ = mean_by_bin(disp, post, grid)
        entry = {"value": value, "n": int(m.sum()), "analytical": (c_a, m_a),
                 "grid": grid,
                 "midpoints": {"analytical": transition_fit(np.abs(c_a), m_a)[0]}}
        for key, *_ in CHANNELS:
            curve = binned_weight(disp, reads[key][0][m], grid, min_count=min_count,
                                  max_se=max_se)
            entry[key] = curve
            entry["midpoints"][key] = transition_fit(np.abs(curve[0]), curve[1])[0]
        out[label] = entry
    return out


# --------------------------------------------------------------------------- #
# figures: reliability
# --------------------------------------------------------------------------- #
def _ramp(n, name="viridis"):
    return plt.get_cmap(name)(np.linspace(0.15, 0.9, n))


def _on_grid(grid, centres, *values):
    """Map per-bin values onto the full grid, NaN where the bin was dropped,
    so that a dropped bin reads as a gap in the line and not as a segment
    drawn straight across it (as in scripts/05_prior_sweep.py)."""
    grid = np.asarray(grid, float)
    out = [np.full(grid.shape, np.nan) for _ in values]
    for c, *vals in zip(np.asarray(centres, float), *values, strict=False):
        j = int(np.abs(grid - c).argmin())
        for o, v in zip(out, vals, strict=False):
            o[j] = v
    return (grid, *out)


def fig_weight_by_reliability(curves, input_key):
    """Weight against signed disparity at each reliability level of one input,
    the visual channel (left) and the proprioceptive one (right). Solid: the
    network's implied weight, mean and 95 % interval per bin; dashed, same
    colour: the analytical posterior at that level. The two dashed families
    differ between inputs because the optimal weight depends on which cue is
    noisy."""
    labels = list(curves)
    colors = _ramp(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, mu, var, *_) in zip(axes, CHANNELS, strict=False):
        for label, color in zip(labels, colors, strict=False):
            e = curves[label]
            ca, ma = e["analytical"]
            ax.plot(ca, ma, "--", color=color, lw=1.1)
            g, w, se = _on_grid(e["grid"], e[key][0], e[key][1], e[key][3])
            ax.errorbar(g, w, yerr=1.96 * np.nan_to_num(se), fmt="o-", ms=3.5, lw=1.5,
                        capsize=2, color=color, label=f"{label} (n={e['n']})")
        ax.set(xlabel="body-frame disparity (deg)", ylim=(-0.1, 1.1),
               title=f"implied weight, {key} ({var} root / {mu} ratio)")
        ax.plot([], [], "--", color="0.3", label="analytical, same level")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                  title=INPUT_LABEL[input_key], title_fontsize=7)
    axes[0].set_ylabel("weight on fused estimate")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_midpoint_by_reliability(curves, input_key):
    """Transition midpoint against the reliability level: network (both
    channels) and analytical. The prediction is that a noisier cue tolerates
    more disparity before segregation, so the midpoint moves outward."""
    vals = np.array([e["value"] for e in curves.values()])
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot(vals, [e["midpoints"]["analytical"] for e in curves.values()], "--o",
            color=COLORS["analytical"], ms=4, label="analytical posterior")
    for (key, *_), color, mk in zip(CHANNELS, (COLORS["visual"], COLORS["prop"]),
                                    ("o", "s"), strict=False):
        ax.plot(vals, [e["midpoints"][key] for e in curves.values()], f"{mk}-",
                color=color, ms=4, label=f"network, {key} channel")
    ax.set(xlabel=f"{INPUT_LABEL[input_key]} (deg^2)",
           ylabel="transition midpoint (deg)",
           title="where fusion gives way to segregation")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def reliability_figures(pred, d, names, sig_out, acfg, inputs, levels, **kw):
    figs, table = {}, {}
    for n, key in enumerate(inputs, start=1):
        curves = weight_by_reliability(pred, d, names, sig_out, acfg, key, levels, **kw)
        if not curves:
            print(f"reliability: no level of {INPUTS[key]} holds enough trials; skipped")
            continue
        figs[f"{n:02d}a_weight_by_{INPUTS[key]}"] = fig_weight_by_reliability(curves, key)
        figs[f"{n:02d}b_midpoint_by_{INPUTS[key]}"] = fig_midpoint_by_reliability(curves, key)
        table[INPUTS[key]] = {
            label: {"value": e["value"], "n": e["n"],
                    "midpoint_analytical": e["midpoints"]["analytical"],
                    "midpoint_vis": e["midpoints"]["vis"],
                    "midpoint_prop": e["midpoints"]["prop"]}
            for label, e in curves.items()}
    return figs, table


# --------------------------------------------------------------------------- #
# figures: the pipeline's weight figures for one run
# --------------------------------------------------------------------------- #
def baseline_figures(pred, d, names, sig_out, acfg, arrays, metrics):
    """04_fusion_weight, 05_fusion_weight_by_reliability, 08v_variance_regression,
    15_weight_vs_posterior and 16_weight_distribution exactly as the main
    pipeline draws them (viz.results), from a weight_analysis result."""
    w, w_prop = arrays["vis"]["w"], arrays["prop"]["w"]
    grid = acfg["disparity_grid"]
    curves = by_reliability(d["disparity"], w, d["sig2_vis"], acfg["reliability_levels"], grid)
    i = names.index("var_vis")
    r = metrics["reads"]["vis"]
    return {
        "04_fusion_weight": viz_results.fusion_weight_curve(
            d["disparity"], w, d["post_c1"], grid, w_prop=w_prop),
        "05_fusion_weight_by_reliability": viz_results.fusion_weight_by_reliability(curves),
        "08v_variance_regression": viz_results.variance_regression_figure(
            pred[:, i], d["fused_var"], d["seg_vis_var"], arrays["vis"]["delta"], d["post_c1"],
            r["variance_regression"], w, r["weight_regression"], "var_vis",
            flags=arrays["vis"]["flags"]),
        "15_weight_vs_posterior": viz_results.weight_vs_posterior(
            arrays["vis"]["by_posterior"], "vis"),
        "16_weight_distribution": viz_results.hybrid_weight_distribution(
            w, w_prop, d["post_c1"], flags_vis=arrays["vis"]["flags"],
            flags_prop=arrays["prop"]["flags"]),
    }


# --------------------------------------------------------------------------- #
# config overrides from the command line
# --------------------------------------------------------------------------- #
def parse_overrides(items):
    """['rf_width=4', 'n_vis=100'] -> {'rf_width': 4.0, 'n_vis': 100}, values
    parsed as yaml so lists and floats work: --set hidden=[32,32]."""
    import yaml

    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out
