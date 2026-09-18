"""The implied-weight investigation.

The network never outputs a weight. Every weight in this project is *implied*:
recovered from where the network's position estimate sits between the two
causal hypotheses. That recovery has three moving parts, and this experiment
looks at each of them on its own, away from the main pipeline:

    sigma_out     the read-out noise, one number per output, measured on a
                  p_common = 1 control where the target is single-valued
    sigma_w       sigma_out / |Delta| per trial -- how readable the weight is
                  on that trial, which the sigma_w criterion turns into a filter
    the weight    per trial (a ratio), per disparity bin (least squares), per
                  posterior bin (both estimators side by side), per reliability
                  level, and per network

Everything is computed on a network's stored test split, exactly as stage 3
does, but written under results/experiments/implied_weight/<name>/ so that
nothing here can disturb results/<run>/. When a variant looks right, its
config goes into configs/ and the whole pipeline is re-run.

Entry points (scripts/06_implied_weight.py):

    figures       the three per-run figures this grew out of, for one run
    sigma         what sigma_out and sigma_w are on one run, and what the
                  criterion keeps
    reliability   the weight against disparity at many levels of each input's
                  reliability, read from either output
    train         train variants (other hidden sizes, other input encodings,
                  anything `tweak` can reach), each with its own control for
                  sigma_out, and compare them
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.accuracy import accuracy
from cmsi.analysis.causal import (
    binned_implied_weight,
    fusion_weight,
    implied_weight_by_posterior,
    mean_by_bin,
    position_regression,
    sigma_w,
    transition_fit,
)
from cmsi.viz import results as viz_results
from cmsi.viz.style import COLORS, SIZE, label_panels

EXPERIMENT = "implied_weight"

# The disparity grid and the guards on the per-bin least-squares weight, as
# in scripts/05_prior_sweep.py: 18 bins from -30 to +30 deg, coarse near zero
# where the two hypotheses coincide and no weight is identifiable. The
# pipeline's own figure 05 keeps the config's disparity_grid; everything new
# here uses this one, which is what the manuscript's prior-sweep figure uses.
GRID = [-30, -24, -19, -15, -12, -9, -6, -3.5, -1.5,
        1.5, 3.5, 6, 9, 12, 15, 19, 24, 30]
MIN_COUNT = 25
MAX_SE = 0.05
# Per reliability level a curve rests on a fifth of the trials or less, so the
# guards are relaxed there; a bin's error bar still says how good it is.
LEVEL_MIN_COUNT = 10
LEVEL_MAX_SE = 0.1

READS = (("vis", "mu_vis", "seg_vis_mu"), ("prop", "mu_prop", "seg_prop_mu"))
INPUTS = {"vis": "sig2_vis", "prop": "sig2_prop", "eye": "sig2_eye"}
INPUT_LABEL = {"vis": "visual noise sigma2_vis",
               "prop": "proprioceptive noise sigma2_prop",
               "eye": "eye-position noise sigma2_eye"}


# --------------------------------------------------------------------------- #
# where things go
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/implied_weight/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def experiment_data_dir():
    """data/experiments/implied_weight/ -- datasets the variants train on."""
    from cmsi.utils.paths import DATA

    path = DATA / "experiments" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# loading an existing run
# --------------------------------------------------------------------------- #
def load_run(run):
    """(pred, d_test, names, cfg) for results/<run>/, on its stored test split."""
    from cmsi.data import subset
    from cmsi.models import predict
    from cmsi.utils import dataset_path, load_checkpoint, load_dataset, run_dir

    out = run_dir(run, create=False)
    model, cfg, _, splits = load_checkpoint(out / "model.pt")
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
def weight_analysis(pred, d, names, sig_out, acfg, grid=GRID,
                    min_count=MIN_COUNT, max_se=MAX_SE):
    """Everything the investigation needs from one network on one test split.

    Returns (metrics, arrays). `metrics` is json-able and goes to metrics.json;
    `arrays` holds the per-trial and per-bin arrays the figures draw.

    Two transition midpoints are reported per read, because the pipeline and
    the sweep use different estimators and the difference is part of what this
    experiment is for:

        midpoint_filtered   logistic fitted to the sigma_w-filtered per-trial
                            ratio, exactly as stage 3 and the sweep do
        midpoint_ls         logistic fitted to the least-squares per-bin
                            weight, which uses every trial
    """
    acfg = dict(acfg)
    crit = float(acfg.get("sigma_w_criterion", 0.1))
    min_sep = float(acfg.get("min_separation", 1.0))
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    target = np.stack([d[k] for k in names], axis=1)
    disp, post = d["disparity"], d["post_c1"]

    metrics = {
        "sigma_w_criterion": crit, "min_separation": min_sep,
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

    for key, out_name, seg_name in READS:
        i, seg = names.index(out_name), d[seg_name]
        delta = d["fused_mu"] - seg
        so = float(sig_out[i])
        w = fusion_weight(pred[:, i], seg, d["fused_mu"], min_sep)
        sw = sigma_w(so, delta)
        w_f = np.where(sw < crit, w, np.nan)
        curve = binned_implied_weight(pred[:, i], seg, d["fused_mu"], disp, grid,
                                      min_count=min_count, sigma_out=so, max_se=max_se)
        by_post = implied_weight_by_posterior(
            pred[:, i], seg, d["fused_mu"], post, so,
            sigma_w_criterion=crit, min_separation=min_sep)
        mid_f, k_f = transition_fit(np.abs(disp), w_f)
        mid_ls, k_ls = transition_fit(np.abs(curve[0]), curve[1])
        reg = position_regression(pred[:, i], seg, d["fused_mu"], post)

        # the ratio on EVERY trial with Delta != 0: no min_separation, no
        # sigma_w criterion. This is what the network's estimated weight looks
        # like before anything is done to it, and it is what the distribution
        # figure draws.
        with np.errstate(divide="ignore", invalid="ignore"):
            w_raw = np.where(delta != 0, (pred[:, i] - seg) / delta, np.nan)
        wr = w_raw[np.isfinite(w_raw)]
        finite = np.isfinite(sw)
        metrics["reads"][key] = {
            "w_raw": {"n": int(wr.size), "median": float(np.median(wr)),
                      "iqr": [float(np.percentile(wr, 25)), float(np.percentile(wr, 75))],
                      "mean": float(wr.mean()), "std": float(wr.std()),
                      "frac_outside_-1_2": float(np.mean((wr < -1) | (wr > 2)))},
            "net_minus_seg_std": float((pred[:, i] - seg).std()),
            "delta_std": float(delta.std()),
            "output": out_name,
            "sigma_out": so,
            "frac_readable": float(np.mean(sw < crit)),
            "n_readable": int(np.sum(sw < crit)),
            "sigma_w_median": float(np.median(sw[finite])),
            "sigma_w_iqr": [float(np.percentile(sw[finite], 25)),
                            float(np.percentile(sw[finite], 75))],
            "delta_threshold_deg": so / crit,
            "midpoint_filtered_deg": mid_f, "sharpness_filtered": k_f,
            "midpoint_ls_deg": mid_ls, "sharpness_ls": k_ls,
            "position_regression": reg,
            "curve": {"centres": curve[0].tolist(), "w": curve[1].tolist(),
                      "n": curve[2].tolist(), "se": curve[3].tolist()},
            "by_posterior": {k: v.tolist() for k, v in by_post.items()},
        }
        arrays[key] = {"w": w, "w_filtered": w_f, "sigma_w": sw, "delta": delta,
                       "resid": pred[:, i] - target[:, i],
                       "w_raw": w_raw, "net_minus_seg": pred[:, i] - seg,
                       "curve": curve, "by_post": by_post}
    return metrics, arrays


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
    """Per reliability level of one input, the least-squares weight curve read
    from each output, and the analytical posterior on the same bins.

    The transition midpoint is fitted to the binned curve for the network and
    for the analytical posterior alike, so the two are comparable; NaN when a
    level keeps too few bins for the logistic to be identifiable.

    Returns {level_label: {"value": v, "n": n, "analytical": (c, m),
                           "vis": (c, w, n, se), "prop": (...),
                           "midpoints": {...}}}
    """
    acfg = dict(acfg)
    grid = np.asarray(acfg["disparity_grid"] if grid is None else grid, float)
    var = d[INPUTS[input_key]]
    idx, info = assign_levels(var, levels)
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
        for key, out_name, seg_name in READS:
            i = names.index(out_name)
            curve = binned_implied_weight(
                pred[m, i], d[seg_name][m], d["fused_mu"][m], disp, grid,
                min_count=min_count, sigma_out=float(sig_out[i]), max_se=max_se)
            entry[key] = curve
            entry["midpoints"][key] = transition_fit(np.abs(curve[0]), curve[1])[0]
        out[label] = entry
    return out


# --------------------------------------------------------------------------- #
# figures: sigma_out and sigma_w
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


def fig_sigma_residuals(arrays, ctrl_resid, names, metrics):
    """The run's own per-trial residuals, network - analytical target, on
    every test trial, per position output. Their std is what stage 3 stores
    as residual_std. The p_common = 1 control's residuals are drawn as a thin
    outline for reference: their std is sigma_out, and the difference between
    the two widths is the causal misweighting the run carries on top of its
    read-out noise. Nothing is filtered; the axis is clipped at +/-4 sd for
    display and the fraction beyond it is printed."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        i = names.index(out_name)
        own = arrays[key]["resid"]
        so = metrics["sigma_out"][out_name]
        own_sd = metrics["residual_std_self"][out_name]
        lim = 4 * max(so, own_sd)
        bins = np.linspace(-lim, lim, 81)
        beyond = np.mean(np.abs(own) > lim)
        ax.hist(np.clip(own, -lim, lim), bins=bins, density=True,
                color=COLORS["network"], alpha=0.75,
                label=f"this run, all {own.size} trials (sd {own_sd:.3f})")
        ax.hist(np.clip(ctrl_resid[:, i], -lim, lim), bins=bins, density=True,
                histtype="step", lw=1.3, color="0.25",
                label=f"p_common=1 control (sd {so:.3f} = sigma_out)")
        for s, c, ls in ((own_sd, COLORS["network"], "-"), (so, "0.25", "--")):
            ax.axvline(s, color=c, ls=ls, lw=1.0)
            ax.axvline(-s, color=c, ls=ls, lw=1.0)
        ax.set(xlabel=f"{out_name}: network - analytical (deg)", ylabel="density",
               title=f"{out_name}: residual sd {own_sd:.3f} deg")
        if beyond > 0:
            ax.text(0.98, 0.97, f"{100 * beyond:.1f}% beyond +/-{lim:.1f}",
                    transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color="0.35")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_weight_distribution(arrays, metrics, lim=(-1.0, 2.0)):
    """The network's estimated weight on every trial, w = (network - seg) /
    Delta, with no filter of any kind. A Bayes-optimal observer would put
    each trial at its posterior, so the analytical posterior's own
    distribution is drawn as an outline for comparison. The ratio is
    unbounded where Delta is small, so the axis is clipped to `lim` and the
    fraction of trials outside it is printed."""
    post = arrays["post_c1"]
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    bins = np.linspace(lim[0], lim[1], 91)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        w = arrays[key]["w_raw"]
        w = w[np.isfinite(w)]
        s = metrics["reads"][key]["w_raw"]
        ax.hist(np.clip(w, *lim), bins=bins, density=True, color=COLORS["network"],
                alpha=0.75, label=f"network, all {w.size} trials")
        ax.hist(post, bins=bins, density=True, histtype="step", lw=1.3,
                color=COLORS["analytical"], label="analytical posterior")
        ax.axvline(s["median"], color=COLORS["network"], lw=1.0)
        ax.set(xlabel=f"w = (network - seg) / Delta, from {out_name}", xlim=lim,
               title=f"{out_name}: median {s['median']:.2f} "
                     f"(IQR {s['iqr'][0]:.2f} to {s['iqr'][1]:.2f})")
        ax.text(0.02, 0.97,
                f"{100 * s['frac_outside_-1_2']:.1f}% of trials outside [{lim[0]:g}, {lim[1]:g}]"
                f"\n(piled into the edge bins)",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.35")
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("density")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_network_minus_seg(arrays, metrics):
    """How far the network's estimate sits from the segregated solution on
    every trial, network - seg, next to how far the fused solution sits from
    it, Delta = fused - seg. The first is w * Delta trial by trial, so a
    network that fuses puts mass where Delta is and a network that
    segregates piles up at zero. No filter."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        nms, delta = arrays[key]["net_minus_seg"], arrays[key]["delta"]
        lim = 4 * max(nms.std(), delta.std())
        bins = np.linspace(-lim, lim, 81)
        ax.hist(np.clip(delta, -lim, lim), bins=bins, density=True, color="0.7",
                alpha=0.8, label=f"Delta = fused - seg (sd {delta.std():.2f})")
        ax.hist(np.clip(nms, -lim, lim), bins=bins, density=True, histtype="step",
                lw=1.5, color=COLORS["network"],
                label=f"network - seg (sd {nms.std():.2f})")
        ax.set(xlabel=f"degrees, {out_name}", ylabel="density (log)", yscale="log",
               title=f"{out_name}: network - seg, all {nms.size} trials")
        ax.legend(fontsize=7.5, frameon=False)
    label_panels(axes)
    fig.tight_layout()
    return fig



def fig_sigma_w_vs_delta(arrays, metrics, n_show=4000, seed=0):
    """Per trial, against |Delta| = |fused - seg|: the network's actual error
    (left) and sigma_w (right), for each position output.

    Left: |network - analytical| does not grow with |Delta| -- the running
    median stays at about sigma_out -- which is what licenses sigma_w =
    sigma_out / |Delta| as the per-trial uncertainty of the weight. Right:
    that ratio, coloured by the analytical posterior, with the criterion and
    the |Delta| it implies. Trials at high posterior sit at small |Delta|,
    which is why the filter removes the fusion end of every curve."""
    rng = np.random.default_rng(seed)
    crit = metrics["sigma_w_criterion"]
    fig, axes = plt.subplots(2, 2, figsize=SIZE["quad"], constrained_layout=True)
    for row, (key, out_name, _) in zip(axes, READS, strict=False):
        a = arrays[key]
        ad = np.abs(a["delta"])
        ok = np.isfinite(a["sigma_w"]) & (ad > 0)
        pick = np.flatnonzero(ok)
        if pick.size > n_show:
            pick = rng.choice(pick, n_show, replace=False)
        so = metrics["sigma_out"][out_name]
        thr = metrics["reads"][key]["delta_threshold_deg"]

        ax = row[0]
        ax.scatter(ad[pick], np.abs(a["resid"][pick]), s=3, color=COLORS["network"],
                   alpha=0.3, linewidths=0, rasterized=True)
        edges = np.quantile(ad[ok], np.linspace(0, 1, 13))
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            m = ok & (ad >= lo) & (ad < hi)
            if m.sum() >= 20:
                mids.append(np.median(ad[m]))
                meds.append(np.median(np.abs(a["resid"][m])))
        ax.plot(mids, meds, "o-", color="0.15", ms=3.5, lw=1.4, label="running median")
        ax.axhline(so * np.sqrt(2 / np.pi), color="0.15", ls="--", lw=1.1,
                   label=f"E|error| for sigma_out {so:.2f}")
        ax.set(xscale="log", yscale="log", ylim=(1e-3, None),
               xlabel=f"|Delta| (deg), {out_name}", ylabel="|network - analytical| (deg)",
               title="the error does not grow with |Delta|")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

        ax = row[1]
        sc = ax.scatter(ad[pick], a["sigma_w"][pick], c=arrays["post_c1"][pick], s=4,
                        cmap="viridis", vmin=0, vmax=1, alpha=0.7, linewidths=0,
                        rasterized=True)
        ax.axhline(crit, color="0.2", ls="--", lw=1.1)
        ax.axvline(thr, color="0.2", ls=":", lw=1.1)
        ax.text(0.97, 0.95, f"criterion {crit:g}  <=>  |Delta| > {thr:.1f} deg\n"
                f"keeps {100 * metrics['reads'][key]['frac_readable']:.0f}% of trials",
                transform=ax.transAxes, fontsize=8, va="top", ha="right")
        ax.set(xscale="log", yscale="log", xlabel=f"|Delta| (deg), {out_name}",
               ylabel="sigma_w = sigma_out / |Delta|",
               title="readability of the weight, per trial")
    cb = fig.colorbar(sc, ax=axes[:, 1], shrink=0.8, pad=0.02)
    cb.set_label("analytical posterior p(C=1|x)", fontsize=9)
    label_panels(axes)
    return fig


def fig_sigma_w_vs_posterior(arrays, metrics, n_bins=10):
    """Median sigma_w (with interquartile band) per posterior bin, and the
    fraction of the bin the criterion keeps. This is the figure that explains
    why the filtered estimator cannot see the fusion end of the curve."""
    crit = metrics["sigma_w_criterion"]
    post = arrays["post_c1"]
    edges = np.linspace(0, 1, n_bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        sw = arrays[key]["sigma_w"]
        c, med, lo, hi, kept = [], [], [], [], []
        for b in range(n_bins):
            top = post <= edges[b + 1] if b == n_bins - 1 else post < edges[b + 1]
            m = (post >= edges[b]) & top & np.isfinite(sw)
            if m.sum() < 5:
                continue
            c.append(post[m].mean())
            med.append(np.median(sw[m]))
            lo.append(np.percentile(sw[m], 25))
            hi.append(np.percentile(sw[m], 75))
            kept.append(np.mean(sw[m] < crit))
        c, med, lo, hi, kept = map(np.asarray, (c, med, lo, hi, kept))
        ax.fill_between(c, lo, hi, color=COLORS["network"], alpha=0.2, lw=0)
        ax.plot(c, med, "o-", color=COLORS["network"], ms=4, label="median sigma_w (IQR band)")
        ax.axhline(crit, color="0.2", ls="--", lw=1.1, label=f"criterion {crit:g}")
        ax.set(yscale="log", xlabel="analytical posterior p(C=1|x)",
               ylabel="sigma_w per trial", xlim=(0, 1), title=f"read from {out_name}")
        ax2 = ax.twinx()
        ax2.plot(c, 100 * kept, ":", color="0.45", lw=1.3)
        ax2.set_ylim(-4, 104)
        ax2.set_ylabel("trials kept by the criterion (%)", color="0.35", fontsize=9)
        ax2.tick_params(axis="y", colors="0.35", labelsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_kept_vs_criterion(arrays, metrics):
    """Fraction of trials the sigma_w filter keeps, as the criterion is varied,
    overall and within the ambiguous zone (0.2 < posterior < 0.8)."""
    crit = metrics["sigma_w_criterion"]
    grid = np.logspace(-2, 0, 60)
    post = arrays["post_c1"]
    amb = (post > 0.2) & (post < 0.8)
    fig, ax = plt.subplots(figsize=SIZE["single"])
    for (key, out_name, _), color in zip(READS, (COLORS["visual"], COLORS["prop"]), strict=False):
        sw = arrays[key]["sigma_w"]
        ax.plot(grid, [np.mean(sw < g) for g in grid], "-", color=color,
                label=f"{out_name}, all trials")
        ax.plot(grid, [np.mean(sw[amb] < g) for g in grid], "--", color=color,
                label=f"{out_name}, 0.2 < posterior < 0.8")
    ax.axvline(crit, color="0.2", ls=":", lw=1.1)
    ax.text(crit, 0.02, f" criterion {crit:g}", fontsize=8, ha="left")
    ax.set(xscale="log", xlabel="sigma_w criterion", ylabel="fraction of trials kept",
           ylim=(0, 1.02), title="what the filter keeps")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def sigma_figures(arrays, ctrl_resid, names, metrics):
    """Nothing here is filtered. The sigma_w criterion appears only as a
    reference line, and figure 04 shows what applying it *would* keep."""
    return {
        "01_residuals_all_trials": fig_sigma_residuals(arrays, ctrl_resid, names, metrics),
        "02_sigma_w_vs_delta": fig_sigma_w_vs_delta(arrays, metrics),
        "03_sigma_w_vs_posterior": fig_sigma_w_vs_posterior(arrays, metrics),
        "04_kept_vs_criterion": fig_kept_vs_criterion(arrays, metrics),
        "05_weight_distribution_all_trials": fig_weight_distribution(arrays, metrics),
        "06_network_minus_seg_all_trials": fig_network_minus_seg(arrays, metrics),
    }


# --------------------------------------------------------------------------- #
# figures: reliability
# --------------------------------------------------------------------------- #
def fig_weight_by_reliability(curves, input_key):
    """Weight against signed disparity at each reliability level of one input,
    read from mu_vis (left) and mu_prop (right). Solid: the network, by least
    squares within the bin; dashed, same colour: the analytical posterior at
    that level. The two dashed families differ between inputs because the
    optimal weight depends on which cue is noisy."""
    labels = list(curves)
    colors = _ramp(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"], sharey=True)
    for ax, (key, out_name, _) in zip(axes, READS, strict=False):
        for label, color in zip(labels, colors, strict=False):
            e = curves[label]
            ca, ma = e["analytical"]
            ax.plot(ca, ma, "--", color=color, lw=1.1)
            g, w, se = _on_grid(e["grid"], e[key][0], e[key][1], e[key][3])
            ax.errorbar(g, w, yerr=1.96 * np.nan_to_num(se), fmt="o-", ms=3.5, lw=1.5,
                        capsize=2, color=color, label=f"{label} (n={e['n']})")
        ax.set(xlabel="body-frame disparity (deg)", ylim=(-0.1, 1.1),
               title=f"read from {out_name}")
        ax.plot([], [], "--", color="0.3", label="analytical, same level")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                  title=INPUT_LABEL[input_key], title_fontsize=7)
    axes[0].set_ylabel("weight on fused estimate")
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_midpoint_by_reliability(curves, input_key):
    """Transition midpoint against the reliability level: network (both
    reads) and analytical. The prediction is that a noisier cue tolerates more
    disparity before segregation, so the midpoint moves outward."""
    vals = np.array([e["value"] for e in curves.values()])
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot(vals, [e["midpoints"]["analytical"] for e in curves.values()], "--o",
            color=COLORS["analytical"], ms=4, label="analytical posterior")
    for (key, out_name, _), color, mk in zip(READS, (COLORS["visual"], COLORS["prop"]),
                                             ("o", "s"), strict=False):
        ax.plot(vals, [e["midpoints"][key] for e in curves.values()], f"{mk}-",
                color=color, ms=4, label=f"network, read from {out_name}")
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
# figures: the three per-run figures this experiment grew out of
# --------------------------------------------------------------------------- #
def baseline_figures(pred, d, names, sig_out, acfg, arrays):
    """05_fusion_weight_by_reliability, 15/16_weight_vs_post_* exactly as the
    main pipeline draws them, plus the least-squares version of 05."""
    from cmsi.analysis.causal import by_reliability

    figs = {}
    curves = by_reliability(d["disparity"], arrays["vis"]["w"], d["sig2_vis"],
                            acfg["reliability_levels"], acfg["disparity_grid"])
    figs["05_fusion_weight_by_reliability"] = viz_results.fusion_weight_by_reliability(curves)
    figs["15_weight_vs_post_ratio"] = viz_results.implied_weight_vs_posterior(
        arrays["vis"]["by_post"], "ratio")
    figs["16_weight_vs_post_leastsq"] = viz_results.implied_weight_vs_posterior(
        arrays["vis"]["by_post"], "least_squares")
    figs["04_fusion_weight"] = viz_results.fusion_weight_curve(
        d["disparity"], arrays["vis"]["w"], d["post_c1"], acfg["disparity_grid"],
        w_prop=arrays["prop"]["w"])
    return figs


# --------------------------------------------------------------------------- #
# variants: train, evaluate, compare
# --------------------------------------------------------------------------- #
def parse_hidden(spec):
    """'64' -> [64, 64];  '32x128' -> [32, 128];  '16,16' -> [16, 16]."""
    parts = [int(p) for p in str(spec).replace(",", "x").split("x") if p]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or min(parts) < 1:
        raise ValueError(f"hidden spec must be N or AxB, got {spec!r}")
    return parts


def variant_name(hidden):
    return f"h{hidden[0]}x{hidden[1]}"


def parse_overrides(items):
    """['rf_width=4', 'n_vis=100'] -> {'rf_width': 4.0, 'n_vis': 100}, values
    parsed as yaml so lists and floats work: --set hidden=[32,32]."""
    import yaml

    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out



def variant_datasets(cfg, exp_dir, regen=False, verbose=True):
    """The causal dataset and its p_common = 1 control for one experiment,
    generated once from the experiment's config and cached next to it.

    A cached dataset is reused only if it was drawn from the same generative
    and encoding parameters, the same number of trials and the same seed as
    the config now asked for; otherwise this stops and says so, because a
    variant trained on the wrong trials would be compared to the others as if
    it were not. Use --regen to redraw, or a new --name for a new experiment.
    """
    from cmsi.data import make_dataset
    from cmsi.utils import load_dataset, save_dataset, tweak

    def signature(c):
        return {"generative": c["generative"], "encoding": c["encoding"],
                "n_trials": c["training"]["n_trials"], "seed": c["seed"]}

    ddir = experiment_data_dir()
    paths = {"causal": ddir / f"{exp_dir.name}_causal.npz",
             "control": ddir / f"{exp_dir.name}_control.npz"}
    cfgs = {"causal": cfg, "control": tweak(cfg, p_common=1.0)}
    data = {}
    for kind, path in paths.items():
        if path.exists() and not regen:
            d, stored = load_dataset(path)
            if signature(stored) != signature(cfgs[kind]):
                raise SystemExit(
                    f"{path.name} was generated from a different config than the "
                    f"one now requested (generative/encoding/n_trials/seed differ). "
                    f"Re-run with --regen to redraw the datasets of experiment "
                    f"'{exp_dir.name}', or use a new --name.")
            data[kind] = (d, stored)
            if verbose:
                print(f"{kind}: reusing {path.relative_to(ddir.parents[1])}")
            continue
        if verbose:
            print(f"{kind}: generating {cfgs[kind]['training']['n_trials']} trials "
                  f"(p_common={cfgs[kind]['generative']['p_common']}) -> {path.name}")
        d = make_dataset(cfgs[kind])
        save_dataset(d, cfgs[kind], path)
        data[kind] = (d, cfgs[kind])
    return data


def train_variant(hidden, data, exp_dir, epochs=None, seed=None, verbose=True):
    """Train the causal network AND its control at one hidden size, evaluate
    both on their test splits, and write everything under
    <exp>/variants/<name>/. Returns (metrics, arrays)."""
    from cmsi.data import subset
    from cmsi.models import predict, train
    from cmsi.utils import save_checkpoint, save_config, save_json, tweak

    name = variant_name(hidden)
    vdir = exp_dir / "variants" / name
    (vdir / "figures").mkdir(parents=True, exist_ok=True)

    preds, tests, cfgs = {}, {}, {}
    for kind in ("causal", "control"):
        d, cfg = data[kind]
        over = {"hidden": list(hidden)}
        if epochs is not None:
            over["epochs"] = int(epochs)
        if seed is not None:
            over["seed"] = int(seed)
        cfg = tweak(cfg, **over)
        if verbose:
            print(f"\n[{name}] training the {kind} network "
                  f"(hidden {hidden}, {cfg['training']['epochs']} epochs)")
        model, history, splits = train(d, cfg, verbose=verbose)
        save_checkpoint(model, cfg, history, splits,
                        vdir / ("model.pt" if kind == "causal" else "control.pt"))
        cfgs[kind] = cfg
        tests[kind] = subset(d, splits["test"])
        preds[kind] = predict(model, tests[kind]["X"])
    save_config(cfgs["causal"], vdir / "config.yaml")

    names = list(data["causal"][0]["target_names"])
    ctrl_target = np.stack([tests["control"][k] for k in names], axis=1)
    ctrl_resid = preds["control"] - ctrl_target
    sig_out = ctrl_resid.std(axis=0)

    metrics, arrays = weight_analysis(preds["causal"], tests["causal"], names,
                                      sig_out, cfgs["causal"]["analysis"])
    metrics.update(variant=name, hidden=list(hidden),
                   n_units=int(sum(hidden)),
                   epochs=int(cfgs["causal"]["training"]["epochs"]),
                   sigma_out_source="control trained at the same hidden size")
    save_json(metrics, vdir / "metrics.json")
    np.savez_compressed(vdir / "arrays.npz",
                        control_resid=ctrl_resid,
                        **{f"{k}_{kk}": v for k in ("vis", "prop")
                           for kk, v in arrays[k].items()
                           if isinstance(v, np.ndarray)},
                        disparity=arrays["disparity"], post_c1=arrays["post_c1"])
    return metrics, arrays, ctrl_resid, names, preds["causal"], tests["causal"], cfgs["causal"]


def load_variants(exp_dirs):
    """Every variants/<name>/metrics.json under one experiment, or under
    several (a list of experiment dirs), sorted by units then name.

    With several experiments the rows are labelled "<experiment>/<hidden>",
    so variants trained on different configs -- another encoding, another
    generative model -- can be put on one set of comparison figures. They
    are then compared on their own test splits, not on shared trials, and
    the analytical curve is drawn once per distinct dataset.
    """
    from cmsi.utils import load_json

    exp_dirs = [exp_dirs] if not isinstance(exp_dirs, (list, tuple)) else list(exp_dirs)
    rows = []
    for exp_dir in exp_dirs:
        for path in sorted((exp_dir / "variants").glob("*/metrics.json")):
            m = load_json(path)
            m["_dir"] = path.parent
            m["experiment"] = exp_dir.name
            h = m.get("hidden", [])
            hidden = f"{h[0]}x{h[1]}" if len(h) == 2 else m.get("variant", "?")
            m["label"] = f"{exp_dir.name}/{hidden}" if len(exp_dirs) > 1 else hidden
            rows.append(m)
    rows.sort(key=lambda m: (m.get("experiment", ""), m.get("n_units", 0), m.get("variant", "")))
    return rows


def _variant_label(m):
    if "label" in m:
        return m["label"]
    h = m.get("hidden", [])
    return f"{h[0]}x{h[1]}" if len(h) == 2 else m.get("variant", "?")


def fig_variants_weight_curves(rows, read="vis"):
    """Least-squares weight against disparity, one curve per variant, with
    the analytical posterior dashed. Variants that share a dataset share one
    black analytical curve; when the rows come from several experiments
    (different configs, so different analytical curves) each variant's own
    analytical curve is dashed in that variant's colour."""
    colors = _ramp(len(rows), "plasma")
    multi = len({m.get("experiment") for m in rows}) > 1
    fig, ax = plt.subplots(figsize=SIZE["single"])
    drawn = []
    for m, color in zip(rows, colors, strict=False):
        r = m["reads"][read]["curve"]
        g, w, se = _on_grid(m["grid"], r["centres"], r["w"], r["se"])
        ax.errorbar(g, w, yerr=1.96 * np.nan_to_num(se), fmt="o-",
                    ms=3.5, lw=1.5, capsize=2, color=color,
                    label=f"{_variant_label(m)} (mid {m['reads'][read]['midpoint_ls_deg']:.1f})")
        mid_a = m["analytical"]["midpoint_deg"]
        key = (m.get("experiment"), round(mid_a or 0, 2))
        if key not in drawn:
            a = np.load(m["_dir"] / "arrays.npz")
            c, mm, _ = mean_by_bin(a["disparity"], a["post_c1"], np.asarray(m["grid"], float))
            tag = f"(mid {mid_a:.1f})" if mid_a is not None else ""
            ax.plot(c, mm, "--", color=color if multi else COLORS["analytical"], lw=1.1,
                    label=(f"analytical, {m['experiment']} {tag}" if multi
                           else f"analytical {tag}"))
            drawn.append(key)
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1),
           title=f"implied weight by {'configuration' if multi else 'network size'}, "
                 f"read from mu_{read}")
    ax.legend(fontsize=7, frameon=False,
              title="experiment/hidden" if multi else "hidden (SIL x MSL)", title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_variants_weight_vs_posterior(rows, read="vis"):
    """Least-squares weight per posterior bin, one curve per variant."""
    colors = _ramp(len(rows), "plasma")
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.plot([0, 1], [0, 1], "--", lw=1.1, color=COLORS["analytical"], label="Bayes-optimal")
    for m, color in zip(rows, colors, strict=False):
        b = m["reads"][read]["by_posterior"]
        slope = m["reads"][read]["position_regression"]["slope"]
        ax.errorbar(b["centres"], b["w_ls"], yerr=1.96 * np.asarray(b["se_ls"]),
                    fmt="o-", ms=3.5, lw=1.5, capsize=2, color=color,
                    label=f"{_variant_label(m)} (slope {slope:.2f})")
    ax.set(xlabel="analytical posterior p(C=1|x)", ylabel="implied weight (least squares)",
           xlim=(0, 1), ylim=(-0.05, 1.05), title=f"weight vs posterior by network size, mu_{read}")
    ax.legend(fontsize=7.5, frameon=False, title="hidden (SIL x MSL)", title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_variants_sigma(rows):
    """Left: sigma_out per variant (control) with the run's own residual std
    beside it. Right: the distribution of sigma_w per variant (median, IQR)
    and the fraction of trials the criterion keeps."""
    labels = [_variant_label(m) for m in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    ax = axes[0]
    wdt = 0.2
    for j, (key, out_name, _) in enumerate(READS):
        so = [m["sigma_out"][out_name] for m in rows]
        own = [m["residual_std_self"][out_name] for m in rows]
        col = COLORS["visual"] if key == "vis" else COLORS["prop"]
        ax.bar(x + (j - 0.5) * 2 * wdt - wdt / 2, so, wdt, color=col, label=f"sigma_out {out_name}")
        ax.bar(x + (j - 0.5) * 2 * wdt + wdt / 2, own, wdt, color=col, alpha=0.35,
               label=f"own residual std {out_name}")
    ax.set(xticks=x, xticklabels=labels, ylabel="deg", title="read-out noise by network size")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    crit = rows[0]["sigma_w_criterion"]
    for j, (key, out_name, _) in enumerate(READS):
        med = np.array([m["reads"][key]["sigma_w_median"] for m in rows])
        lo = np.array([m["reads"][key]["sigma_w_iqr"][0] for m in rows])
        hi = np.array([m["reads"][key]["sigma_w_iqr"][1] for m in rows])
        col = COLORS["visual"] if key == "vis" else COLORS["prop"]
        off = (j - 0.5) * 0.15
        ax.errorbar(x + off, med, yerr=[med - lo, hi - med], fmt="o", ms=4, capsize=3,
                    color=col, label=f"median sigma_w (IQR), {out_name}")
    ax.axhline(crit, color="0.2", ls="--", lw=1.1, label=f"criterion {crit:g}")
    ax.set(yscale="log", xticks=x, xticklabels=labels, ylabel="sigma_w per trial",
           title="readability of the weight")
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx()
    for j, (key, _, _) in enumerate(READS):
        kept = [100 * m["reads"][key]["frac_readable"] for m in rows]
        ax2.plot(x + (j - 0.5) * 0.15, kept, ":", color="0.45", lw=1.2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("trials kept by the criterion (%)", color="0.35", fontsize=9)
    ax2.tick_params(axis="y", colors="0.35", labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig



def fig_variants_summary(rows):
    """Four summaries per variant, ordered by total hidden units: transition
    midpoint (both estimators, both reads) against the analytical value; the
    position-regression slope; the read-out R^2 per output; and the fraction
    of trials the sigma_w criterion keeps."""
    x = np.arange(len(rows), dtype=float)
    labels = [f"{_variant_label(m)}\n({m['n_units']} units)" for m in rows]
    fig, axes = plt.subplots(2, 2, figsize=SIZE["quad"])
    reads = list(READS)
    cols = (COLORS["visual"], COLORS["prop"])

    ax = axes[0, 0]
    ax.plot(x, [m["analytical"]["midpoint_deg"] for m in rows], "--",
            color=COLORS["analytical"], label="analytical")
    for (key, out_name, _), col in zip(reads, cols, strict=False):
        ax.plot(x, [m["reads"][key]["midpoint_ls_deg"] for m in rows], "o-",
                color=col, ms=4, label=f"{out_name}, least squares")
        ax.plot(x, [m["reads"][key]["midpoint_filtered_deg"] for m in rows], "s:",
                color=col, ms=4, label=f"{out_name}, filtered ratio")
    ax.set(ylabel="transition midpoint (deg)", title="midpoint")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.axhline(1, color=COLORS["analytical"], ls="--", lw=1.1, label="Bayes-optimal")
    for (key, out_name, _), col in zip(reads, cols, strict=False):
        pr = [m["reads"][key]["position_regression"] for m in rows]
        s = np.array([q["slope"] for q in pr])
        lo = np.array([q["slope_ci95"][0] for q in pr])
        hi = np.array([q["slope_ci95"][1] for q in pr])
        ax.errorbar(x, s, yerr=[s - lo, hi - s], fmt="o-", ms=4, capsize=3,
                    color=col, label=out_name)
    ax.set(ylabel="position-regression slope", title="pull toward fusion")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    names = [a["output"] for a in rows[0]["accuracy"]]
    for j, name in enumerate(names):
        ax.plot(x, [m["accuracy"][j]["r2"] for m in rows], "o-", ms=4, label=name)
    ax.set(ylabel="read-out R^2 (test split)", title="did it learn the task")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    for (key, out_name, _), col in zip(reads, cols, strict=False):
        ax.plot(x, [100 * m["reads"][key]["frac_readable"] for m in rows], "o-",
                color=col, ms=4, label=out_name)
    ax.set(ylabel="trials kept (%)", ylim=(0, 100),
           title="readability (sigma_w criterion)")
    ax.legend(fontsize=7, frameon=False)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("hidden units, SIL x MSL (total)")
    label_panels(axes)
    fig.tight_layout()
    return fig


def variant_figures(rows):
    if not rows:
        return {}
    figs = {
        "A_weight_vs_disparity_by_variant": fig_variants_weight_curves(rows, "vis"),
        "B_weight_vs_posterior_by_variant": fig_variants_weight_vs_posterior(rows, "vis"),
        "C_sigma_by_variant": fig_variants_sigma(rows),
    }
    if len(rows) > 1:
        figs["D_summary_vs_units"] = fig_variants_summary(rows)
    return figs


def variants_table(rows):
    """The comparison as one row per variant, for metrics.json and the console."""
    out = []
    for m in rows:
        r = {"variant": m["variant"], "hidden": m["hidden"], "n_units": m["n_units"],
             "epochs": m.get("epochs"),
             "sigma_out_mu_vis": m["sigma_out"]["mu_vis"],
             "sigma_out_mu_prop": m["sigma_out"]["mu_prop"],
             "midpoint_analytical": m["analytical"]["midpoint_deg"]}
        for key in ("vis", "prop"):
            rd = m["reads"][key]
            r[f"midpoint_ls_{key}"] = rd["midpoint_ls_deg"]
            r[f"midpoint_filtered_{key}"] = rd["midpoint_filtered_deg"]
            r[f"slope_{key}"] = rd["position_regression"]["slope"]
            r[f"frac_readable_{key}"] = rd["frac_readable"]
            r[f"sigma_w_median_{key}"] = rd["sigma_w_median"]
        r["r2"] = {a["output"]: a["r2"] for a in m["accuracy"]}
        r["experiment"], r["label"] = m.get("experiment"), _variant_label(m)
        out.append(r)
    return out


def print_variants(rows):
    width = max(10, max(len(r["label"]) for r in variants_table(rows)))
    print(f"\n{'variant':>{width}} {'sig_out v':>9} {'sig_out p':>9} {'mid_ls v':>8} "
          f"{'mid_filt v':>10} {'mid_an':>7} {'slope v':>7} {'kept v':>6} {'R2 mu_vis':>9}")
    for r in variants_table(rows):
        print(f"{r['label']:>{width}} {r['sigma_out_mu_vis']:9.3f} {r['sigma_out_mu_prop']:9.3f} "
              f"{r['midpoint_ls_vis']:8.2f} {r['midpoint_filtered_vis']:10.2f} "
              f"{r['midpoint_analytical']:7.2f} {r['slope_vis']:7.3f} "
              f"{100 * r['frac_readable_vis']:5.0f}% {r['r2']['mu_vis']:9.3f}")

