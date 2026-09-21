"""The architecture experiment: the same task at other network sizes.

Trains one network per hidden-layer specification -- 16x16, 64x64, 32x128,
... -- and, for each, a p_common = 1 control at the same size, so every
variant has its own sigma_out. Then it runs the implied-weight analysis on
each and compares them: the weight against disparity and against the
posterior, sigma_out and sigma_w, the transition midpoint, the
position-regression slope and the read-out accuracy, all against the number
of hidden units.

Everything is written under results/experiments/architecture/<name>/ and the
datasets under data/experiments/architecture/, so results/<run>/ and data/
are never touched. When a size is chosen, it goes into configs/ and the
pipeline is re-run through the numbered stages.

Entry point: scripts/07_architecture.py (train, compare). The per-run
analysis and figures come from cmsi.experiments.implied_weight; this module
adds the variant plumbing and the cross-variant figures.
"""

import numpy as np
from matplotlib import pyplot as plt

from cmsi.analysis.causal import mean_by_bin
from cmsi.experiments.implied_weight import (
    READS,
    _on_grid,
    _ramp,
    weight_analysis,
)
from cmsi.viz.style import COLORS, SIZE, label_panels

EXPERIMENT = "architecture"


# --------------------------------------------------------------------------- #
# where things go
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/architecture/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def experiment_data_dir():
    """data/experiments/architecture/ -- the datasets the variants train on."""
    from cmsi.utils.paths import DATA

    path = DATA / "experiments" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# variants: train and evaluate
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


def train_variant(hidden, data, exp_dir, epochs=None, seed=None, verbose=True,
                  label=None, extra=None):
    """Train the causal network AND its control on `data` (the causal and
    control datasets of one configuration), evaluate both on their test
    splits, and write everything under <exp>/variants/<label>/.

    `hidden` sets the architecture of both networks; None keeps the
    configuration's own. `label` names the variant (default: the hidden
    size, hAxB) and `extra` is merged into its metrics -- this is how the
    other experiments (fixed_variance) reuse the same plumbing with a
    variant that differs in something other than its size.

    Returns (metrics, arrays, control residuals, output names, causal
    predictions, causal test split, causal config)."""
    from cmsi.data import subset
    from cmsi.models import predict, train
    from cmsi.utils import save_checkpoint, save_config, save_json, tweak

    hidden = list(hidden) if hidden is not None else list(data["causal"][1]["model"]["hidden"])
    name = label or variant_name(hidden)
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
                   sigma_out_source="control trained on the same configuration")
    metrics.update(extra or {})
    save_json(metrics, vdir / "metrics.json")
    np.savez_compressed(vdir / "arrays.npz",
                        control_resid=ctrl_resid,
                        **{f"{k}_{kk}": v for k in ("vis", "prop")
                           for kk, v in arrays[k].items()
                           if isinstance(v, np.ndarray)},
                        disparity=arrays["disparity"], post_c1=arrays["post_c1"])
    return metrics, arrays, ctrl_resid, names, preds["causal"], tests["causal"], cfgs["causal"]


def load_variants(exp_dirs, label_by="hidden"):
    """Every variants/<name>/metrics.json under one experiment, or under
    several (a list of experiment dirs), in order.

    `label_by` is "hidden" for the architecture experiment (rows labelled
    AxB, ordered by units) or "variant" for experiments whose variants
    differ in something else (rows labelled by their variant name, ordered
    by the `order` key their metrics carry, if any).

    With several experiments the rows are labelled "<experiment>/<label>",
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
            if label_by == "hidden" and len(h) == 2:
                short = f"{h[0]}x{h[1]}"
            else:
                short = m.get("variant", "?")
            m["label"] = f"{exp_dir.name}/{short}" if len(exp_dirs) > 1 else short
            rows.append(m)
    rows.sort(key=lambda m: (m.get("experiment", ""),
                             tuple(m["order"]) if m.get("order") else (m.get("n_units", 0),),
                             m.get("variant", "")))
    return rows


def _variant_label(m):
    if "label" in m:
        return m["label"]
    h = m.get("hidden", [])
    return f"{h[0]}x{h[1]}" if len(h) == 2 else m.get("variant", "?")


def fig_variants_weight_curves(rows, read="vis", by="network size", legend="hidden (SIL x MSL)"):
    """Least-squares weight against disparity, one curve per variant, with
    the analytical posterior dashed. Variants that share a dataset share one
    black analytical curve; when the rows come from several experiments
    (different configs, so different analytical curves) each variant's own
    analytical curve is dashed in that variant's colour."""
    colors = _ramp(len(rows), "plasma")
    multi = len({m.get("experiment") for m in rows}) > 1
    # several datasets among the rows (several experiments, or one experiment
    # whose variants differ in their generative model, as in fixed_variance):
    # each analytical curve then takes its variant's colour
    own_curve = multi or len({round(m["analytical"]["midpoint_deg"] or 0, 2)
                              for m in rows}) > 1
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
            if "_analytical_curve" in m:          # rows built in memory (06 compare)
                c, mm = m["_analytical_curve"]
            else:
                a = np.load(m["_dir"] / "arrays.npz")
                c, mm, _ = mean_by_bin(a["disparity"], a["post_c1"],
                                       np.asarray(m["grid"], float))
            tag = f"(mid {mid_a:.1f})" if mid_a is not None else ""
            ax.plot(c, mm, "--", color=color if own_curve else COLORS["analytical"], lw=1.1,
                    label=(f"analytical, {_variant_label(m)} {tag}" if own_curve
                           else f"analytical {tag}"))
            drawn.append(key)
    ax.set(xlabel="body-frame disparity (deg)", ylabel="weight on fused estimate",
           ylim=(-0.1, 1.1),
           title=f"implied weight by {'configuration' if multi else by}, "
                 f"read from mu_{read}")
    ax.legend(fontsize=7, frameon=False,
              title="experiment/variant" if multi else legend, title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_variants_weight_vs_posterior(rows, read="vis", by="network size",
                                     legend="hidden (SIL x MSL)"):
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
           xlim=(0, 1), ylim=(-0.05, 1.05), title=f"weight vs posterior by {by}, mu_{read}")
    ax.legend(fontsize=7.5, frameon=False, title=legend, title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_variants_sigma(rows, by="network size"):
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
    ax.set(xticks=x, xticklabels=labels, ylabel="deg", title=f"read-out noise by {by}")
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



def fig_variants_summary(rows, xlabel="hidden units, SIL x MSL (total)", units=True):
    """Four summaries per variant, ordered by total hidden units: transition
    midpoint (both estimators, both reads) against the analytical value; the
    position-regression slope; the read-out R^2 per output; and the fraction
    of trials the sigma_w criterion keeps."""
    x = np.arange(len(rows), dtype=float)
    labels = [f"{_variant_label(m)}\n({m['n_units']} units)" if units else _variant_label(m)
              for m in rows]
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
        ax.set_xlabel(xlabel)
    label_panels(axes)
    fig.tight_layout()
    return fig


def fig_variants_readability(rows, by="network size"):
    """Why the per-trial ratio spikes on each variant, from the readability
    block of weight_analysis. Left: the fraction of ratios outside [-1, 2]
    observed, beside what a Gaussian error of the variant's own residual sd
    would give and what its sigma_out alone would give (an optimal network
    with that read-out noise). Right: the own residual sd against sigma_out
    (how far from optimal the variant is) and the share of trials with
    |Delta| < 1 deg (how many trials no network can be read on)."""
    labels = [_variant_label(m) for m in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    ax, wdt = axes[0], 0.13
    for j, (key, out_name, _) in enumerate(READS):
        col = COLORS["visual"] if key == "vis" else COLORS["prop"]
        rb = [m["reads"][key]["readability"] for m in rows]
        fields = (("outside", "observed", 1.0),
                  ("outside_predicted_own_sd", "Gaussian, own sd", 0.55),
                  ("outside_predicted_sigma_out", "Gaussian, sigma_out", 0.3))
        for k, (field, lab, alpha) in enumerate(fields):
            ax.bar(x + (j - 0.5) * 3 * wdt + (k - 1) * wdt, [100 * r[field] for r in rb], wdt,
                   color=col, alpha=alpha, label=f"{lab}, {out_name}")
    ax.set(xticks=x, xticklabels=labels, ylabel="ratios outside [-1, 2] (%)",
           title=f"per-trial ratio by {by}")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    for key, out_name, _ in READS:
        col = COLORS["visual"] if key == "vis" else COLORS["prop"]
        rb = [m["reads"][key]["readability"] for m in rows]
        ax.plot(x, [r["own_sd_over_sigma_out"] for r in rb], "o-", color=col, ms=4,
                label=f"own sd / sigma_out, {out_name}")
    ax.axhline(1, color=COLORS["analytical"], ls="--", lw=1.1, label="optimal (own sd = sigma_out)")
    ax.set(xticks=x, xticklabels=labels, ylabel="own residual sd / sigma_out",
           title="distance from optimal, and unreadable trials")
    ax.tick_params(axis="x", labelsize=7.5)
    ax2 = ax.twinx()
    for key, _, _ in READS:
        col = COLORS["visual"] if key == "vis" else COLORS["prop"]
        rb = [m["reads"][key]["readability"] for m in rows]
        ax2.plot(x, [100 * r["frac_absdelta_lt1"] for r in rb], ":", color=col, lw=1.3)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("trials with |Delta| < 1 deg (%, dotted)", color="0.35", fontsize=9)
    ax2.tick_params(axis="y", colors="0.35", labelsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    label_panels(axes)
    fig.tight_layout()
    return fig


def variant_figures(rows, by="network size", legend="hidden (SIL x MSL)",
                    xlabel="hidden units, SIL x MSL (total)", units=True):
    """The comparison figures. The wording arguments let another experiment
    (fixed_variance, sweep) draw the same figures with its own axis."""
    if not rows:
        return {}
    figs = {
        "A_weight_vs_disparity_by_variant": fig_variants_weight_curves(rows, "vis", by, legend),
        "B_weight_vs_posterior_by_variant": fig_variants_weight_vs_posterior(rows, "vis", by,
                                                                             legend),
        "C_sigma_by_variant": fig_variants_sigma(rows, by),
    }
    if len(rows) > 1:
        figs["D_summary_by_variant" if not units else "D_summary_vs_units"] = \
            fig_variants_summary(rows, xlabel, units)
    # variants analysed before the readability block existed have no entry;
    # the figure is drawn when every row has one
    if all("readability" in m["reads"]["vis"] for m in rows):
        figs["E_readability_by_variant"] = fig_variants_readability(rows, by)
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

