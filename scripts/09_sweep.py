"""The sweep experiment -- one config key, several values, everything else fixed.

For every value it trains two networks on that value's own trials: the
causal network and a p_common = 1 control, whose residuals give that
variant's sigma_out. It then runs the implied-weight analysis on each
(cmsi.experiments.implied_weight) and compares the variants: the weight
against disparity and against the posterior, sigma_out and sigma_w, the
transition midpoint, the position-regression slope, the read-out accuracy
and the readability of the per-trial ratio, all against the swept value.
This is the systematic range testing of the design discussion, one knob at
a time; `06_implied_weight.py design` screens the same values without
training first.

    # the spread of the hand positions (the input domain), 128-unit layers
    python scripts/09_sweep.py train --param sigma0_sq --values 100 169 250 425 \\
                                     --set 'hidden=[128,128]' --name s0

    # one noise range alone; the others stay at the base config's
    python scripts/09_sweep.py train --param sigma2_prop_range --values '[2,2.5]' '[4,4.5]' \\
                                     '[8,9]' --config configs/exp2_128_mean.yaml --name propnoise

    # the same values, trained on trials thinned to a flat posterior histogram
    python scripts/09_sweep.py train --param sigma0_sq --values 100 425 --balance 0.5 --name s0_flat

    # anything utils.tweak can reach: rf_width, gain_K, hidden, epochs, lr ...
    python scripts/09_sweep.py train --param rf_width --values 4 6 9 --name rf --quick

    # redraw the comparison across every variant trained under <name>,
    # or across several experiments side by side
    python scripts/09_sweep.py compare --name s0
    python scripts/09_sweep.py compare --name s0 s0_flat

Everything goes to results/experiments/sweep/<name>/ and the datasets to
data/experiments/sweep/<name>_<value>_{causal,control}.npz; results/<run>/
and data/ are never touched, and nothing in the pipeline changes.

Layout of results/experiments/sweep/<name>/:

    config.yaml                 the base configuration (before the key is set)
    metrics.json                the comparison table (`compare`)
    figures/compare/            A weight vs disparity, B weight vs posterior,
                                C sigma_out and sigma_w, D summary,
                                E readability of the per-trial ratio
    variants/<param>=<value>/   model.pt, control.pt, config.yaml,
                                metrics.json, arrays.npz, figures/ (04, 05,
                                15, 16, sigma_*, reliability_*)

Values are parsed as yaml, so lists work; QUOTE them, zsh treats [ ] as a
glob. Every variant draws its own datasets because the key usually changes
the generative model; a variant's cached datasets are reused only if they
came from the same config (--regen redraws).

`--balance KEEP` keeps a fraction KEEP of the causal trials, chosen so that
the histogram of the analytical posterior is as flat as that allows
(cmsi.experiments.design.balance_posterior). The rule depends on the trials
only through the posterior -- a function of the measurements -- so the
targets stay exactly right and no cue to C is created; what changes is how
many ambiguous trials the network trains on. It is the safe reading of "a
balanced weight distribution"; the calibration gate is not run on these
variants, and `06_implied_weight.py design --balance` reports what the
anti-confound audit would say about the thinned trials.
"""

import argparse

import numpy as np
import yaml

import _bootstrap  # noqa: F401
from cmsi.experiments import implied_weight as iw
from cmsi.experiments import sweep
from cmsi.utils import load_config, load_json, save_config, save_json, tweak
from cmsi.utils.paths import CONFIGS, ROOT
from cmsi.viz import apply_style, save_figures

FORMATS = ("png", "svg")


def _merge_metrics(exp_dir, block, value):
    path = exp_dir / "metrics.json"
    m = load_json(path) if path.exists() else {}
    m[block] = value
    save_json(m, path)


def _print_reads(metrics):
    a = metrics["analytical"]
    print(f"analytical midpoint {a['midpoint_deg']:.2f} deg, sharpness {a['sharpness']:.3f}")
    for r in metrics["reads"].values():
        rb = r["readability"]
        print(f"  {r['output']}: sigma_out {r['sigma_out']:.4f}, own sd {rb['own_sd']:.4f}; "
              f"midpoint {r['midpoint_ls_deg']:.2f} (least squares) / "
              f"{r['midpoint_filtered_deg']:.2f} (filtered ratio) deg; "
              f"slope {r['position_regression']['slope']:.3f}; "
              f"{100 * r['frac_readable']:.0f}% readable; "
              f"{100 * rb['outside']:.1f}% of ratios outside [-1, 2]")


def _report(written, exp):
    for path in written:
        if path.suffix == ".png":
            print(f"  {path.relative_to(ROOT)}")
    print(f"-> {exp.relative_to(ROOT)}/")


def _levels(values):
    if len(values) == 1 and float(values[0]).is_integer() and float(values[0]) >= 2:
        return int(values[0])
    return [float(v) for v in values]


def cmd_train(args):
    cfg = load_config(args.config or (CONFIGS / "flagship.yaml"))
    overrides = iw.parse_overrides(args.set)
    if args.param in overrides:
        raise SystemExit(f"--set also sets {args.param}; it is the swept key")
    if args.quick:
        overrides.setdefault("n_trials", 8000)
        args.epochs = args.epochs or 60
    if args.n is not None:
        overrides["n_trials"] = int(args.n)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if overrides:
        print("config overrides:", overrides)
        cfg = tweak(cfg, **overrides)
    if args.balance is not None and not 0 < args.balance <= 1:
        raise SystemExit("--balance is a kept fraction in (0, 1]")

    values = [yaml.safe_load(v) for v in args.values]
    exp = sweep.experiment_dir(args.name)
    save_config(cfg, exp / "config.yaml")
    print(f"sweeping {args.param} over {values} ({len(values)} variant(s))"
          + (f", causal trials balanced to keep {args.balance:g}" if args.balance else ""))

    levels = _levels(args.levels)
    for order, value in enumerate(values):
        (metrics, arrays, ctrl_resid, names, pred, d_test, vcfg) = sweep.train_variant(
            args.param, value, cfg, exp, order, balance=args.balance, epochs=args.epochs,
            seed=args.seed, regen=args.regen)
        vdir = exp / "variants" / sweep.variant_label(args.param, value)
        sig_out = np.asarray(list(metrics["sigma_out"].values()))
        figs = {}
        figs.update(iw.baseline_figures(pred, d_test, names, sig_out, vcfg["analysis"], arrays,
                                        metrics))
        figs.update({f"sigma_{k}": v for k, v in
                     iw.sigma_figures(arrays, ctrl_resid, names, metrics).items()})
        rfigs, rtable = iw.reliability_figures(pred, d_test, names, sig_out,
                                               vcfg["analysis"], args.inputs, levels)
        figs.update({f"reliability_{k}": v for k, v in rfigs.items()})
        save_figures(figs, vdir / "figures", formats=FORMATS)
        m = load_json(vdir / "metrics.json")
        m["reliability"] = {"levels": levels, "table": rtable}
        save_json(m, vdir / "metrics.json")
        print(f"[{sweep.variant_label(args.param, value)}] done -> {vdir.relative_to(ROOT)}")
        _print_reads(metrics)
    cmd_compare(args)


def cmd_compare(args):
    names = [args.name] if isinstance(args.name, str) else list(args.name)
    exps = [sweep.experiment_dir(n, create=False) for n in names]
    rows = sweep.load_variants(exps)
    if not rows:
        raise SystemExit("no variants under " + ", ".join(str(e / "variants") for e in exps)
                         + " -- run `train` first")
    figs = sweep.variant_figures(rows)
    exp = exps[0] if len(exps) == 1 else sweep.experiment_dir("compare_" + "+".join(names))
    written = save_figures(figs, exp / "figures" / "compare", formats=FORMATS)
    _merge_metrics(exp, "compare", sweep.variants_table(rows))
    sweep.print_variants(rows)
    _report(written, exp)


if __name__ == "__main__":
    apply_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train", help="train one variant per value of a config key, each "
                                      "with its own control, and compare")
    sp.add_argument("--param", required=True, help="config key to sweep (any section)")
    sp.add_argument("--values", nargs="+", required=True,
                    help="values, parsed as yaml; quote values with brackets")
    sp.add_argument("--name", default="sweep", help="experiment folder name")
    sp.add_argument("--config", default=None, help="base config (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides applied to every variant; quote brackets")
    sp.add_argument("--balance", type=float, default=None, metavar="KEEP",
                    help="thin the causal trials to a flat posterior histogram, keeping "
                         "this fraction (the control is left alone)")
    sp.add_argument("--n", type=int, default=None, help="trials per dataset")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--quick", action="store_true", help="8000 trials, 60 epochs")
    sp.add_argument("--regen", action="store_true",
                    help="regenerate the variants' datasets")
    sp.add_argument("--inputs", nargs="+", default=["vis", "prop"],
                    choices=list(iw.INPUTS), help="reliability inputs for the per-variant figures")
    sp.add_argument("--levels", nargs="+", type=float, default=[4])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="redraw the comparison across trained variants")
    sp.add_argument("--name", nargs="+", default=["sweep"],
                    help="one experiment, or several to compare side by side")
    sp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)
