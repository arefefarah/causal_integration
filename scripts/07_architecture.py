"""The architecture experiment -- the same task at other network sizes.

For every hidden-layer specification it trains two networks on the same
trials: the causal network and a p_common = 1 control of the same size,
whose residuals give that size's own sigma_out. It then runs the
implied-weight analysis on each (cmsi.experiments.implied_weight) and
compares the variants: weight against disparity and against the posterior,
sigma_out and sigma_w, the transition midpoint, the position-regression
slope and the read-out accuracy, all against the number of hidden units.

    python scripts/07_architecture.py train --hidden 16 32 64 128 256 --name units
    python scripts/07_architecture.py train --hidden 32x128 128x32   --name asym
    python scripts/07_architecture.py train --hidden 16 256 --name units --quick
    python scripts/07_architecture.py train --hidden 32 64 --config configs/exp_weight.yaml \\
                                            --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise_units

    # redraw the comparison across every variant trained under <name>,
    # or across several experiments side by side
    python scripts/07_architecture.py compare --name units
    python scripts/07_architecture.py compare --name units lownoise_units

Everything goes to results/experiments/architecture/<name>/ and the datasets
to data/experiments/architecture/<name>_{causal,control}.npz; results/<run>/
and data/ are never touched. A hidden size is `N` (both layers) or `AxB`
(SIL x MSL).

Layout of results/experiments/architecture/<name>/:

    config.yaml                 the configuration every variant was trained on
    metrics.json                the comparison table (`compare`)
    figures/compare/            A weight vs disparity, B weight vs posterior,
                                C sigma_out and sigma_w, D summary vs units
    variants/<hAxB>/            model.pt, control.pt, config.yaml,
                                metrics.json, arrays.npz, figures/ (04, 05,
                                15, 16, sigma_*, reliability_*)

The datasets of an experiment are drawn once from its config and reused by
every later variant of the same experiment, so variants stay comparable; a
different config needs a different --name (or --regen, which redraws and
prints a warning if variants already exist). `--set key=value` changes any
config key by name; quote values with brackets, zsh treats [ ] as a glob.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi.experiments import architecture as arch
from cmsi.experiments import implied_weight as iw
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
        print(f"  {r['output']}: sigma_out {r['sigma_out']:.4f}; "
              f"midpoint {r['midpoint_ls_deg']:.2f} (least squares) / "
              f"{r['midpoint_filtered_deg']:.2f} (filtered ratio) deg; "
              f"slope {r['position_regression']['slope']:.3f}; "
              f"{100 * r['frac_readable']:.0f}% readable")


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

    exp = arch.experiment_dir(args.name)
    if args.regen and (exp / "variants").exists() and any((exp / "variants").iterdir()):
        print(f"warning: --regen redraws the datasets of '{args.name}', but variants "
              f"already trained under it keep the old draw -- the comparison will "
              f"mix two datasets. A new --name keeps experiments separate.")
    data = arch.variant_datasets(cfg, exp, regen=args.regen)
    save_config(cfg, exp / "config.yaml")

    levels = _levels(args.levels)
    for spec in args.hidden:
        hidden = arch.parse_hidden(spec)
        (metrics, arrays, ctrl_resid, names, pred, d_test, vcfg) = arch.train_variant(
            hidden, data, exp, epochs=args.epochs, seed=args.seed)
        vdir = exp / "variants" / arch.variant_name(hidden)
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
        print(f"[{arch.variant_name(hidden)}] done -> {vdir.relative_to(ROOT)}")
        _print_reads(metrics)
    cmd_compare(args)


def cmd_compare(args):
    names = [args.name] if isinstance(args.name, str) else list(args.name)
    exps = [arch.experiment_dir(n, create=False) for n in names]
    rows = arch.load_variants(exps)
    if not rows:
        raise SystemExit("no variants under " + ", ".join(str(e / "variants") for e in exps)
                         + " -- run `train` first")
    figs = arch.variant_figures(rows)
    # one experiment: its own compare/ folder. Several: a folder named after
    # all of them, so nothing inside any one experiment is overwritten.
    exp = exps[0] if len(exps) == 1 else arch.experiment_dir("compare_" + "+".join(names))
    written = save_figures(figs, exp / "figures" / "compare", formats=FORMATS)
    _merge_metrics(exp, "compare", arch.variants_table(rows))
    arch.print_variants(rows)
    _report(written, exp)


if __name__ == "__main__":
    apply_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train", help="train variants, each with its own control, and compare")
    sp.add_argument("--hidden", nargs="+", required=True,
                    help="hidden sizes: 32 (both layers) or 32x128 (SIL x MSL)")
    sp.add_argument("--name", default="units", help="experiment folder name")
    sp.add_argument("--config", default=None, help="base config (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides by key; quote values with brackets")
    sp.add_argument("--n", type=int, default=None, help="trials per dataset")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--quick", action="store_true", help="8000 trials, 60 epochs")
    sp.add_argument("--regen", action="store_true",
                    help="regenerate this experiment's datasets")
    sp.add_argument("--inputs", nargs="+", default=["vis", "prop"],
                    choices=list(iw.INPUTS), help="reliability inputs for the per-variant figures")
    sp.add_argument("--levels", nargs="+", type=float, default=[4])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="redraw the comparison across trained variants")
    sp.add_argument("--name", nargs="+", default=["units"],
                    help="one experiment, or several to compare side by side")
    sp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)
