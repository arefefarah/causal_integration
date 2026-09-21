"""The fixed-variance experiment -- every input at one noise level.

Pins each input's measurement variance to a single value instead of a range
(sigma2_vis_range: [v, v], and likewise for prop and eye), trains the causal
network and a p_common = 1 control on those trials, and runs the
implied-weight analysis on the result. Give several values to sweep them;
every combination becomes one variant, and the variants are compared.

    # one variant: vis 3, prop 5.7, eye 8.25 (an input not given keeps the
    # midpoint of its range in the base config)
    python scripts/08_fixed_variance.py train --vis 3 --prop 5.7 --eye 8.25 --name mid

    # sweep the visual noise, the other two at their midpoints
    python scripts/08_fixed_variance.py train --vis 1.5 3 6 --name vis_sweep

    # a 3 x 2 grid over vis and prop
    python scripts/08_fixed_variance.py train --vis 1.5 3 6 --prop 2 8 --name grid

    # other base configs and overrides work as elsewhere
    python scripts/08_fixed_variance.py train --vis 3 --config configs/exp_weight.yaml \\
                                              --set rf_width=4 --name rf4_fixed
    python scripts/08_fixed_variance.py train --vis 1.5 6 --name vis_sweep --quick

    # redraw the comparison across every variant trained under <name>,
    # or across several experiments side by side
    python scripts/08_fixed_variance.py compare --name vis_sweep
    python scripts/08_fixed_variance.py compare --name vis_sweep grid

Everything goes to results/experiments/fixed_variance/<name>/ and the
datasets to data/experiments/fixed_variance/; results/<run>/ and data/ are
never touched. Nothing in the pipeline is changed either: a fixed variance
is a range whose ends coincide, which every stage already accepts.

Layout of results/experiments/fixed_variance/<name>/:

    config.yaml                 the base configuration (before pinning)
    metrics.json                the comparison table (`compare`)
    figures/compare/            A weight vs disparity, B weight vs posterior,
                                C sigma_out and sigma_w, D summary per variant
    variants/v3_p5.7_e8.25/     model.pt, control.pt, config.yaml (pinned),
                                metrics.json, arrays.npz, figures/ (04, 05,
                                15, 16, sigma_*)

Values are in deg^2, like the ranges in the config. Reliability is constant
within a variant by construction, so the per-variant reliability figures
are not drawn here; the pipeline's calibration gate would also fail its
reliability-coverage check on such a dataset, which is expected and is why
the gate is not run for these variants. To take one fixed-variance
configuration through the whole pipeline instead, use
06_implied_weight.py train with --set 'sigma2_vis_range=[3,3]' etc.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi.experiments import fixed_variance as fv
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

    exp = fv.experiment_dir(args.name)
    save_config(cfg, exp / "config.yaml")
    grid = fv.variant_grid(cfg, args.vis, args.prop, args.eye)
    mid = fv.range_midpoints(cfg)
    print(f"{len(grid)} variant(s); range midpoints of the base config: "
          + ", ".join(f"{k} {v:g}" for k, v in mid.items()))

    for sigma2 in grid:
        (metrics, arrays, ctrl_resid, names, pred, d_test, vcfg) = fv.train_variant(
            sigma2, cfg, exp, hidden=None, epochs=args.epochs, seed=args.seed,
            regen=args.regen)
        vdir = exp / "variants" / fv.variant_label(sigma2)
        sig_out = np.asarray(list(metrics["sigma_out"].values()))
        figs = {}
        figs.update(iw.baseline_figures(pred, d_test, names, sig_out, vcfg["analysis"], arrays,
                                        metrics))
        figs.update({f"sigma_{k}": v for k, v in
                     iw.sigma_figures(arrays, ctrl_resid, names, metrics).items()})
        save_figures(figs, vdir / "figures", formats=FORMATS)
        print(f"[{fv.variant_label(sigma2)}] done -> {vdir.relative_to(ROOT)}")
        _print_reads(metrics)
    cmd_compare(args)


def cmd_compare(args):
    names = [args.name] if isinstance(args.name, str) else list(args.name)
    exps = [fv.experiment_dir(n, create=False) for n in names]
    rows = fv.load_variants(exps)
    if not rows:
        raise SystemExit("no variants under " + ", ".join(str(e / "variants") for e in exps)
                         + " -- run `train` first")
    figs = fv.variant_figures(rows)
    exp = exps[0] if len(exps) == 1 else fv.experiment_dir("compare_" + "+".join(names))
    written = save_figures(figs, exp / "figures" / "compare", formats=FORMATS)
    _merge_metrics(exp, "compare", fv.variants_table(rows))
    fv.print_variants(rows)
    _report(written, exp)


if __name__ == "__main__":
    apply_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train", help="train variants at fixed variances, each with its "
                                      "own control, and compare")
    sp.add_argument("--vis", nargs="+", type=float, default=None,
                    help="fixed sigma2_vis value(s), deg^2 (default: range midpoint)")
    sp.add_argument("--prop", nargs="+", type=float, default=None,
                    help="fixed sigma2_prop value(s), deg^2 (default: range midpoint)")
    sp.add_argument("--eye", nargs="+", type=float, default=None,
                    help="fixed sigma2_eye value(s), deg^2 (default: range midpoint)")
    sp.add_argument("--name", default="fixed", help="experiment folder name")
    sp.add_argument("--config", default=None, help="base config (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides by key; quote values with brackets")
    sp.add_argument("--n", type=int, default=None, help="trials per dataset")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--quick", action="store_true", help="8000 trials, 60 epochs")
    sp.add_argument("--regen", action="store_true",
                    help="regenerate the variants' datasets")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="redraw the comparison across trained variants")
    sp.add_argument("--name", nargs="+", default=["fixed"],
                    help="one experiment, or several to compare side by side")
    sp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)
