"""The implied-weight investigation -- an experiment, not a pipeline stage.

Everything goes to results/experiments/implied_weight/<name>/ and nothing
under results/<run>/ is touched, so architectures and input encodings can be
tried freely. When one is chosen, put it in configs/ and re-run the pipeline.

    # the three per-run figures this grew out of (05, 15, 16), for one run
    python scripts/06_implied_weight.py figures     --run flagship --control pcommon1

    # 1. what sigma_out and sigma_w are, and what the criterion keeps
    python scripts/06_implied_weight.py sigma       --run flagship --control pcommon1

    # 2. the weight at many reliability levels of each input, read from each output
    python scripts/06_implied_weight.py reliability --run flagship --control pcommon1 \\
                                                    --inputs vis prop --levels 5
    python scripts/06_implied_weight.py reliability --run flagship --levels 1.5 2.5 3.5 4.5 5.5 6.5

    # 3. other networks: train each hidden size with its own p_common=1 control
    python scripts/06_implied_weight.py train --hidden 16 32 64 128 256 --name units
    python scripts/06_implied_weight.py train --hidden 32x128 64x32   --name asym
    python scripts/06_implied_weight.py train --hidden 64 --set rf_width=4 --name rf4
    python scripts/06_implied_weight.py train --hidden 16 256 --name units --quick

    # a different configuration, at its own architecture, with its own control
    python scripts/06_implied_weight.py train --config configs/realistic.yaml --name realistic
    python scripts/06_implied_weight.py train --set sigma2_vis_range=[1,4] --name lownoise

    # redraw the comparison across every variant already trained under <name>,
    # or across several experiments (different configs) side by side
    python scripts/06_implied_weight.py compare --name units
    python scripts/06_implied_weight.py compare --name units realistic lownoise

Layout of results/experiments/implied_weight/<name>/:

    metrics.json                one block per sub-command that has been run
    figures/figures/            05_fusion_weight_by_reliability, 15, 16, 04
    figures/sigma/              01-04, the sigma_out / sigma_w figures
    figures/reliability/        one pair of figures per input
    figures/compare/            A-D, every variant under this experiment
    variants/<hAxB>/            model.pt, control.pt, config.yaml,
                                metrics.json, arrays.npz, figures/

`--set key=value` (repeatable) changes any config key by name, in whichever
section owns it (utils.tweak): --set rf_width=4 --set n_vis=100 --set
sigma2_vis_range=[1,4]. The datasets a `train` experiment uses are generated
once from that config under data/experiments/implied_weight/ and reused by
every variant of the same experiment; --regen forces a fresh draw.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi.experiments import implied_weight as iw
from cmsi.utils import load_config, load_json, save_json, tweak
from cmsi.viz import apply_style, save_figures

FORMATS = ("png", "svg")


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def _merge_metrics(exp_dir, block, value):
    """metrics.json holds one block per sub-command; keep the others."""
    path = exp_dir / "metrics.json"
    m = load_json(path) if path.exists() else {}
    m[block] = value
    save_json(m, path)


def _run_context(args):
    """Load the run and its control, run the shared analysis once."""
    print(f"loading results/{args.run} (control: {args.control})")
    pred, d, names, cfg = iw.load_run(args.run)
    ctrl_resid, ctrl_names = iw.control_residuals(args.control)
    if ctrl_names != names:
        raise SystemExit(f"control '{args.control}' has outputs {ctrl_names}, "
                         f"run '{args.run}' has {names} -- not comparable")
    sig_out = ctrl_resid.std(axis=0)
    print("sigma_out from the control: "
          + ", ".join(f"{n} {s:.4f}" for n, s in zip(names, sig_out, strict=False)))
    metrics, arrays = iw.weight_analysis(pred, d, names, sig_out, cfg["analysis"])
    metrics["run"], metrics["control"] = args.run, args.control
    return pred, d, names, cfg, sig_out, ctrl_resid, metrics, arrays


def _print_reads(metrics):
    a = metrics["analytical"]
    print(f"analytical midpoint {a['midpoint_deg']:.2f} deg, sharpness {a['sharpness']:.3f}")
    for r in metrics["reads"].values():
        print(f"  {r['output']}: sigma_out {r['sigma_out']:.4f}  ->  |Delta| must exceed "
              f"{r['delta_threshold_deg']:.2f} deg; {100 * r['frac_readable']:.1f}% of trials "
              f"readable (median sigma_w {r['sigma_w_median']:.3f})")
        print(f"          midpoint {r['midpoint_ls_deg']:.2f} (least squares) / "
              f"{r['midpoint_filtered_deg']:.2f} (filtered ratio) deg; "
              f"position-regression slope {r['position_regression']['slope']:.3f}")


# --------------------------------------------------------------------------- #
# sub-commands on an existing run
# --------------------------------------------------------------------------- #
def cmd_figures(args):
    pred, d, names, cfg, sig_out, _, metrics, arrays = _run_context(args)
    exp = iw.experiment_dir(args.name or args.run)
    figs = iw.baseline_figures(pred, d, names, sig_out, cfg["analysis"], arrays)
    written = save_figures(figs, exp / "figures" / "figures", formats=FORMATS)
    _merge_metrics(exp, "figures", {k: metrics[k] for k in
                                    ("run", "control", "sigma_out", "residual_std_self",
                                     "analytical", "reads")})
    _print_reads(metrics)
    _report(written, exp)


def cmd_sigma(args):
    pred, d, names, cfg, sig_out, ctrl_resid, metrics, arrays = _run_context(args)
    exp = iw.experiment_dir(args.name or args.run)
    figs = iw.sigma_figures(arrays, ctrl_resid, names, metrics)
    written = save_figures(figs, exp / "figures" / "sigma", formats=FORMATS)
    block = {k: metrics[k] for k in ("run", "control", "sigma_w_criterion",
                                     "sigma_out", "residual_std_self")}
    block["reads"] = {k: {kk: v for kk, v in r.items() if kk not in ("curve", "by_posterior")}
                      for k, r in metrics["reads"].items()}
    _merge_metrics(exp, "sigma", block)
    _print_reads(metrics)
    print("\nunfiltered per-trial weight w = (network - seg)/Delta, every trial with Delta != 0:")
    for r in metrics["reads"].values():
        s = r["w_raw"]
        print(f"  {r['output']}: n {s['n']}, median {s['median']:.3f}, IQR "
              f"[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}], mean {s['mean']:.2f}, sd {s['std']:.2f}, "
              f"{100 * s['frac_outside_-1_2']:.1f}% outside [-1, 2]")
    print("\nsigma_out is one number per output (the control's residual std); "
          "sigma_w = sigma_out/|Delta| is the per-trial quantity. No figure here is filtered.")
    _report(written, exp)


def _levels(values):
    if len(values) == 1 and float(values[0]).is_integer() and float(values[0]) >= 2:
        return int(values[0])
    return [float(v) for v in values]


def cmd_reliability(args):
    pred, d, names, cfg, sig_out, _, metrics, _ = _run_context(args)
    exp = iw.experiment_dir(args.name or args.run)
    levels = _levels(args.levels)
    how = (f"{levels} quantile bins" if isinstance(levels, int)
           else f"nearest of {levels}")
    print(f"reliability levels: {how}; inputs: {', '.join(args.inputs)}")
    figs, table = iw.reliability_figures(pred, d, names, sig_out, cfg["analysis"],
                                         args.inputs, levels)
    written = save_figures(figs, exp / "figures" / "reliability", formats=FORMATS)
    _merge_metrics(exp, "reliability", {"run": args.run, "control": args.control,
                                        "levels": levels, "table": table})
    for var, rows in table.items():
        print(f"\n{var}:  level  n  midpoint analytical / from mu_vis / from mu_prop")
        for label, r in rows.items():
            print(f"  {label:>10} {r['n']:6d}  {r['midpoint_analytical']:6.2f} / "
                  f"{r['midpoint_vis']:6.2f} / {r['midpoint_prop']:6.2f}")
    _report(written, exp)


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def cmd_train(args):
    cfg = load_config(args.config or _default_config())
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

    exp = iw.experiment_dir(args.name)
    if args.regen and (exp / "variants").exists() and any((exp / "variants").iterdir()):
        print(f"warning: --regen redraws the datasets of '{args.name}', but variants "
              f"already trained under it keep the old draw -- the comparison will "
              f"mix two datasets. A new --name keeps experiments separate.")
    data = iw.variant_datasets(cfg, exp, regen=args.regen)
    from cmsi.utils import save_config
    save_config(cfg, exp / "config.yaml")

    levels = _levels(args.levels)
    # no --hidden: one variant at the config's own architecture, which is how
    # a different config (another encoding, another generative model) gets
    # the whole sigma / reliability treatment with its own control
    specs = args.hidden or ["x".join(str(h) for h in cfg["model"]["hidden"])]
    for spec in specs:
        hidden = iw.parse_hidden(spec)
        (metrics, arrays, ctrl_resid, names, pred, d_test, vcfg) = iw.train_variant(
            hidden, data, exp, epochs=args.epochs, seed=args.seed)
        vdir = exp / "variants" / iw.variant_name(hidden)
        figs = {}
        figs.update(iw.baseline_figures(pred, d_test, names,
                                        np.asarray(list(metrics["sigma_out"].values())),
                                        vcfg["analysis"], arrays))
        figs.update({f"sigma_{k}": v for k, v in
                     iw.sigma_figures(arrays, ctrl_resid, names, metrics).items()})
        rfigs, rtable = iw.reliability_figures(
            pred, d_test, names, np.asarray(list(metrics["sigma_out"].values())),
            vcfg["analysis"], args.inputs, levels)
        figs.update({f"reliability_{k}": v for k, v in rfigs.items()})
        save_figures(figs, vdir / "figures", formats=FORMATS)
        m = load_json(vdir / "metrics.json")
        m["reliability"] = {"levels": levels, "table": rtable}
        save_json(m, vdir / "metrics.json")
        print(f"[{iw.variant_name(hidden)}] done -> {vdir.relative_to(exp.parents[2])}")
        _print_reads(metrics)
    cmd_compare(args)


def cmd_compare(args):
    names = [args.name] if isinstance(args.name, str) else list(args.name)
    exps = [iw.experiment_dir(n, create=False) for n in names]
    rows = iw.load_variants(exps)
    if not rows:
        raise SystemExit("no variants under " + ", ".join(str(e / "variants") for e in exps)
                         + " -- run `train` first")
    figs = iw.variant_figures(rows)
    # one experiment: its own compare/ folder. Several: a folder named after
    # all of them, so nothing inside any one experiment is overwritten.
    exp = exps[0] if len(exps) == 1 else iw.experiment_dir("compare_" + "+".join(names))
    written = save_figures(figs, exp / "figures" / "compare", formats=FORMATS)
    _merge_metrics(exp, "compare", iw.variants_table(rows))
    iw.print_variants(rows)
    _report(written, exp)


def _default_config():
    from cmsi.utils.paths import CONFIGS
    return CONFIGS / "flagship.yaml"


def _report(written, exp):
    from cmsi.utils.paths import ROOT
    for path in written:
        if path.suffix == ".png":
            print(f"  {path.relative_to(ROOT)}")
    print(f"-> {exp.relative_to(ROOT)}/")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    apply_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def run_args(sp):
        sp.add_argument("--run", default="flagship", help="results/<run>/ to analyse")
        sp.add_argument("--control", default="pcommon1",
                        help="p_common=1 run whose residuals give sigma_out")
        sp.add_argument("--name", default=None,
                        help="experiment folder name (default: the run name)")

    sp = sub.add_parser("figures", help="05, 15, 16 (and 04) for one run")
    run_args(sp)
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("sigma", help="sigma_out and per-trial sigma_w on one run")
    run_args(sp)
    sp.set_defaults(func=cmd_sigma)

    sp = sub.add_parser("reliability", help="weight vs disparity per reliability level")
    run_args(sp)
    sp.add_argument("--inputs", nargs="+", default=["vis", "prop"],
                    choices=list(iw.INPUTS), help="which input's noise to split by")
    sp.add_argument("--levels", nargs="+", type=float, default=[5],
                    help="one integer = that many quantile bins; several numbers = "
                         "level centres (nearest match), e.g. 1.5 3.5 6")
    sp.set_defaults(func=cmd_reliability)

    sp = sub.add_parser("train", help="train variants, each with its own control")
    sp.add_argument("--hidden", nargs="+", default=None,
                    help="hidden sizes: 32 (both layers) or 32x128 (SIL x MSL); "
                         "omitted = the config's own architecture, one variant")
    sp.add_argument("--name", default="variants", help="experiment folder name")
    sp.add_argument("--config", default=None, help="base config (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides by key, e.g. rf_width=4 n_vis=100")
    sp.add_argument("--n", type=int, default=None, help="trials per dataset")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--quick", action="store_true", help="8000 trials, 60 epochs")
    sp.add_argument("--regen", action="store_true",
                    help="regenerate this experiment's datasets")
    sp.add_argument("--inputs", nargs="+", default=["vis", "prop"],
                    choices=list(iw.INPUTS))
    sp.add_argument("--levels", nargs="+", type=float, default=[4])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="redraw the comparison across trained variants")
    sp.add_argument("--name", nargs="+", default=["variants"],
                    help="one experiment, or several to compare across configs")
    sp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)
