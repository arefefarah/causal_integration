"""The implied-weight investigation -- an experiment, not a pipeline stage.

The analyses (sigma, reliability, figures) read an ordinary pipeline run in
results/<run>/ and write only to results/experiments/implied_weight/<name>/,
so nothing in results/<run>/ is ever changed by them.

    # the three per-run figures this grew out of (05, 15, 16), for one run
    python scripts/06_implied_weight.py figures     --run flagship

    # 1. what sigma_out and sigma_w are, and what the criterion would keep;
    #    the weight on every trial read from the position outputs, from the
    #    variance outputs (sharp where Delta is small), and from all four
    python scripts/06_implied_weight.py sigma       --run flagship

    # 2. the weight at many reliability levels of each input, read from each output
    python scripts/06_implied_weight.py reliability --run flagship --inputs vis prop --levels 5
    python scripts/06_implied_weight.py reliability --run flagship --levels 1.5 2.5 3.5 4.5 5.5 6.5

`--control` names the p_common = 1 run whose residuals give sigma_out. Left
out, it is read from the run's own metrics.json (stage 3 records the control
it used as sigma_out_source), falling back to pcommon1.

    # 3. a NEW configuration: run the whole pipeline on it, exactly as
    #    run_all.sh does, into results/<name>/ -- plus a p_common = 1 control
    #    of the same configuration in results/<name>_pcommon1/
    python scripts/06_implied_weight.py train --config configs/exp_weight.yaml --name lownoise
    python scripts/06_implied_weight.py train --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise
    python scripts/06_implied_weight.py train --config configs/realistic.yaml --name real --quick

    # ... after which every analysis in the project applies to it:
    python scripts/06_implied_weight.py sigma --run lownoise
    python scripts/04_figures.py --run lownoise --only manuscript

`train` writes the exact configs it used to configs/experiments/<name>.yaml
and <name>_pcommon1.yaml, so a configuration you decide to keep is one
`make all CONFIG=configs/experiments/<name>.yaml` away. The calibration gate
(stage 0) is run and reported but does not stop an experiment; read its
verdict in results/calibration/<name>/.

`--set key=value` (repeatable) changes any config key by name, in whichever
section owns it (utils.tweak). QUOTE values that contain brackets, because
zsh treats [ ] as a glob:  --set 'sigma2_vis_range=[1,4]'  --set rf_width=4

    # 4. several trained runs side by side: what their trials allowed the
    #    per-trial ratio to show, and what each network achieved
    python scripts/06_implied_weight.py compare --runs flagship exp2 exp2_128_mean

    # 5. a configuration BEFORE training: the posterior distribution, |Delta|,
    #    the fraction of ratios that would fall outside [-1, 2] for any
    #    read-out error, the floor on that error set by the spike code
    python scripts/06_implied_weight.py design --config configs/flagship.yaml
    python scripts/06_implied_weight.py design --param sigma0_sq --values 100 169 425 \\
                                               --set 'hidden=[128,128]' --name s0_design
    python scripts/06_implied_weight.py design --config configs/exp2.yaml \\
                                               --param sigma2_prop_range --values '[2,2.5]' '[8,9]'
    python scripts/06_implied_weight.py design --config a.yaml b.yaml --mark flagship exp2_128_mean

`compare` reads results/<run>/ for each run (control resolved as above) and
writes to results/experiments/implied_weight/compare_<runs>/. `design`
trains nothing; `--mark RUN` draws the error a trained run achieved on the
readability curves, `--balance KEEP` thins the trials to a flat posterior
histogram first (see scripts/09_sweep.py --balance).

Other network sizes are a separate experiment: scripts/07_architecture.py;
one config key over several values: scripts/09_sweep.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from cmsi.experiments import implied_weight as iw
from cmsi.utils import load_config, load_json, save_config, save_json, tweak
from cmsi.utils.paths import CONFIGS, RESULTS, ROOT
from cmsi.viz import apply_style, save_figures

FORMATS = ("png", "svg")
SCRIPTS = ROOT / "scripts"


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def _merge_metrics(exp_dir, block, value):
    """metrics.json holds one block per sub-command; keep the others."""
    path = exp_dir / "metrics.json"
    m = load_json(path) if path.exists() else {}
    m[block] = value
    save_json(m, path)


def _resolve_control(run, control):
    """The p_common = 1 run that supplies sigma_out for `run`."""
    if control:
        return control
    path = RESULTS / run / "metrics.json"
    if path.exists():
        src = load_json(path).get("sigma_out_source")
        if src and src != "self" and (RESULTS / src / "model.pt").exists():
            print(f"control: {src} (recorded by stage 3 in results/{run}/metrics.json)")
            return src
    if (RESULTS / "pcommon1" / "model.pt").exists():
        print("control: pcommon1 (default; pass --control to use another)")
        return "pcommon1"
    raise SystemExit(f"no p_common=1 control found for '{run}': pass --control <run>")


def _run_context(args):
    """Load the run and its control, run the shared analysis once."""
    control = _resolve_control(args.run, args.control)
    print(f"loading results/{args.run} (control: {control})")
    pred, d, names, cfg = iw.load_run(args.run)
    ctrl_resid, ctrl_names = iw.control_residuals(control)
    if ctrl_names != names:
        raise SystemExit(f"control '{control}' has outputs {ctrl_names}, "
                         f"run '{args.run}' has {names} -- not comparable")
    sig_out = ctrl_resid.std(axis=0)
    print("sigma_out from the control: "
          + ", ".join(f"{n} {s:.4f}" for n, s in zip(names, sig_out, strict=False)))
    metrics, arrays = iw.weight_analysis(pred, d, names, sig_out, cfg["analysis"])
    metrics["run"], metrics["control"] = args.run, control
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


def _report(written, exp):
    for path in written:
        if path.suffix == ".png":
            print(f"  {path.relative_to(ROOT)}")
    print(f"-> {exp.relative_to(ROOT)}/")


# --------------------------------------------------------------------------- #
# analyses of an existing run
# --------------------------------------------------------------------------- #
def cmd_figures(args):
    pred, d, names, cfg, sig_out, _, metrics, arrays = _run_context(args)
    exp = iw.experiment_dir(args.name or args.run)
    figs = iw.baseline_figures(pred, d, names, sig_out, cfg["analysis"], arrays, metrics)
    written = save_figures(figs, exp / "figures" / "figures", formats=FORMATS)
    iw.print_per_trial(metrics)
    pt = metrics.get("per_trial")
    if pt:
        for key in ("vis", "prop"):
            r = pt["variance_regression"][key]
            print(f"  variance-domain regression, var_{key}: slope {r['slope']:.3f} "
                  f"[{r['slope_ci95'][0]:.3f}, {r['slope_ci95'][1]:.3f}], "
                  f"intercept {r['intercept']:+.3f}")
        r = pt["weight_regression"]["variance"]
        print(f"  weight read from the variance outputs on the posterior: slope {r['slope']:.3f} "
              f"[{r['slope_ci95'][0]:.3f}, {r['slope_ci95'][1]:.3f}], "
              f"intercept {r['intercept']:+.3f}")
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
                                     "sigma_out", "residual_std_self", "per_trial")}
    block["reads"] = {k: {kk: v for kk, v in r.items() if kk not in ("curve", "by_posterior")}
                      for k, r in metrics["reads"].items()}
    _merge_metrics(exp, "sigma", block)
    _print_reads(metrics)
    iw.print_per_trial(metrics)
    print("\nunfiltered per-trial weight w = (network - seg)/Delta, every trial with Delta != 0:")
    for r in metrics["reads"].values():
        s = r["w_raw"]
        print(f"  {r['output']}: n {s['n']}, median {s['median']:.3f}, IQR "
              f"[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}], mean {s['mean']:.2f}, sd {s['std']:.2f}, "
              f"{100 * s['frac_outside_-1_2']:.1f}% outside [-1, 2]")
    table = iw.weight_by_delta_table(arrays, arrays["post_c1"])
    block["weight_by_delta"] = table
    _merge_metrics(exp, "sigma", block)
    print("\nthe same ratio grouped by |Delta| (share of trials / share of the least-squares "
          "weight sum(Delta^2) / fraction outside [-1, 2] / median / mean posterior):")
    for key, rows in table.items():
        print(f"  from mu_{key}:")
        for r in rows:
            print(f"    {r['label']:>16}: {100 * r['share_trials']:5.1f}%  "
                  f"{100 * r['share_delta2']:6.2f}%  {100 * r['frac_outside']:5.1f}%   "
                  f"median {r['median']:6.2f}   posterior {r['mean_posterior']:.2f}")
    print("\nsigma_out is one number per output (the control's residual std); "
          "sigma_w = sigma_out/|Delta| is the per-trial quantity of the position read. "
          "No figure here is filtered.")
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
    _merge_metrics(exp, "reliability", {"run": args.run, "control": metrics["control"],
                                        "levels": levels, "table": table})
    for var, rows in table.items():
        print(f"\n{var}:  level  n  midpoint analytical / from mu_vis / from mu_prop")
        for label, r in rows.items():
            print(f"  {label:>10} {r['n']:6d}  {r['midpoint_analytical']:6.2f} / "
                  f"{r['midpoint_vis']:6.2f} / {r['midpoint_prop']:6.2f}")
    _report(written, exp)


# --------------------------------------------------------------------------- #
# trained runs side by side
# --------------------------------------------------------------------------- #
def cmd_compare(args):
    """The implied-weight analysis on several pipeline runs, on one set of
    figures: what each run's trials allowed and what each network achieved."""
    from cmsi.experiments import design

    rows = []
    for run in args.runs:
        ctx = argparse.Namespace(run=run, control=None, name=None)
        print(f"\n--- {run} ---")
        pred, d, names, cfg, sig_out, _, metrics, arrays = _run_context(ctx)
        row = {"label": run, "metrics": metrics, "arrays": arrays,
               "hidden": list(cfg["model"]["hidden"]),
               "sigma0_sq": cfg["generative"]["sigma0_sq"],
               "eye_sigma_sq": cfg["generative"]["eye_sigma_sq"]}
        if not args.no_floor:
            row["floor"] = design.decoding_floor(d, cfg)
            print(f"decoding floor on sigma_out {row['floor']['fused_mu']:.3f} deg "
                  f"(mu_vis {row['floor']['mu_vis']:.3f}, mu_prop {row['floor']['mu_prop']:.3f})")
        _print_reads(metrics)
        rows.append(row)
    exp = iw.experiment_dir(args.name or ("compare_" + "+".join(args.runs)))
    figs = design.runs_figures(rows)
    written = save_figures(figs, exp / "figures" / "compare", formats=FORMATS)
    _merge_metrics(exp, "compare", design.runs_table(rows))
    design.print_runs(rows)
    _report(written, exp)


# --------------------------------------------------------------------------- #
# what the trials allow, before training
# --------------------------------------------------------------------------- #
def _design_configs(args):
    """[(label, cfg)]: the configs named on the command line, each with --set
    applied, and/or the first of them with --param set to each of --values."""
    from cmsi.utils.paths import ROOT as _root

    paths = args.config or [CONFIGS / "flagship.yaml"]
    overrides = iw.parse_overrides(args.set)
    if args.n is not None:
        overrides["n_trials"] = int(args.n)
    base = []
    for path in paths:
        cfg = load_config(path)
        if overrides:
            cfg = tweak(cfg, **overrides)
        label = str(path)
        try:
            label = str(Path(path).resolve().relative_to(_root))
        except ValueError:
            pass
        base.append((label.replace("configs/", "").replace(".yaml", ""), cfg))
    if not args.values:
        return base
    if not args.param:
        raise SystemExit("--values needs --param KEY")
    label0, cfg0 = base[0]
    values = [yaml.safe_load(v) for v in args.values]
    tag = f"{label0}: " if len(base) > 1 else ""
    return [(f"{tag}{args.param}={v}", tweak(cfg0, **{args.param: v})) for v in values]


def _marks(names):
    """{run: {own_sd, sigma_out}} for the pipeline runs named, from their metrics."""
    out = {}
    for run in names or []:
        path = RESULTS / run / "metrics.json"
        if not path.exists():
            print(f"--mark {run}: results/{run}/metrics.json not found; skipped")
            continue
        m = load_json(path)
        so = m.get("sigma_out")
        if so is None:
            src = m.get("sigma_out_source")
            if src and (RESULTS / src / "metrics.json").exists():
                so = load_json(RESULTS / src / "metrics.json").get("residual_std")
        if so is None:
            print(f"--mark {run}: no sigma_out recorded; skipped")
            continue
        out[run] = {"own_sd": float(m["residual_std"][0]), "sigma_out": float(so[0])}
    return out


def cmd_design(args):
    """Screen configurations without training: the posterior distribution,
    |Delta|, the fraction of ratios that would fall outside [-1, 2] for any
    read-out error, and the floor on that error set by the spike code."""
    from cmsi.experiments import design

    configs = _design_configs(args)
    n = int(configs[0][1]["training"]["n_trials"]) if args.n is None else int(args.n)
    n = min(n, 20000) if args.n is None else n
    rows = []
    for label, cfg in configs:
        how = f", balanced to keep {args.balance:g}" if args.balance else ""
        print(f"\n--- {label}: {n} trials{how} ---")
        stats, arrays = design.design_analysis(cfg, n=n, seed=args.seed, balance=args.balance,
                                               floor=not args.no_floor)
        if "balance" in stats:
            b = stats["balance"]
            print(f"balance: kept {b['n_after']} of {b['n_before']} trials; posterior histogram "
                  + " ".join(f"{100 * h:.0f}" for h in b["posterior_hist_after"])
                  + f" %; worst single-channel AUC deviation "
                    f"{b['anticonfound'].get('max_auc_deviation', float('nan')):.3f} (SS9.4)")
        rows.append({"label": label, "stats": stats, "arrays": arrays})
    marks = _marks(args.mark)
    exp = iw.experiment_dir(args.name or "design")
    figs = design.design_figures(rows, marks)
    written = save_figures(figs, exp / "figures" / "design", formats=FORMATS)
    _merge_metrics(exp, "design", {"n": n, "marks": marks, "configs": design.design_table(rows)})
    design.print_design(rows)
    _report(written, exp)


# --------------------------------------------------------------------------- #
# a new configuration, through the whole pipeline
# --------------------------------------------------------------------------- #
def _stage(script, *argv, check=True):
    """Run one numbered stage as a subprocess, with this interpreter."""
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, argv)]
    print("\n$ " + " ".join(c if " " not in c else repr(c) for c in cmd[1:]), flush=True)
    return subprocess.run(cmd, check=check, cwd=ROOT).returncode


def experiment_config(args):
    """The base config with --set / --n / --epochs / --seed / --quick applied."""
    cfg = load_config(args.config or (CONFIGS / "flagship.yaml"))
    overrides = iw.parse_overrides(args.set)
    if args.quick:
        overrides.setdefault("n_trials", 8000)
        overrides.setdefault("epochs", 60)
    for key in ("n_trials", "epochs", "seed"):
        value = getattr(args, "n" if key == "n_trials" else key)
        if value is not None:
            overrides[key] = int(value)
    if overrides:
        print("config overrides:", overrides)
        cfg = tweak(cfg, **overrides)
    return cfg


def cmd_train(args):
    """Stages 0-4 for one configuration and its p_common = 1 control, into
    results/<name>/ and results/<name>_pcommon1/, exactly as run_all.sh
    would lay them out -- so every analysis in the project applies."""
    name, ctrl = args.name, f"{args.name}_pcommon1"
    twin = f"{name}_twin" if args.twin else None
    for r in (name, ctrl):
        if (RESULTS / r / "model.pt").exists() and not args.force:
            raise SystemExit(f"results/{r}/ already holds a trained network; pass "
                             f"--force to retrain it, or choose another --name")

    cfg = experiment_config(args)
    cdir = CONFIGS / "experiments"
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_path = save_config(cfg, cdir / f"{name}.yaml")
    ctrl_path = save_config(tweak(cfg, p_common=1.0), cdir / f"{ctrl}.yaml")
    print(f"configs written: {cfg_path.relative_to(ROOT)}, {ctrl_path.relative_to(ROOT)}")

    print("\n=== 0. calibrate (reported, not enforced: this is an experiment) ===")
    rc = _stage("00_calibrate.py", "--config", cfg_path, check=False)
    if rc != 0:
        print(f"\nwarning: the calibration gate reported FAIL for '{name}' -- "
              f"continuing, but read results/calibration/{name}/ before trusting "
              f"the causal analyses of this run")

    print("\n=== 1. data ===")
    _stage("01_generate_data.py", "--config", cfg_path, "--name", name)
    _stage("01_generate_data.py", "--config", ctrl_path, "--name", ctrl)
    if twin:
        _stage("01_generate_data.py", "--config", cfg_path, "--name", twin, "--head", "fused")

    print("\n=== 2. train ===")
    _stage("02_train.py", "--data", ctrl, "--run", ctrl)
    _stage("02_train.py", "--data", name, "--run", name)
    if twin:
        _stage("02_train.py", "--data", twin, "--run", twin)

    print("\n=== 3. analyse (the control first, so its residual_std exists) ===")
    _stage("03_analyze.py", "--run", ctrl)
    _stage("03_analyze.py", "--run", name, "--control", ctrl,
           *(["--twin", twin] if twin else []))

    print("\n=== 4. figures ===")
    _stage("04_figures.py", "--run", ctrl)
    _stage("04_figures.py", "--run", name)

    print(f"\ndone -> results/{name}/ and results/{ctrl}/")
    print("next, for example:")
    print(f"  python scripts/06_implied_weight.py sigma       --run {name}")
    print(f"  python scripts/06_implied_weight.py reliability --run {name} --levels 5")
    print(f"  python scripts/06_implied_weight.py figures     --run {name}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    apply_style()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def run_args(sp):
        sp.add_argument("--run", default="flagship", help="results/<run>/ to analyse")
        sp.add_argument("--control", default=None,
                        help="p_common=1 run whose residuals give sigma_out "
                             "(default: the one stage 3 recorded, else pcommon1)")
        sp.add_argument("--name", default=None,
                        help="experiment folder name (default: the run name)")

    sp = sub.add_parser("figures", help="05, 15, 16 (and 04) for one run")
    run_args(sp)
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("sigma", help="sigma_out and per-trial sigma_w on one run, unfiltered")
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

    sp = sub.add_parser("compare", help="several trained runs side by side: readability of "
                                        "the per-trial ratio and what each network achieved")
    sp.add_argument("--runs", nargs="+", required=True, help="results/<run>/ names")
    sp.add_argument("--name", default=None,
                    help="experiment folder name (default: compare_<run>+<run>...)")
    sp.add_argument("--no-floor", action="store_true",
                    help="skip the decoding floor (the slow part)")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("design", help="screen configurations before training: posterior, "
                                       "|Delta|, expected out-of-range ratios, decoding floor")
    sp.add_argument("--config", nargs="*", default=None,
                    help="one or more configs (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides applied to every config; quote brackets")
    sp.add_argument("--param", default=None, help="config key to vary (with --values)")
    sp.add_argument("--values", nargs="*", default=None,
                    help="values of --param, one configuration each (yaml; quote brackets)")
    sp.add_argument("--n", type=int, default=None,
                    help="trials to draw per configuration (default: n_trials, at most 20000)")
    sp.add_argument("--seed", type=int, default=None, help="draw seed (default: the config's)")
    sp.add_argument("--balance", type=float, default=None, metavar="KEEP",
                    help="thin the trials to a flat posterior histogram, keeping this fraction")
    sp.add_argument("--mark", nargs="*", default=None, metavar="RUN",
                    help="trained runs whose own sd / sigma_out to mark on the curves")
    sp.add_argument("--name", default=None, help="experiment folder name (default: design)")
    sp.add_argument("--no-floor", action="store_true", help="skip the decoding floor")
    sp.set_defaults(func=cmd_design)

    sp = sub.add_parser("train", help="run the whole pipeline on a configuration "
                                      "(+ its p_common=1 control) into results/<name>/")
    sp.add_argument("--name", required=True, help="run name -> results/<name>/")
    sp.add_argument("--config", default=None, help="base config (default flagship)")
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides by key; quote values with brackets")
    sp.add_argument("--n", type=int, default=None, help="trials per dataset")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--quick", action="store_true", help="8000 trials, 60 epochs")
    sp.add_argument("--twin", action="store_true",
                    help="also train the always-fuse twin (for figure 07)")
    sp.add_argument("--force", action="store_true",
                    help="retrain even if results/<name>/ already exists")
    sp.set_defaults(func=cmd_train)

    args = p.parse_args()
    args.func(args)
