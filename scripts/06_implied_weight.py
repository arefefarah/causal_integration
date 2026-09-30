"""The implied-weight experiment -- an experiment, not a pipeline stage.

The analyses read an ordinary pipeline run in results/<run>/ and write only to
results/experiments/implied_weight/<name>/, so nothing in results/<run>/ is
ever changed by them. The weight is the hybrid read of each channel
(analysis.hybrid_weight): the mixture-variance root of the variance output
where it is unique, the channel's own position ratio where it is not; every
trial, nothing filtered.

    # 1. the pipeline's weight figures (04, 05, 08v, 15, 16) for one run, and
    #    the per-run numbers on the console
    python scripts/06_implied_weight.py figures     --run flagship

    # 2. the weight at many reliability levels of each input, per channel
    python scripts/06_implied_weight.py reliability --run flagship --inputs vis prop --levels 5
    python scripts/06_implied_weight.py reliability --run flagship --levels 1.5 2.5 3.5 4.5 5.5 6.5

`--control` names the p_common = 1 run whose residuals give sigma_out, the
read-out noise that sets the weight's nominal per-trial sd. Left out, it is
read from the run's own metrics.json (stage 3 records the control it used as
sigma_out_source), falling back to pcommon1.

    # 3. a NEW configuration: run the whole pipeline on it, exactly as
    #    run_all.sh does, into results/<name>/ -- plus a p_common = 1 control
    #    of the same configuration in results/<name>_pcommon1/
    python scripts/06_implied_weight.py train --config configs/exp_weight.yaml --name lownoise
    python scripts/06_implied_weight.py train --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise
    python scripts/06_implied_weight.py train --config configs/realistic.yaml --name real --quick

    # ... after which every analysis in the project applies to it:
    python scripts/06_implied_weight.py figures --run lownoise
    python scripts/04_figures.py --run lownoise --only manuscript

`train` writes the exact configs it used to configs/experiments/<name>.yaml
and <name>_pcommon1.yaml, so a configuration you decide to keep is one
`make all CONFIG=configs/experiments/<name>.yaml` away. The calibration gate
(stage 0) is run and reported but does not stop an experiment; read its
verdict in results/calibration/<name>/.

`--set key=value` (repeatable) changes any config key by name, in whichever
section owns it (utils.tweak). QUOTE values that contain brackets, because
zsh treats [ ] as a glob:  --set 'sigma2_vis_range=[1,4]'  --set rf_width=4
"""

import argparse
import subprocess
import sys

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
    return pred, d, names, cfg, sig_out, metrics, arrays


def _report(written, exp):
    for path in written:
        if path.suffix == ".png":
            print(f"  {path.relative_to(ROOT)}")
    print(f"-> {exp.relative_to(ROOT)}/")


# --------------------------------------------------------------------------- #
# analyses of an existing run
# --------------------------------------------------------------------------- #
def cmd_figures(args):
    pred, d, names, cfg, sig_out, metrics, arrays = _run_context(args)
    exp = iw.experiment_dir(args.name or args.run)
    figs = iw.baseline_figures(pred, d, names, sig_out, cfg["analysis"], arrays, metrics)
    written = save_figures(figs, exp / "figures" / "figures", formats=FORMATS)
    iw.print_summary(metrics)
    _merge_metrics(exp, "figures", {k: metrics[k] for k in
                                    ("run", "control", "sigma_out", "residual_std_self",
                                     "analytical", "reads", "consistency")})
    _report(written, exp)


def _levels(values):
    if len(values) == 1 and float(values[0]).is_integer() and float(values[0]) >= 2:
        return int(values[0])
    return [float(v) for v in values]


def cmd_reliability(args):
    pred, d, names, cfg, sig_out, metrics, _ = _run_context(args)
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
        print(f"\n{var}:  level  n  midpoint analytical / vis channel / prop channel")
        for label, r in rows.items():
            print(f"  {label:>10} {r['n']:6d}  {r['midpoint_analytical']:6.2f} / "
                  f"{r['midpoint_vis']:6.2f} / {r['midpoint_prop']:6.2f}")
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
    print(f"  python scripts/06_implied_weight.py figures     --run {name}")
    print(f"  python scripts/06_implied_weight.py reliability --run {name} --levels 5")


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

    sp = sub.add_parser("figures", help="the pipeline's weight figures (04, 05, 08v, 15, 16) "
                                        "for one run")
    run_args(sp)
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("reliability", help="weight vs disparity per reliability level")
    run_args(sp)
    sp.add_argument("--inputs", nargs="+", default=["vis", "prop"],
                    choices=list(iw.INPUTS), help="which input's noise to split by")
    sp.add_argument("--levels", nargs="+", type=float, default=[5],
                    help="one integer = that many quantile bins; several numbers = "
                         "level centres (nearest match), e.g. 1.5 3.5 6")
    sp.set_defaults(func=cmd_reliability)

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
