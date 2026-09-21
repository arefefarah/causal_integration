"""The sweep experiment: one configuration key, several values.

Trains one causal network and one p_common = 1 control per value of a
single config key -- a prior width, a noise range, the hidden size, the
receptive-field width, anything utils.tweak can reach -- and runs the
implied-weight analysis on each, so that the effect of that one knob on the
weight, on sigma_out, on the position-regression slope and on the
readability of the per-trial ratio can be read off one set of comparison
figures. This is the systematic range testing of the design discussion:
vary the mean spread alone (sigma0_sq), or one noise range alone, with
everything else held at the base configuration.

Every variant draws its own datasets, because the key usually changes the
generative model; they are cached under data/experiments/sweep/ with the
config that produced them and reused only if that config matches. With
`balance` the causal dataset of every variant is thinned to a flat
posterior histogram (cmsi.experiments.design.balance_posterior) before
training -- the safe reading of "a balanced weight distribution", which
touches the trial distribution and never the targets.

Everything is written under results/experiments/sweep/<name>/; results/<run>/
and data/ are never touched. The per-variant analysis and figures come from
cmsi.experiments.implied_weight, the variant plumbing and the comparison
figures from cmsi.experiments.architecture. Entry point: scripts/09_sweep.py.
"""

import numpy as np

from cmsi.experiments import architecture as arch

EXPERIMENT = "sweep"


# --------------------------------------------------------------------------- #
# where things go
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/sweep/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def experiment_data_dir():
    """data/experiments/sweep/ -- one pair of datasets per variant."""
    from cmsi.utils.paths import DATA

    path = DATA / "experiments" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def value_label(value):
    """A short, filesystem-safe name for one value: 100, 1.5, 2-2.5, 32-128."""
    if isinstance(value, (list, tuple)):
        return "-".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}"
    return str(value).replace("/", "_").replace(" ", "")


def variant_label(param, value):
    return f"{param}={value_label(value)}"


def sort_key(value):
    """Scalars by value, lists by their mean, anything else by text."""
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, float)) for v in value):
        return float(np.mean(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float("inf")


def variant_config(base_cfg, param, value):
    """The base config with `param` set to `value` (utils.tweak: the key is
    found in whichever section owns it; unknown keys raise)."""
    from cmsi.utils import tweak

    return tweak(base_cfg, **{param: value})


def variant_datasets(cfg, exp_dir, label, balance=None, regen=False, verbose=True):
    """The causal dataset and its p_common = 1 control for one variant,
    drawn from that variant's config and cached under
    data/experiments/sweep/<experiment>_<label>_{causal,control}.npz.

    With `balance` the causal dataset is drawn with n_trials/balance trials
    and thinned to a flat posterior histogram, so it ends up with about
    n_trials; the control is left alone (its posterior is 1 on every trial).
    The kept fraction is recorded in the stored config as `balance_posterior`
    and enters the cache signature, so a balanced and an unbalanced draw of
    the same configuration are never confused.

    A cached dataset is reused only if it came from the same generative and
    encoding parameters, trial count, seed and balance; otherwise this stops
    and says so (--regen redraws)."""
    from cmsi.data import make_dataset
    from cmsi.experiments.design import balance_posterior
    from cmsi.utils import load_dataset, save_dataset, tweak

    def signature(c):
        return {"generative": c["generative"], "encoding": c["encoding"],
                "n_trials": c["training"]["n_trials"], "seed": c["seed"],
                "balance": c.get("balance_posterior")}

    ddir = experiment_data_dir()
    paths = {"causal": ddir / f"{exp_dir.name}_{label}_causal.npz",
             "control": ddir / f"{exp_dir.name}_{label}_control.npz"}
    causal_cfg = dict(cfg)
    if balance:
        causal_cfg["balance_posterior"] = float(balance)
    cfgs = {"causal": causal_cfg, "control": tweak(cfg, p_common=1.0)}
    data = {}
    for kind, path in paths.items():
        if path.exists() and not regen:
            d, stored = load_dataset(path)
            if signature(stored) != signature(cfgs[kind]):
                raise SystemExit(
                    f"{path.name} was generated from a different config than the one "
                    f"now requested (generative/encoding/n_trials/seed/balance differ). "
                    f"Re-run with --regen to redraw the datasets of experiment "
                    f"'{exp_dir.name}', or use a new --name.")
            data[kind] = (d, stored)
            if verbose:
                print(f"{kind}: reusing {path.relative_to(ddir.parents[1])}")
            continue
        n = cfgs[kind]["training"]["n_trials"]
        if kind == "causal" and balance:
            n_draw = int(round(n / balance))
            if verbose:
                print(f"{kind}: generating {n_draw} trials (p_common="
                      f"{cfgs[kind]['generative']['p_common']}) and keeping ~{n} with a "
                      f"flat posterior histogram -> {path.name}")
            d = make_dataset(cfgs[kind], n=n_draw)
            d, info = balance_posterior(d, balance, seed=cfgs[kind]["seed"])
            if verbose:
                print(f"        kept {info['n_after']} of {info['n_before']} "
                      f"({100 * info['kept']:.0f}%); posterior histogram now "
                      + " ".join(f"{100 * h:.0f}" for h in info["posterior_hist_after"]) + " %")
        else:
            if verbose:
                print(f"{kind}: generating {n} trials "
                      f"(p_common={cfgs[kind]['generative']['p_common']}) -> {path.name}")
            d = make_dataset(cfgs[kind])
        save_dataset(d, cfgs[kind], path)
        data[kind] = (d, cfgs[kind])
    return data


def train_variant(param, value, base_cfg, exp_dir, order, balance=None, epochs=None,
                  seed=None, regen=False, verbose=True):
    """Datasets, the two networks and the analysis for one value of `param`.
    Returns what architecture.train_variant returns."""
    label = variant_label(param, value)
    cfg = variant_config(base_cfg, param, value)
    data = variant_datasets(cfg, exp_dir, value_label(value), balance=balance,
                            regen=regen, verbose=verbose)
    extra = {"param": param, "value": value, "order": [float(order)],
             "balance_posterior": float(balance) if balance else None,
             "sigma_out_source": "control trained on the same configuration"}
    return arch.train_variant(None, data, exp_dir, epochs=epochs, seed=seed,
                              verbose=verbose, label=label, extra=extra)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def load_variants(exp_dirs):
    rows = arch.load_variants(exp_dirs, label_by="variant")
    # label by the value alone when every row varies the same key; the key
    # goes into the axis title instead
    params = {m.get("param") for m in rows}
    if len(params) == 1 and len({m.get("experiment") for m in rows}) == 1:
        for m in rows:
            m["label"] = value_label(m["value"])
    return rows


def sweep_param(rows):
    params = sorted({str(m.get("param")) for m in rows})
    return params[0] if len(params) == 1 else " / ".join(params)


def variant_figures(rows):
    param = sweep_param(rows)
    return arch.variant_figures(rows, by=param, legend=param, xlabel=param, units=False)


def variants_table(rows):
    out = arch.variants_table(rows)
    for r, m in zip(out, rows, strict=False):
        r["param"], r["value"] = m.get("param"), m.get("value")
        r["balance_posterior"] = m.get("balance_posterior")
        rb = m["reads"]["vis"].get("readability", {})
        r["outside_vis"] = rb.get("outside")
        r["own_sd_over_sigma_out_vis"] = rb.get("own_sd_over_sigma_out")
        pt = (m.get("per_trial") or {}).get("estimators", {})
        r["variance_read_sd_vs_posterior"] = pt.get("variance", {}).get("sd_vs_posterior")
        r["joint_read_outside"] = pt.get("joint", {}).get("outside")
    return out


def print_variants(rows):
    width = max(12, max(len(m["label"]) for m in rows))
    print(f"\n{'variant':>{width}} {'sig_out v':>9} {'own sd v':>8} {'out% v':>6} "
          f"{'out% joint':>10} {'var sd':>6} {'mid_ls v':>8} {'mid_an':>6} {'slope v':>7} "
          f"{'slope p':>7} {'kept v':>6} {'R2 mu_vis':>9}")
    for m in rows:
        rv, rp = m["reads"]["vis"], m["reads"]["prop"]
        rb = rv.get("readability", {})
        own = rb.get("own_sd", np.nan)
        out = rb.get("outside", np.nan)
        pt = (m.get("per_trial") or {}).get("estimators", {})
        out_j = pt.get("joint", {}).get("outside", np.nan)
        var_sd = pt.get("variance", {}).get("sd_vs_posterior", np.nan)
        print(f"{m['label']:>{width}} {rv['sigma_out']:9.3f} {own:8.3f} {100 * out:6.1f} "
              f"{100 * out_j:10.1f} {var_sd:6.3f} "
              f"{rv['midpoint_ls_deg']:8.2f} {m['analytical']['midpoint_deg']:6.2f} "
              f"{rv['position_regression']['slope']:7.3f} "
              f"{rp['position_regression']['slope']:7.3f} "
              f"{100 * rv['frac_readable']:5.0f}% "
              f"{next(a['r2'] for a in m['accuracy'] if a['output'] == 'mu_vis'):9.3f}")
    print("  out% joint: ratios outside [-1, 2] when both position outputs are read together; "
          "var sd: sd of the weight read from the variance outputs against the posterior, "
          "all trials")
