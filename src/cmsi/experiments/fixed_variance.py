"""The fixed-variance experiment: every input at one noise level.

In the pipeline each trial draws its three measurement variances from a
range, sig2_vis ~ U(a, b) and so on, so reliability varies from trial to
trial and the network has to read it out of the spike counts. This
experiment pins each variance to a single value instead -- one sigma2 per
input, the same on every trial -- and asks what the implied weight, sigma_out
and sigma_w look like when reliability is not a variable at all, and how
they move as that single value is changed.

Nothing in the pipeline changes for this. A fixed variance is a range whose
two ends coincide, sigma2_vis_range: [v, v], which the sampler, the observer,
the encoder and the config check all accept as they are. What changes is
what the run means: with reliability constant the posterior is a function of
the measurements alone, the reliability-dependent analyses have nothing to
vary, and the calibration gate's reliability-coverage check fails by
construction (it is not run here).

Each variant is one (sigma2_vis, sigma2_prop, sigma2_eye) triple. For each,
the causal network and a p_common = 1 control are trained on their own
datasets, so every variant has its own sigma_out, and the per-variant
figures and the cross-variant comparison are the ones of the architecture
experiment (cmsi.experiments.architecture), which this module reuses.

Everything is written under results/experiments/fixed_variance/<name>/ and
data/experiments/fixed_variance/. Entry point: scripts/08_fixed_variance.py.
"""

import itertools

from cmsi.experiments import architecture as arch

EXPERIMENT = "fixed_variance"
INPUTS = ("vis", "prop", "eye")


# --------------------------------------------------------------------------- #
# where things go
# --------------------------------------------------------------------------- #
def experiment_dir(name, create=True):
    """results/experiments/fixed_variance/<name>/"""
    from cmsi.utils.paths import RESULTS

    path = RESULTS / "experiments" / EXPERIMENT / name
    if create:
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def experiment_data_dir():
    """data/experiments/fixed_variance/ -- one pair of datasets per variant."""
    from cmsi.utils.paths import DATA

    path = DATA / "experiments" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def range_midpoints(cfg):
    """The middle of each input's noise range in the config -- the default
    fixed value for any input the command line does not set."""
    gen = cfg["generative"]
    return {k: float(sum(gen[f"sigma2_{k}_range"]) / 2) for k in INPUTS}


def fixed_config(cfg, sigma2):
    """`cfg` with every input's variance pinned to sigma2[input].

    The range collapses to [v, v]. The analysis' reliability levels are set
    to the fixed visual value too, so the pipeline's figure 05 draws its one
    curve under the right label instead of the nearest of the old levels.
    """
    from cmsi.utils import tweak

    over = {f"sigma2_{k}_range": [float(sigma2[k]), float(sigma2[k])] for k in INPUTS}
    over["reliability_levels"] = [float(sigma2["vis"])]
    return tweak(cfg, **over)


def variant_label(sigma2):
    """'v3_p5.7_e8.25' -- the three fixed values, in deg^2."""
    return "_".join(f"{k[0]}{float(sigma2[k]):g}" for k in INPUTS)


def variant_grid(cfg, vis=None, prop=None, eye=None):
    """Every combination of the values given per input (an input with no
    values keeps its range midpoint), as a list of sigma2 dicts in order."""
    mid = range_midpoints(cfg)
    axes = {"vis": vis or [mid["vis"]], "prop": prop or [mid["prop"]],
            "eye": eye or [mid["eye"]]}
    return [dict(zip(INPUTS, combo, strict=False))
            for combo in itertools.product(*(axes[k] for k in INPUTS))]


def variant_datasets(cfg, exp_dir, label, regen=False, verbose=True):
    """The causal dataset and its p_common = 1 control for one variant,
    drawn from that variant's fixed-variance config and cached under
    data/experiments/fixed_variance/<experiment>_<label>_{causal,control}.npz.

    A cached dataset is reused only if it came from the same generative and
    encoding parameters, trial count and seed; otherwise this stops and
    says so (--regen redraws)."""
    from cmsi.data import make_dataset
    from cmsi.utils import load_dataset, save_dataset, tweak

    def signature(c):
        return {"generative": c["generative"], "encoding": c["encoding"],
                "n_trials": c["training"]["n_trials"], "seed": c["seed"]}

    ddir = experiment_data_dir()
    paths = {"causal": ddir / f"{exp_dir.name}_{label}_causal.npz",
             "control": ddir / f"{exp_dir.name}_{label}_control.npz"}
    cfgs = {"causal": cfg, "control": tweak(cfg, p_common=1.0)}
    data = {}
    for kind, path in paths.items():
        if path.exists() and not regen:
            d, stored = load_dataset(path)
            if signature(stored) != signature(cfgs[kind]):
                raise SystemExit(
                    f"{path.name} was generated from a different config than the one "
                    f"now requested (generative/encoding/n_trials/seed differ). Re-run "
                    f"with --regen to redraw the datasets of experiment "
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


def train_variant(sigma2, base_cfg, exp_dir, hidden=None, epochs=None, seed=None,
                  regen=False, verbose=True):
    """Datasets, the two networks and the analysis for one sigma2 triple.
    Returns what architecture.train_variant returns."""
    label = variant_label(sigma2)
    cfg = fixed_config(base_cfg, sigma2)
    data = variant_datasets(cfg, exp_dir, label, regen=regen, verbose=verbose)
    extra = {"sigma2": {k: float(sigma2[k]) for k in INPUTS},
             "order": [float(sigma2[k]) for k in INPUTS],
             "sigma_out_source": "control trained at the same fixed variances"}
    return arch.train_variant(hidden, data, exp_dir, epochs=epochs, seed=seed,
                              verbose=verbose, label=label, extra=extra)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def load_variants(exp_dirs):
    return arch.load_variants(exp_dirs, label_by="variant")


def variant_figures(rows):
    return arch.variant_figures(rows, by="fixed variance", legend="sigma2 (vis, prop, eye)",
                                xlabel="fixed sigma2 per input (deg^2)", units=False)


def variants_table(rows):
    out = arch.variants_table(rows)
    for r, m in zip(out, rows, strict=False):
        r["sigma2"] = m.get("sigma2")
    return out


def print_variants(rows):
    width = max(12, max(len(m["label"]) for m in rows))
    print(f"\n{'variant':>{width}} {'sig_out v':>9} {'sig_out p':>9} {'mid_ls v':>8} "
          f"{'mid_an':>7} {'slope v':>7} {'slope p':>7} {'kept v':>6} {'R2 mu_vis':>9}")
    for m in rows:
        rv, rp = m["reads"]["vis"], m["reads"]["prop"]
        print(f"{m['label']:>{width}} {rv['sigma_out']:9.3f} {rp['sigma_out']:9.3f} "
              f"{rv['midpoint_ls_deg']:8.2f} {m['analytical']['midpoint_deg']:7.2f} "
              f"{rv['position_regression']['slope']:7.3f} "
              f"{rp['position_regression']['slope']:7.3f} "
              f"{100 * rv['frac_readable']:5.0f}% "
              f"{next(a['r2'] for a in m['accuracy'] if a['output'] == 'mu_vis'):9.3f}")
