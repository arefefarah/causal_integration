"""Stage 0 -- calibrate a config BEFORE any training (design SS8, SS9.4, SS4).

    python scripts/00_calibrate.py --config configs/flagship.yaml
    python scripts/00_calibrate.py --config configs/flagship.yaml --n 40000

Prints PASS / WARN / FAIL per design criterion and writes

    results/calibration/<config-name>/metrics.json
    results/calibration/<config-name>/figures/*.png

Exit code is 1 when any FAIL is present, so run_all.sh stops before wasting a
training run on a mis-calibrated dataset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cmsi.analysis.calibration import calibrate, checks
from cmsi.utils import load_config, save_json
from cmsi.utils.paths import RESULTS
from cmsi.viz import SIZE, apply_style, label_panels, save_figures


def figures(stats, arrays, out_dir):
    from matplotlib import pyplot as plt
    figs = {}

    # SS8.1 posterior histogram
    fig, ax = plt.subplots(figsize=SIZE["single"])
    ax.hist(arrays["post_c1"], bins=60, color="#4878cf")
    ax.axvspan(0.2, 0.8, color="orange", alpha=0.12)
    mi = stats["posterior_mass_intermediate"]
    ax.set(xlabel="analytical p(C=1|x)", ylabel="trials",
           title=f"posterior histogram -- {mi:.0%} intermediate (target ~25%)")
    fig.tight_layout()
    figs["01_posterior_histogram"] = fig

    # SS8.2 joint (w, |Delta|)
    fig, axes = plt.subplots(1, 2, figsize=SIZE["pair"])
    for ax, key, name in ((axes[0], "delta_vis", "visual output"),
                          (axes[1], "delta_prop", "hand output")):
        *_, mesh = ax.hist2d(arrays["post_c1"], np.abs(arrays[key]), bins=50,
                             cmap="Blues")
        mesh.set_rasterized(True)      # 2,500 cells -> one image in the SVG
        ax.set(xlabel="p(C=1|x)", ylabel="|Delta| (deg)",
               title=f"joint (w, |Delta|): {name}")
    label_panels(axes)
    fig.tight_layout()
    figs["02_joint_w_delta"] = fig

    # SS8.3 reliability coverage: posterior spread within disparity bins
    fig, ax = plt.subplots(figsize=SIZE["single"])
    rng_ = stats["reliability_coverage"]["within_bin_posterior_range"]
    std_ = stats["reliability_coverage"]["within_bin_posterior_std"]
    x = np.arange(len(rng_))
    ax.bar(x, rng_, color="#4878cf", label="range (what SS7.2 leverages)")
    ax.bar(x, std_, color="#2a4a80", width=0.45, label="std")
    ax.axhline(0.15, ls="--", color="crimson", lw=1, label="power floor 0.15")
    ax.set(xlabel="|disparity| bin (quantiles)",
           ylabel="spread of p(C=1|x) within bin",
           title="reliability-driven posterior spread at matched disparity")
    ax.legend()
    fig.tight_layout()
    figs["03_reliability_coverage"] = fig

    # SS4 Poisson validity
    fig, ax = plt.subplots(figsize=SIZE["single"])
    po = stats["poisson"]
    labels = [f"sigma2={po[t]['sigma2']:g}" for t in ("min", "max")]
    vals = [po[t]["inflation"] * 100 for t in ("min", "max")]
    ax.bar(labels, vals, color="#4878cf")
    ax.axhline(5, ls="--", color="crimson", lw=1, label="5% tolerance")
    ax.set(ylabel="decoded variance above nominal (%)",
           title="Poisson validity (SS4)")
    ax.legend()
    fig.tight_layout()
    figs["04_poisson_validity"] = fig

    paths = save_figures(figs, out_dir)
    return [p.name for p in paths]


def main(args):
    apply_style()
    cfg = load_config(args.config)
    name = Path(args.config).stem if args.config else "default"
    out = RESULTS / "calibration" / name

    print(f"calibrating {name} on {args.n} trials ...")
    stats, arrays = calibrate(cfg, n=args.n)
    results = checks(stats, cfg)

    print()
    worst = "PASS"
    for level, msg in results:
        print(f"  [{level}] {msg}")
        if level == "FAIL" or (level == "WARN" and worst == "PASS"):
            worst = level

    save_json({"stats": stats, "checks": [list(r) for r in results]},
              out / "metrics.json")
    written = figures(stats, arrays, out / "figures")
    print(f"\nwrote {out}/metrics.json and {len(written)} figures")
    if worst == "FAIL":
        print("calibration FAILED -- fix the config before training")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None,
                   help="yaml config (default: configs/default.yaml)")
    p.add_argument("--n", type=int, default=20000)
    main(p.parse_args())
