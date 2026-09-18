"""Stage 4 -- render every figure for a run.

    python scripts/04_figures.py --run baseline
    python scripts/04_figures.py --run baseline --only model

Reads what stages 1-3 already computed, so this is cheap to re-run while you
fiddle with a panel. Figures go to:

    results/<run>/figures/inputs      what the network is shown
    results/<run>/figures/training    did it converge
    results/<run>/figures/model       network vs analytical observer
    results/manuscript/<figure>/      the manuscript figures built from this
                                      run, one folder per figure, on the
                                      standard panel (viz/manuscript.py);
                                      results/manuscript_<run>/ for a run
                                      other than the flagship
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi import analysis
from cmsi.data import subset
from cmsi.utils import dataset_path, load_checkpoint, load_dataset, load_json, run_dir
from cmsi.viz import apply_style, inputs, results, save_figures, training
from cmsi.viz.manuscript import manuscript_dir


def main(args):
    apply_style()
    out = run_dir(args.run)
    _, cfg, history, splits = load_checkpoint(out / "model.pt")

    dataset_name = (out / "dataset.txt").read_text().strip() \
        if (out / "dataset.txt").exists() else "main"
    d_full, _ = load_dataset(dataset_path(dataset_name))
    groups = (("inputs", "training", "model", "manuscript") if args.only is None
              else (args.only,))
    written = []

    if "inputs" in groups:
        figs = inputs.all_figures(d_full, cfg)
        written += save_figures(figs, out / "figures" / "inputs")

    if "training" in groups:
        figs = training.all_figures(history)
        written += save_figures(figs, out / "figures" / "training")

    if "model" in groups or "manuscript" in groups:
        analysis_file = out / "analysis.npz"
        if not analysis_file.exists():
            raise SystemExit("run 03_analyze.py first -- analysis.npz is missing")
        saved = np.load(analysis_file)
        d = subset(d_full, splits["test"])

        w = saved["fusion_weight"] if "fusion_weight" in saved.files else None
        w_prop = saved["fusion_weight_prop"] if "fusion_weight_prop" in saved.files else None
        curves = None
        if w is not None:
            curves = analysis.by_reliability(
                d["disparity"], w, d["sig2_vis"],
                cfg["analysis"]["reliability_levels"],
                cfg["analysis"]["disparity_grid"])

        metrics = load_json(out / "metrics.json") if (out / "metrics.json").exists() else {}
        if "sigma_out" not in metrics and "residual_std" in metrics:
            # metrics.json predates the sigma_out key: resolve it from the
            # named control, exactly as 03_analyze.py did, so that figures
            # never fall back to the flagship's own (inflated) residuals.
            control = metrics.get("sigma_out_source", "self")
            cpath = run_dir(control, create=False) / "metrics.json"
            if control != "self" and cpath.exists():
                metrics["sigma_out"] = load_json(cpath)["residual_std"]
            else:
                metrics["sigma_out"] = metrics["residual_std"]
        if "model" in groups:
            figs = results.all_figures(
                saved["pred"], d, d_full["target_names"], cfg["analysis"],
                w=w, w_prop=w_prop, curves=curves,
                decoding=metrics.get("post_c1_decoding_r2"),
                twin_decoding=metrics.get("twin_post_c1_decoding_r2"),
                saved=saved, metrics=metrics)
            written += save_figures(figs, out / "figures" / "model")
        if "manuscript" in groups:
            figs = results.manuscript_panels(
                saved["pred"], d, d_full["target_names"], cfg["analysis"],
                w, w_prop, saved, metrics)
            mdir = manuscript_dir(args.run)
            written += save_figures(figs, mdir, formats=("png", "tif", "svg", "pdf"))
            print(f"manuscript figures from run '{args.run}' -> {mdir}")

    for path in written:
        print(f"  {path.relative_to(out.parent)}")   # both live under results/
    print(f"{len(written)} figures written under {out.parent}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="baseline")
    p.add_argument("--only", choices=["inputs", "training", "model", "manuscript"],
                   default=None,
                   help="render just one group")
    main(p.parse_args())
