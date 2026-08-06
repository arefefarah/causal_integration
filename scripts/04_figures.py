"""Stage 4 -- render every figure for a run.

    python scripts/04_figures.py --run baseline
    python scripts/04_figures.py --run baseline --only model

Reads what stages 1-3 already computed, so this is cheap to re-run while you
fiddle with a panel. Figures go to:

    results/<run>/figures/inputs      what the network is shown
    results/<run>/figures/training    did it converge
    results/<run>/figures/model       network vs analytical observer
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi import analysis
from cmsi.data import subset
from cmsi.utils import dataset_path, load_checkpoint, load_dataset, load_json, run_dir
from cmsi.viz import apply_style, inputs, results, save_figures, training


def main(args):
    apply_style()
    out = run_dir(args.run)
    _, cfg, history, splits = load_checkpoint(out / "model.pt")

    dataset_name = (out / "dataset.txt").read_text().strip() \
        if (out / "dataset.txt").exists() else "main"
    d_full, _ = load_dataset(dataset_path(dataset_name))
    groups = ("inputs", "training", "model") if args.only is None else (args.only,)
    written = []

    if "inputs" in groups:
        figs = inputs.all_figures(d_full, cfg)
        written += save_figures(figs, out / "figures" / "inputs")

    if "training" in groups:
        figs = training.all_figures(history)
        written += save_figures(figs, out / "figures" / "training")

    if "model" in groups:
        analysis_file = out / "analysis.npz"
        if not analysis_file.exists():
            raise SystemExit("run 03_analyze.py first -- analysis.npz is missing")
        saved = np.load(analysis_file)
        d = subset(d_full, splits["test"])

        w = saved["fusion_weight"] if "fusion_weight" in saved.files else None
        curves = None
        if w is not None:
            curves = analysis.by_reliability(
                d["disparity"], w, d["sig2_vis"],
                cfg["analysis"]["reliability_levels"],
                cfg["analysis"]["disparity_grid"])

        metrics = load_json(out / "metrics.json") if (out / "metrics.json").exists() else {}
        figs = results.all_figures(
            saved["pred"], d, d_full["target_names"], cfg["analysis"],
            w=w, curves=curves,
            decoding=metrics.get("p_common_decoding_r2"),
            twin_decoding=metrics.get("twin_p_common_decoding_r2"))
        written += save_figures(figs, out / "figures" / "model")

    for path in written:
        print(f"  {path.relative_to(out.parent)}")
    print(f"{len(written)} figures written to {out / 'figures'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="baseline")
    p.add_argument("--only", choices=["inputs", "training", "model"], default=None,
                   help="render just one group")
    main(p.parse_args())
