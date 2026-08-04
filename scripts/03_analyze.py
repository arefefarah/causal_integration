"""Stage 3 -- compare the trained network to the analytical observer.

    python scripts/03_analyze.py --run baseline
    python scripts/03_analyze.py --run baseline --twin twin   # emergent vs imposed

Everything is computed on the test split stored in the checkpoint. Writes:

    results/<run>/metrics.json     every number, for the write-up
    results/<run>/analysis.npz     the per-trial arrays stage 4 plots

Splitting the numbers from the figures means you can re-plot without re-fitting
any decoders, and diff metrics.json between runs.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from cmsi import analysis
from cmsi.data import subset
from cmsi.models import hidden_activations, predict
from cmsi.utils import (
    dataset_path,
    load_checkpoint,
    load_dataset,
    run_dir,
    save_json,
)


def analyse(run, twin=None):
    out = run_dir(run)
    model, cfg, history, splits = load_checkpoint(out / "model.pt")
    acfg = cfg["analysis"]

    # the dataset the checkpoint's config points at, restricted to the test split
    d_full, _ = load_dataset(dataset_path(_dataset_name(out, cfg)))
    d = subset(d_full, splits["test"])
    names = d_full["target_names"]

    pred = predict(model, d["X"])
    target = np.stack([d[k] for k in names], axis=1)

    metrics = {"run": run, "head": cfg["model"]["head"],
               "n_test": int(len(pred)),
               "best_val": history["best_val"], "best_epoch": history["best_epoch"]}
    arrays = {"pred": pred, "disparity": d["disparity"], "p_common": d["p_common"]}

    # --- 1. accuracy of the readout ---------------------------------------
    metrics["accuracy"] = analysis.accuracy(pred, target, names)
    analysis.print_accuracy(metrics["accuracy"], "accuracy on the test split")

    # --- 2. does the readout imply the optimal fusion weight? -------------
    # only meaningful for the causal head: the twin always fuses, so w == 1
    # by construction and there is nothing to compare against.
    if cfg["model"]["head"] == "causal":
        w = analysis.fusion_weight(pred[:, 0], d["seg_vis_mu"], d["fused_mu"],
                                   acfg["min_separation"])
        arrays["fusion_weight"] = w
        metrics["implied_weight_vs_p_common"] = analysis.compare(d["p_common"], w)

        midpoint, sharpness = analysis.transition_fit(np.abs(d["disparity"]), w)
        opt_mid, opt_sharp = analysis.transition_fit(np.abs(d["disparity"]), d["p_common"])
        metrics["transition"] = {
            "network": {"midpoint_deg": midpoint, "sharpness": sharpness},
            "analytical": {"midpoint_deg": opt_mid, "sharpness": opt_sharp},
        }
        print("\nimplied fusion weight vs analytical p(C=1):",
              _round(metrics["implied_weight_vs_p_common"]))
        print("transition midpoint (deg):  network "
              f"{midpoint:.2f}   analytical {opt_mid:.2f}")

        # --- 3. decision strategy -----------------------------------------
        metrics["strategy"] = analysis.strategy_fit(
            pred[:, 0], d["p_common"], d["fused_mu"], d["seg_vis_mu"])
        print("decision strategy:", _round(metrics["strategy"]))

    # --- 4. what do the hidden layers carry? ------------------------------
    acts = hidden_activations(model, d["X"])
    pc_decoding = analysis.decode_by_layer(
        acts, d["p_common"], acfg["ridge_alpha"], acfg["decoder_test_size"])
    metrics["p_common_decoding_r2"] = analysis.summarise(pc_decoding)
    print("\np(C=1) decodable from:", _round(metrics["p_common_decoding_r2"]))

    estimate_decoding = analysis.decode(
        acts["msl"], d["mu_vis"], acfg["ridge_alpha"], acfg["decoder_test_size"])
    metrics["mu_vis_decoding_r2"] = estimate_decoding["r2"]
    print(f"mu_vis decodable from MSL: r2={estimate_decoding['r2']:.3f}")

    # --- 5. emergent vs imposed (optional) --------------------------------
    if twin:
        twin_model, twin_cfg, _, twin_splits = load_checkpoint(run_dir(twin) / "model.pt")
        twin_acts = hidden_activations(twin_model, d["X"])
        twin_decoding = analysis.decode_by_layer(
            twin_acts, d["p_common"], acfg["ridge_alpha"], acfg["decoder_test_size"])
        metrics["twin_p_common_decoding_r2"] = analysis.summarise(twin_decoding)
        print(f"p(C=1) decodable from the '{twin}' twin:",
              _round(metrics["twin_p_common_decoding_r2"]))

    save_json(metrics, out / "metrics.json")
    np.savez_compressed(out / "analysis.npz", **arrays)
    print(f"\nwrote {out / 'metrics.json'} and {out / 'analysis.npz'}")
    return metrics


def _dataset_name(out, cfg):
    """Which dataset this run was trained on (recorded at train time)."""
    marker = out / "dataset.txt"
    return marker.read_text().strip() if marker.exists() else "main"


def _round(obj, nd=3):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in obj.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="baseline")
    p.add_argument("--twin", default=None,
                   help="run name of an always-fuse twin, for the emergence test")
    args = p.parse_args()
    analyse(args.run, args.twin)
