"""Stage 3 -- compare the trained network to the analytical observer.

    python scripts/03_analyze.py --run baseline
    python scripts/03_analyze.py --run baseline --twin twin      # emergent vs imposed
    python scripts/03_analyze.py --run flagship --control pcommon1
                                 # sigma_out for the sigma_w filter (SS7.1)

Everything is computed on the test split stored in the checkpoint. Writes:

    results/<run>/metrics.json     every number, for the write-up
    results/<run>/analysis.npz     the per-trial arrays stage 4 plots

Control-run behaviour (SS9.1/SS9.2): when the run's own config has
p_common == 1 the residual std per output is saved as sigma_out and the target
reduction to the fused solution is asserted; when p_common == 0 the hand
output's visual bias is measured and reported (it must be ~0 -- any residual
bias is a pipeline leak).
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cmsi import analysis
from cmsi.data import subset
from cmsi.models import hidden_activations, predict
from cmsi.utils import (
    dataset_path,
    load_checkpoint,
    load_dataset,
    load_json,
    run_dir,
    save_json,
)


def analyse(run, twin=None, control=None):
    out = run_dir(run)
    model, cfg, history, splits = load_checkpoint(out / "model.pt")
    acfg = cfg["analysis"]
    p_prior = cfg["generative"]["p_common"]

    # the dataset the checkpoint's config points at, restricted to the test split
    d_full, _ = load_dataset(dataset_path(_dataset_name(out, cfg)))
    d = subset(d_full, splits["test"])
    names = d_full["target_names"]

    pred = predict(model, d["X"])
    target = np.stack([d[k] for k in names], axis=1)

    metrics = {"run": run, "head": cfg["model"]["head"], "p_common": p_prior,
               "n_test": int(len(pred)),
               "best_val": history["best_val"], "best_epoch": history["best_epoch"]}
    if "rejection_rate" in d_full:
        metrics["rejection_rate"] = float(np.asarray(d_full["rejection_rate"]).ravel()[0])
    arrays = {"pred": pred, "target": target,
              "disparity": d["disparity"], "post_c1": d["post_c1"]}

    # --- 1. accuracy of the readout ---------------------------------------
    metrics["accuracy"] = analysis.accuracy(pred, target, names)
    analysis.print_accuracy(metrics["accuracy"], "accuracy on the test split")

    # residual std per output: on a p_common = 1 control this IS sigma_out
    resid_std = analysis.sigma_out(pred, target)
    metrics["residual_std"] = resid_std.tolist()

    # --- control-run assertions (SS9.1 / SS9.2) ---------------------------
    if p_prior >= 1.0:
        # targets must reduce exactly to the fused (always-integrate) solution
        red = max(float(np.abs(d["mu_vis"] - d["fused_mu"]).max()),
                  float(np.abs(d["mu_prop"] - d["fused_mu"]).max()),
                  float(np.abs(d["var_vis"] - d["fused_var"]).max()),
                  float(np.abs(d["var_prop"] - d["fused_var"]).max()))
        metrics["pcommon1_target_reduction_maxdiff"] = red
        print(f"\np_common=1 control: max |target - fused| = {red:.2e} "
              f"(must be ~0); sigma_out per output = "
              + ", ".join(f"{s:.3f}" for s in resid_std))
    if p_prior <= 0.0:
        # hand output must show ZERO visual bias: regress its error on disparity
        err = pred[:, names.index("mu_prop")] - d["seg_prop_mu"]
        slope = float(np.polyfit(d["disparity"], err, 1)[0])
        metrics["pcommon0_visual_bias_slope"] = slope
        print(f"\np_common=0 control: visual-bias slope of the hand output = "
              f"{slope:+.4f} (must be ~0; residual bias = pipeline leak)")

    # sigma_out for the sigma_w filter: from the control run if given (SS7.1).
    # Never fatal -- a missing, stale, or mis-specified control degrades to this
    # run's own residuals with a loud warning, because on the flagship those
    # residuals also contain any causal-inference misweighting and so give a
    # conservative (too large) sigma_w rather than a wrong answer.
    sig_out = resid_std
    metrics["sigma_out_source"] = "self"
    if control is not None:
        cpath = run_dir(control, create=False) / "metrics.json"
        if not cpath.exists():
            print(f"\nwarning: control run '{control}' has no metrics.json -- "
                  f"run 03_analyze.py --run {control} first. "
                  f"Falling back to this run's own residuals for sigma_out.")
        else:
            cmetrics = load_json(cpath)
            if "residual_std" not in cmetrics:
                print(f"\nwarning: '{control}/metrics.json' predates the "
                      f"residual_std field (written by an older 03_analyze.py). "
                      f"Re-run 03_analyze.py --run {control} to refresh it. "
                      f"Falling back to this run's own residuals for sigma_out.")
            elif len(cmetrics["residual_std"]) != len(resid_std):
                print(f"\nwarning: control '{control}' has "
                      f"{len(cmetrics['residual_std'])} outputs, this run has "
                      f"{len(resid_std)} -- not comparable. "
                      f"Falling back to this run's own residuals for sigma_out.")
            else:
                if cmetrics.get("p_common") != 1.0:
                    print(f"\nwarning: control '{control}' was trained with "
                          f"p_common={cmetrics.get('p_common')}, not 1.0. "
                          f"sigma_out is only clean on a p_common=1 control "
                          f"(SS9.1) -- using it anyway, as requested.")
                sig_out = np.asarray(cmetrics["residual_std"])
                metrics["sigma_out_source"] = control
                print(f"\nsigma_out from control '{control}': "
                      + ", ".join(f"{s:.3f}" for s in sig_out))

    # --- 2. causal-inference analyses (causal head only) ------------------
    if cfg["model"]["head"] == "causal" and 0.0 < p_prior < 1.0:
        i_vis, i_prop = names.index("mu_vis"), names.index("mu_prop")
        dv = d["fused_mu"] - d["seg_vis_mu"]
        dp = d["fused_mu"] - d["seg_prop_mu"]

        # 2a. per-trial implied weight, sigma_w-filtered (SS7.1)
        sw_vis = analysis.sigma_w(sig_out[i_vis], dv)
        sw_prop = analysis.sigma_w(sig_out[i_prop], dp)
        crit = acfg.get("sigma_w_criterion", 0.1)
        w = analysis.fusion_weight(pred[:, i_vis], d["seg_vis_mu"], d["fused_mu"],
                                   acfg["min_separation"])
        w_prop = analysis.fusion_weight(pred[:, i_prop], d["seg_prop_mu"],
                                        d["fused_mu"], acfg["min_separation"])
        w_f = np.where(sw_vis < crit, w, np.nan)
        w_prop_f = np.where(sw_prop < crit, w_prop, np.nan)
        arrays.update(fusion_weight=w, fusion_weight_prop=w_prop,
                      sigma_w_vis=sw_vis, sigma_w_prop=sw_prop)
        metrics["implied_weight_vs_post"] = analysis.compare(d["post_c1"], w_f)
        metrics["implied_weight_vs_post_prop"] = analysis.compare(d["post_c1"], w_prop_f)

        # 2b. HEADLINE: position-domain regression, per output (SS7.1)
        metrics["position_regression_vis"] = analysis.position_regression(
            pred[:, i_vis], d["seg_vis_mu"], d["fused_mu"], d["post_c1"])
        metrics["position_regression_prop"] = analysis.position_regression(
            pred[:, i_prop], d["seg_prop_mu"], d["fused_mu"], d["post_c1"])
        pr_v, pr_p = metrics["position_regression_vis"], metrics["position_regression_prop"]
        print("\nposition-domain regression (Bayes-optimal: slope 1, intercept 0)")
        print(f"  mu_vis : slope {pr_v['slope']:.3f} "
              f"[{pr_v['slope_ci95'][0]:.3f}, {pr_v['slope_ci95'][1]:.3f}]"
              f"   intercept {pr_v['intercept']:+.3f}")
        print(f"  mu_prop: slope {pr_p['slope']:.3f} "
              f"[{pr_p['slope_ci95'][0]:.3f}, {pr_p['slope_ci95'][1]:.3f}]"
              f"   intercept {pr_p['intercept']:+.3f}")

        # 2c. joint two-output weight + hand-vs-visual consistency (SS7.1)
        w_joint = analysis.joint_fusion_weight(
            pred[:, i_vis], d["seg_vis_mu"], pred[:, i_prop], d["seg_prop_mu"],
            d["fused_mu"])
        arrays["fusion_weight_joint"] = w_joint
        metrics["implied_weight_joint_vs_post"] = analysis.compare(
            d["post_c1"], np.where(np.minimum(sw_vis, sw_prop) < crit, w_joint, np.nan))
        metrics["weight_consistency"] = analysis.weight_consistency(
            w, w_prop, sw_vis, sw_prop, criterion=crit)
        wc = metrics["weight_consistency"]
        print(f"hand-vs-visual w consistency (sigma_w < {crit}): "
              f"corr {wc['corr']:.3f}, mean|diff| {wc['mean_abs_diff']:.3f}, "
              f"n {wc['n']}")

        # 2d. Bayes vs disparity heuristic (SS7.2)
        metrics["reliability_within_disparity"] = analysis.reliability_within_disparity(
            np.where(sw_vis < crit, w, np.nan), d["post_c1"],
            np.abs(d["disparity"]))
        rwd = metrics["reliability_within_disparity"]
        print(f"within-disparity-bin slope of w_implied on w_opt: "
              f"{rwd['combined_slope']:.3f} +- {rwd['combined_se']:.3f} "
              f"(heuristic predicts 0, Bayes ~1)")

        # 2e. variance signature of causal ambiguity (SS7.3)
        metrics["variance_signature_vis"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in analysis.variance_signature(
                pred[:, names.index("var_vis")], d["post_c1"],
                d["fused_mu"], d["fused_var"], d["seg_vis_mu"], d["seg_vis_var"]
            ).items()}
        metrics["variance_signature_prop"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in analysis.variance_signature(
                pred[:, names.index("var_prop")], d["post_c1"],
                d["fused_mu"], d["fused_var"], d["seg_prop_mu"], d["seg_prop_var"]
            ).items()}
        hv = metrics["variance_signature_vis"].get("hump_net")
        ha = metrics["variance_signature_vis"].get("hump_analytical")
        if hv is not None:
            print(f"variance hump (mid-ambiguity elevation), var_vis: network "
                  f"{hv:+.2f}  analytical {ha:+.2f} deg^2")

        # 2f. binned five-way model comparison (SS7.4)
        metrics["model_comparison_vis"] = analysis.model_comparison(
            pred[:, i_vis], d["post_c1"], d["fused_mu"], d["seg_vis_mu"])
        metrics["model_comparison_prop"] = analysis.model_comparison(
            pred[:, i_prop], d["post_c1"], d["fused_mu"], d["seg_prop_mu"])
        mc = metrics["model_comparison_vis"]
        print("model comparison (overall RMSE):",
              {k: round(v, 3) for k, v in mc["overall"].items()},
              "-> best:", mc["best"])

        # 2g. transition curves (kept from before)
        midpoint, sharpness = analysis.transition_fit(np.abs(d["disparity"]), w_f)
        opt_mid, opt_sharp = analysis.transition_fit(np.abs(d["disparity"]),
                                                     d["post_c1"])
        metrics["transition"] = {
            "network": {"midpoint_deg": midpoint, "sharpness": sharpness},
            "analytical": {"midpoint_deg": opt_mid, "sharpness": opt_sharp},
        }

        # 2h. behavioral signatures (SS7.6)
        grid = acfg["disparity_grid"]
        bias = analysis.bias_vs_disparity(
            pred[:, i_prop], d["seg_prop_mu"], d["disparity"], grid,
            prediction=d["post_c1"] * dp)
        arrays["bias_centres"] = bias["centres"]
        arrays["bias_net"] = bias["bias_net"]
        if "bias_opt" in bias:
            arrays["bias_centres_opt"] = bias["centres_opt"]
            arrays["bias_opt"] = bias["bias_opt"]
        inferred_common = np.where(np.isfinite(w_joint), w_joint, d["post_c1"]) > 0.5
        cb = analysis.conditioned_bias(
            pred[:, i_prop], d["seg_prop_mu"], d["disparity"],
            inferred_common, grid)
        metrics["conditioned_bias_negative_seen"] = cb["negative_bias_seen"]
        for label in ("common", "separate"):
            arrays[f"cond_bias_{label}_centres"] = cb[label]["centres"]
            arrays[f"cond_bias_{label}"] = cb[label]["bias"]

        # 2i. decision strategy (kept for continuity)
        metrics["strategy"] = analysis.strategy_fit(
            pred[:, i_vis], d["post_c1"], d["fused_mu"], d["seg_vis_mu"])

    # --- 3. what do the hidden layers carry? ------------------------------
    acts = hidden_activations(model, d["X"])
    pc_decoding = analysis.decode_by_layer(
        acts, d["post_c1"], acfg["ridge_alpha"], acfg["decoder_test_size"])
    metrics["post_c1_decoding_r2"] = analysis.summarise(pc_decoding)
    print("\np(C=1|x) decodable from:", _round(metrics["post_c1_decoding_r2"]))

    estimate_decoding = analysis.decode(
        acts["msl"], d["mu_vis"], acfg["ridge_alpha"], acfg["decoder_test_size"])
    metrics["mu_vis_decoding_r2"] = estimate_decoding["r2"]

    # --- 4. unit-level analyses of the MSL (SS7.5) ------------------------
    cong = analysis.congruency(model, d_full["encoders"], cfg)
    metrics["congruency"] = {k: cong[k] for k in
                             ("n_congruent", "n_opposite", "n_mixed", "n_untuned")}
    arrays["congruency_index"] = cong["index"]
    print("MSL congruency:", metrics["congruency"])

    bal = analysis.balance(acts["msl"], cong["classes"], d["post_c1"])
    metrics["balance_vs_post"] = {k: bal[k] for k in
                                  ("n_congruent", "n_opposite", "corr", "slope", "r2")}
    if "balance" in bal:
        arrays["congruent_opposite_balance"] = bal["balance"]
    print(f"congruent-opposite balance vs p(C=1|x): corr "
          f"{metrics['balance_vs_post']['corr']:.3f}  "
          f"r2 {metrics['balance_vs_post']['r2']:.3f}")

    if cong["n_congruent"] > 0 and cong["n_opposite"] > 0:
        # Mean-clamp ablation against a size-matched random baseline. Both parts
        # matter: zeroing a sigmoid unit injects a perturbation rather than
        # removing information, and without the baseline the damage number just
        # reflects how many units were removed.
        metrics["lesion"] = analysis.lesion_comparison(
            model, d["X"], cong["classes"], target,
            mode="mean", n_random=acfg.get("lesion_n_random", 100))
        print("MSL lesion (mean-clamp, vs size-matched random baseline):")
        for lab in ("no_congruent", "no_opposite", "no_mixed"):
            r = metrics["lesion"].get(lab, {})
            if "rmse" not in r:
                continue
            print(f"  {lab:14s} k={r['n_units']:2d}  rmse {r['rmse']:6.3f}  "
                  f"random {r['null_mean']:6.3f}+-{r['null_sd']:.3f}  "
                  f"z={r['z']:+5.2f}")

    shifts = analysis.rf_shift(model, d_full["encoders"], cfg)
    metrics["rf_shift_median_gain"] = shifts["median_shift_gain"]
    arrays["rf_shift_gain"] = shifts["shift_gain"]
    arrays["rf_gain_field"] = shifts["gain_field"]
    print(f"median RF shift gain (0 = spatial code, +1 = retinal): "
          f"{shifts['median_shift_gain']:.2f}")

    # --- 5. emergent vs imposed (optional) --------------------------------
    if twin:
        twin_model, twin_cfg, _, twin_splits = load_checkpoint(run_dir(twin) / "model.pt")
        twin_acts = hidden_activations(twin_model, d["X"])
        twin_decoding = analysis.decode_by_layer(
            twin_acts, d["post_c1"], acfg["ridge_alpha"], acfg["decoder_test_size"])
        metrics["twin_post_c1_decoding_r2"] = analysis.summarise(twin_decoding)
        print(f"p(C=1|x) decodable from the '{twin}' twin:",
              _round(metrics["twin_post_c1_decoding_r2"]))

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
    p.add_argument("--control", default=None,
                   help="run name of a p_common=1 control; its residual_std "
                        "becomes sigma_out for the sigma_w filter (SS7.1)")
    args = p.parse_args()
    analyse(args.run, args.twin, args.control)
