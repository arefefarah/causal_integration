"""Pre-training dataset calibration (design SS8) and audits (SS9.4, SS4).

Run BEFORE training, on every new config:

    stats, arrays = calibrate(cfg, n=20000)
    for level, msg in checks(stats, cfg):  print(level, msg)

The design's criteria, as implemented here:

  SS8.1  posterior histogram: ~25% of trials at intermediate posteriors
         (0.2-0.8) and substantial mass at confident ones. Bimodal-at-{0,1}
         -> shrink sigma0_sq; all-intermediate -> widen sigma0_sq or tighten
         the sensory-noise ranges.
  SS8.2  |Delta| histograms per output: enough trials with |Delta| large
         enough for SS7.1 to have usable trials.
  SS8.3  reliability coverage: within fixed-disparity bins the posterior must
         still vary (else SS7.2 is powerless -- widen the sigma ranges).
  SS3    containment: rejection rate below a few percent.
  SS4    Poisson validity: ML-decoding the visual population must return the
         nominal sigma2, or the targets are miscalibrated (raise gain_K).
  SS9.4  anti-confound: C must not be predictable from the eye signal or from
         any single modality beyond chance, and the eye/sigma distributions
         must match across C.
"""

import numpy as np

from cmsi.data.dataset import make_dataset
from cmsi.data.encoding import gaussian_code, make_encoders


def calibrate(cfg, n=20000, seed=None):
    """All SS8/SS9.4/SS4 statistics for one config. Returns (stats, arrays)."""
    d = make_dataset(cfg, n=n, seed=seed)
    p = d["post_c1"]
    dv = d["fused_mu"] - d["seg_vis_mu"]
    dp = d["fused_mu"] - d["seg_prop_mu"]
    c1 = d["C"] == 1

    stats = {
        "n": int(n),
        "p_common": float(cfg["generative"]["p_common"]),
        "realized_c1_fraction": float(c1.mean()),
        "rejection_rate": float(np.asarray(d["rejection_rate"]).ravel()[0]),
        # SS8.1
        "posterior_mass_intermediate": float(((p > 0.2) & (p < 0.8)).mean()),
        "posterior_mass_confident": float(((p < 0.05) | (p > 0.95)).mean()),
        # SS8.2
        "abs_delta_vis_median": float(np.median(np.abs(dv))),
        "abs_delta_prop_median": float(np.median(np.abs(dp))),
        "frac_delta_vis_gt2": float((np.abs(dv) > 2).mean()),
        "frac_delta_prop_gt2": float((np.abs(dp) > 2).mean()),
        # SS8.3
        "reliability_coverage": _reliability_coverage(p, d["disparity"]),
        # SS9.4
        "anticonfound": _anticonfound(d),
        # SS4
        "poisson": _poisson_validity(cfg),
    }
    arrays = {"post_c1": p, "delta_vis": dv, "delta_prop": dp,
              "disparity": d["disparity"], "C": d["C"],
              "sig2_vis": d["sig2_vis"]}
    return stats, arrays


def _reliability_coverage(post, disparity, n_bins=8):
    """Within-|disparity|-bin spread of the optimal weight (SS8.3).

    Reports BOTH the std and the peak-to-peak range per bin. The range is the
    quantity that matters: `causal.reliability_within_disparity` regresses
    w_implied on w_optimal within each bin, and its leverage comes from how far
    apart the extreme w_optimal values in the bin are. The floor in `checks`
    was calibrated by simulating a Bayes-optimal network with realistic readout
    noise -- a mean range of ~0.5 recovers slope 1.00 +- 0.003, so the test
    stays powered well below that; the std of the same bins is only ~0.06,
    which is why a std-based floor would reject perfectly usable designs.
    """
    ad = np.abs(disparity)
    edges = np.quantile(ad, np.linspace(0, 1, n_bins + 1))
    spreads, ranges = [], []
    for b in range(n_bins):
        m = (ad >= edges[b]) & (ad < edges[b + 1]) if b < n_bins - 1 \
            else (ad >= edges[b])
        if m.sum() > 20:
            spreads.append(float(post[m].std()))
            ranges.append(float(np.ptp(post[m])))
    return {"within_bin_posterior_std": spreads,
            "within_bin_posterior_range": ranges,
            "mean": float(np.mean(spreads)) if spreads else np.nan,
            "mean_range": float(np.mean(ranges)) if ranges else np.nan}


def _auc(score, label):
    """Rank AUC of `score` for predicting the boolean `label`."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _anticonfound(d):
    """SS9.4: C must not leak through side channels.

    AUCs of single variables for predicting C=1 (must be ~0.5: |AUC-0.5|
    small), and the moment match of eye and the sigmas across C.
    """
    c1 = d["C"] == 1
    if c1.all() or not c1.any():
        return {"skipped": "single-C dataset"}
    aucs = {k: _auc(np.abs(np.asarray(d[k], float)), c1)
            for k in ("x_eye", "x_vis", "x_prop", "sig2_vis", "sig2_prop",
                      "sig2_eye")}
    moments = {k: {"mean_c1": float(d[k][c1].mean()),
                   "mean_c2": float(d[k][~c1].mean()),
                   "std_c1": float(d[k][c1].std()),
                   "std_c2": float(d[k][~c1].std())}
               for k in ("eye", "sig2_vis", "sig2_prop", "sig2_eye")}
    return {"auc_abs": aucs, "moments": moments,
            "max_auc_deviation": float(max(abs(v - 0.5) for v in aucs.values()
                                           if np.isfinite(v)))}


def _poisson_validity(cfg, n_trials=6000, grid_points=1201):
    """SS4: decode the visual population at both ends of the sigma range and
    compare the empirical estimator variance to the nominal sigma2.

    The network effectively sees the measurement x through the spike code, so
    its true uncertainty about the source is sigma2 + Var(x_hat - x). The
    reported `inflation` is that PAIRED ratio Var(x_hat - x) / sigma2 -- the
    quantity the targets get wrong if it is not ~0. (`inflation_unpaired`
    compares Var(x_hat) to sigma2 directly; it estimates the same thing but
    carries the sampling noise of an unpaired variance, ~sqrt(2/n), which at
    n = 6000 is already +-1.8% on its own.)
    """
    gen, enc = cfg["generative"], cfg["encoding"]
    encoders = make_encoders(enc, np.random.default_rng(cfg["seed"]))
    centers, width, K = encoders["rf_centers"], enc["rf_width"], enc["gain_K"]
    rng = np.random.default_rng(12345)
    grid = np.linspace(centers[0], centers[-1], grid_points)
    F = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / width) ** 2)

    out = {}
    for tag, sig2 in (("min", min(gen["sigma2_vis_range"])),
                      ("max", max(gen["sigma2_vis_range"]))):
        gain = K / sig2
        x = rng.normal(0.0, np.sqrt(sig2), n_trials)
        counts = rng.poisson(gaussian_code(x, centers, width,
                                           np.full(n_trials, gain)))
        logF = np.log(np.clip(gain * F, 1e-12, None))
        ll = counts @ logF.T - (gain * F).sum(1)[None, :]
        xhat = grid[ll.argmax(1)]
        out[tag] = {"sigma2": float(sig2),
                    "spikes_per_trial": float(counts.sum(1).mean()),
                    "decode_noise_var": float(np.var(xhat - x)),
                    "total_var": float(np.var(xhat)),
                    "inflation": float(np.var(xhat - x) / sig2),
                    "inflation_unpaired": float(np.var(xhat) / sig2 - 1)}
    out["worst_inflation"] = max(out["min"]["inflation"], out["max"]["inflation"])
    return out


def checks(stats, cfg):
    """The design's pass/warn/fail rules, as (level, message) tuples."""
    msgs = []
    p_prior = stats["p_common"]

    if 0 < p_prior < 1:
        drift = abs(stats["realized_c1_fraction"] - p_prior)
        msgs.append(("PASS" if drift < 0.02 else "WARN",
                     f"realized C=1 fraction {stats['realized_c1_fraction']:.3f} "
                     f"vs p_common {p_prior} (SS3)"))

        mi = stats["posterior_mass_intermediate"]
        mc = stats["posterior_mass_confident"]
        level = "PASS" if 0.15 <= mi <= 0.40 and mc >= 0.15 else "WARN"
        hint = ""
        if mi > 0.40:
            hint = " -- all-intermediate: widen sigma0_sq or tighten sensory noise"
        elif mi < 0.15:
            hint = " -- bimodal at {0,1}: shrink sigma0_sq"
        msgs.append((level, f"posterior mass: {mi:.2f} intermediate (target ~0.25), "
                            f"{mc:.2f} confident (SS8.1){hint}"))

        cov = stats["reliability_coverage"]["mean_range"]
        msgs.append(("PASS" if cov > 0.15 else "WARN",
                     f"within-disparity-bin posterior range {cov:.2f} "
                     f"(std {stats['reliability_coverage']['mean']:.3f}) "
                     f"(SS8.3; < 0.15 leaves SS7.2 powerless)"))

        for tag in ("vis", "prop"):
            f2 = stats[f"frac_delta_{tag}_gt2"]
            msgs.append(("PASS" if f2 >= 0.25 else "WARN",
                         f"|Delta_{tag}| > 2 deg on {f2:.0%} of trials (SS8.2)"))

        mad = stats["anticonfound"].get("max_auc_deviation", np.nan)
        msgs.append(("PASS" if mad < 0.05 else "FAIL",
                     f"anti-confound: worst single-channel AUC deviation "
                     f"{mad:.3f} from 0.5 (SS9.4)"))

    rr = stats["rejection_rate"]
    msgs.append(("PASS" if rr < 0.03 else ("WARN" if rr < 0.10 else "FAIL"),
                 f"containment rejection rate {rr:.1%} (SS3; keep < a few %)"))

    infl = stats["poisson"]["worst_inflation"]
    msgs.append(("PASS" if infl < 0.05 else ("WARN" if infl < 0.20 else "FAIL"),
                 f"Poisson validity: decoded variance exceeds nominal sigma2 by "
                 f"{infl:.0%} at worst (SS4; raise gain_K if > a few %)"))
    return msgs
