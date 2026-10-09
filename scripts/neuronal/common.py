"""Shared pieces for the neuronal-level analyses (scripts/neuronal/run.py).

Everything here is torch-free (see npmodel.py). Function names follow the
analysis module they mirror, so a reader of analysis/units.py recognises
`tuning_curves_np` and `congruency_np` as the same computations on NpNet.
"""

import json
from pathlib import Path

import numpy as np

from npmodel import ROOT, NpNet, load_run, stub_cmsi  # noqa: F401

stub_cmsi()
from cmsi.analysis.causal import (binned_weight, hybrid_weight,     # noqa: E402
                                  mean_by_bin, position_regression,
                                  transition_fit, variance_signature)
from cmsi.analysis.decoding import decode                            # noqa: E402
from cmsi.data.encoding import encode                                # noqa: E402
from cmsi.viz.manuscript import (CELL_SQUARE, CELL_WIDE, FIG_BIAS,    # noqa: E402,F401
                                 FIG_LESION, FIG_UNITS, PANEL_FONT,   # (FIG_*: paper names)
                                 PANEL_LW, PANEL_MS, finish_panel,
                                 manuscript_dir, manuscript_grid, panel_axes)
from cmsi.viz.style import COLORS, apply_style, save_figures         # noqa: E402,F401



def out_dir(run="flagship"):
    """Where the neuronal-level analyses of `run` go: beside the run's other
    model figures, results/<run>/figures/model/neuronal_level_analysis/, one
    folder per figure (composed figure named as its folder, panels, numbers)."""
    return ROOT / "results" / run / "figures" / "model" / "neuronal_level_analysis"


CLASS_COLORS = {"congruent": "#1f77b4", "opposite": "#d62728", "mixed": "#ccbb44"}  # mixed: rgb(204,187,68), never the grey of the random bands or the intact bars
CLASSES = ("congruent", "opposite", "mixed")
OUTPUT_NAMES = ("mu_vis", "var_vis", "mu_prop", "var_prop")
OUTPUT_LABELS = ("μ vis", "σ² vis", "μ hand", "σ² hand")
RUN_PRIOR = {"pcommon0": 0.0, "pcommon03": 0.3, "flagship": 0.5, "pcommon07": 0.7, "pcommon1": 1.0}
TWIN_RUNS = ("pcommon03", "flagship", "pcommon07")      # twins exist for these worlds
RUN_LABEL = {"pcommon0": "always-segregate control", "pcommon03": "p = 0.3", "flagship": "main network",
             "pcommon07": "p = 0.7", "pcommon1": "always-fuse control"}


def mid_sigmas(gen):
    return {"vis": float(np.mean(gen["sigma2_vis_range"])),
            "prop": float(np.mean(gen["sigma2_prop_range"])),
            "eye": float(np.mean(gen["sigma2_eye_range"]))}


def sweep_dict(x_vis, x_prop, x_eye, sig2):
    n = max(np.size(x_vis), np.size(x_prop), np.size(x_eye))

    def full(v):
        return np.full(n, v, float) if np.size(v) == 1 else np.asarray(v, float)

    return {"x_vis": full(x_vis), "x_prop": full(x_prop), "x_eye": full(x_eye),
            "sig2_vis": np.full(n, sig2["vis"]), "sig2_prop": np.full(n, sig2["prop"]),
            "sig2_eye": np.full(n, sig2["eye"])}


def clean_msl(net, encoders, cfg, x_vis, x_prop, x_eye=0.0):
    """MSL activations to noiseless rates at mid-range reliability, eye at 0
    unless given: the stimulus set behind units.tuning_curves."""
    gen, enc = cfg["generative"], cfg["encoding"]
    d = sweep_dict(x_vis, x_prop, x_eye, mid_sigmas(gen))
    X = encode(d, encoders, dict(enc, poisson_noise=False), np.random.default_rng(0))
    return net.hidden(X)[-1]


def tuning_curves_np(net, encoders, cfg, positions=None):
    gen = cfg["generative"]
    if positions is None:
        span = 2.0 * np.sqrt(gen["sigma0_sq"])
        positions = np.linspace(-span, span, 41)
    resp_vis = clean_msl(net, encoders, cfg, positions, 0.0)
    resp_prop = clean_msl(net, encoders, cfg, np.zeros(len(positions)), positions)
    return np.asarray(positions), resp_vis, resp_prop


def congruency_np(net, encoders, cfg, positions=None, threshold=0.5):
    """units.congruency on an NpNet -> the same dict."""
    positions, rv, rp = tuning_curves_np(net, encoders, cfg, positions)
    n_units = rv.shape[1]
    index = np.zeros(n_units)
    tuned = np.zeros(n_units, bool)
    for u in range(n_units):
        tuned[u] = (rv[:, u].std() > 1e-4) and (rp[:, u].std() > 1e-4)
        if tuned[u]:
            index[u] = np.corrcoef(rv[:, u], rp[:, u])[0, 1]
    classes = np.where(~tuned, "untuned",
                       np.where(index > threshold, "congruent",
                                np.where(index < -threshold, "opposite", "mixed")))
    return {"index": index, "classes": classes, "tuned": tuned,
            "n_congruent": int((classes == "congruent").sum()),
            "n_opposite": int((classes == "opposite").sum()),
            "n_mixed": int((classes == "mixed").sum()),
            "n_untuned": int((classes == "untuned").sum()),
            "positions": positions, "resp_vis": rv, "resp_prop": rp}


class Bundle:
    """One run, loaded once: net, config, test split, encoders, metrics,
    sigma_out, unit classes (checked against metrics.json)."""

    def __init__(self, run):
        self.run = run
        self.net, self.ckpt, self.cfg, self.d, self.d_full = load_run(run)
        self.metrics = json.loads((ROOT / "results" / run / "metrics.json").read_text())
        self.names = list(self.d_full["target_names"])
        self.sig_out = np.asarray(self.metrics.get("sigma_out",
                                                   self.metrics["residual_std"]))
        self.cong = congruency_np(self.net, self.d_full["encoders"], self.cfg)
        self.classes = self.cong["classes"]
        m = self.metrics.get("congruency", {})
        mine = {k: self.cong[k] for k in ("n_congruent", "n_opposite", "n_mixed", "n_untuned")}
        self.counts_match = all(m.get(k) == v for k, v in mine.items()) if m else None
        self.pred = self.net.predict(self.d["X"])
        self.msl = self.net.hidden(self.d["X"])[-1]
        self.p_common = float(self.cfg["generative"]["p_common"])

    @property
    def counts(self):
        return {k: self.cong[k] for k in ("n_congruent", "n_opposite", "n_mixed", "n_untuned")}

    def idx(self, name):
        return self.names.index(name)


def hybrid_reads(b, pred):
    """The visual and hand implied weights, exactly as 03_analyze computes them."""
    d, n = b.d, b.names
    i_vis, i_prop = n.index("mu_vis"), n.index("mu_prop")
    i_vv, i_vp = n.index("var_vis"), n.index("var_prop")
    dv = d["fused_mu"] - d["seg_vis_mu"]
    dp = d["fused_mu"] - d["seg_prop_mu"]
    w, _, _ = hybrid_weight(pred[:, i_vv], d["fused_var"], d["seg_vis_var"], dv,
                            pred[:, i_vis], d["seg_vis_mu"], b.sig_out[i_vis], b.sig_out[i_vv])
    wp, _, _ = hybrid_weight(pred[:, i_vp], d["fused_var"], d["seg_prop_var"], dp,
                             pred[:, i_prop], d["seg_prop_mu"], b.sig_out[i_prop], b.sig_out[i_vp])
    return w, wp


def behaviour_stats(b, pred, msl, decode_kw=None):
    """The four behavioural statistics of a (possibly lesioned) network on the
    test split: transition midpoint, position-regression slope (vis, hand),
    variance hump (vis, hand), posterior decoding from the MSL. Plus RMSE."""
    d, n = b.d, b.names
    acfg = b.cfg["analysis"]
    decode_kw = decode_kw or {}
    post = d["post_c1"]
    target = np.stack([d[k] for k in n], axis=1)
    w, wp = hybrid_reads(b, pred)
    mid, sharp = transition_fit(np.abs(d["disparity"]), w)
    pr_v = position_regression(pred[:, n.index("mu_vis")], d["seg_vis_mu"], d["fused_mu"], post)
    pr_p = position_regression(pred[:, n.index("mu_prop")], d["seg_prop_mu"], d["fused_mu"], post)
    vs_v = variance_signature(pred[:, n.index("var_vis")], post, d["fused_mu"], d["fused_var"],
                              d["seg_vis_mu"], d["seg_vis_var"])
    vs_p = variance_signature(pred[:, n.index("var_prop")], post, d["fused_mu"], d["fused_var"],
                              d["seg_prop_mu"], d["seg_prop_var"])
    dec = decode(msl, post, acfg.get("ridge_alpha", 1.0), acfg.get("decoder_test_size", 0.25),
                 **decode_kw)
    return {"rmse": float(np.sqrt(((pred - target) ** 2).mean())),
            "midpoint": float(mid), "sharpness": float(sharp),
            "posreg_vis": float(pr_v["slope"]), "posreg_prop": float(pr_p["slope"]),
            "hump_vis": float(vs_v["hump_net"]), "hump_prop": float(vs_p["hump_net"]),
            "decode_msl": float(dec["r2"]),
            "w": w, "wp": wp}

def fusion_segregation(abs_disparity, w, near=2.0, far=15.0):
    """Two robust summaries of the weight curve: the mean weight where the
    cues agree (|d| <= near, fusion) and one minus the mean weight where they
    clearly disagree (|d| >= far, segregation). Both are 1 for a perfect
    observer on this task's scale and need no curve fit."""
    ad = np.abs(abs_disparity)
    ok = np.isfinite(w)
    return (float(np.nanmean(w[ok & (ad <= near)])),
            float(1.0 - np.nanmean(w[ok & (ad >= far)])))


STAT_LABELS = {"rmse": "RMSE", "midpoint": "midpoint",
               "posreg_vis": "slope vis", "posreg_prop": "slope hand",
               "hump_vis": "hump vis", "hump_prop": "hump hand",
               "decode_msl": "decoding R²"}


def bootstrap_ci(values, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if len(v) == 0:
        return (np.nan, np.nan)
    draws = np.array([stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n)])
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def save_json(obj, path):
    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=enc))
