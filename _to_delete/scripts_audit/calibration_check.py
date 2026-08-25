"""Design SS8 calibration + SS3 containment + SS9.4 audit stats, per config."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmsi.data.generative import sample_trials, observer
from cmsi.utils.config import load_config

for name in ["default", "realistic", "matlab_match", "equal_n"]:
    cfg = load_config(str(Path(__file__).resolve().parents[2] / "configs" / f"{name}.yaml"))
    gen, enc = cfg["generative"], cfg["encoding"]
    rng = np.random.default_rng(0)
    n = 40000
    d = sample_trials(n, gen, rng)
    o = observer(d, gen)
    p = o["p_common"]
    inter = ((p > 0.2) & (p < 0.8)).mean()
    conf = ((p < 0.05) | (p > 0.95)).mean()
    fracC1 = (d["C"] == 1).mean()
    # containment: retinal measurement vs encoded span with 2*rf_width margin
    lo, hi = enc["visual_field"]; m = 2 * enc["rf_width"]
    out_vis = ((d["x_vis"] < lo + m) | (d["x_vis"] > hi - m)).mean()
    out_vis_hard = ((d["x_vis"] < lo) | (d["x_vis"] > hi)).mean()
    # Delta per output
    dv = np.abs(o["fused_mu"] - o["seg_vis_mu"]); dp = np.abs(o["fused_mu"] - o["seg_prop_mu"])
    # conditional distributions given C (audit 9.4)
    c1, c2 = d["C"]==1, d["C"]==2
    print(f"--- {name} ---")
    print(f"  frac C=1: {fracC1:.3f}   posterior mass 0.2-0.8: {inter:.3f}   mass <0.05 or >0.95: {conf:.3f}")
    print(f"  x_vis outside field: {out_vis_hard*100:.2f}%   outside field-2rf margin: {out_vis*100:.2f}%")
    print(f"  |Delta| vis-output: median {np.median(dv):.2f}  >2deg: {(dv>2).mean():.2f}  >5deg: {(dv>5).mean():.2f}")
    print(f"  |Delta| prop-output: median {np.median(dp):.2f}  >2deg: {(dp>2).mean():.2f}  >5deg: {(dp>5).mean():.2f}")
    print(f"  eye|C=1 mean/sd {d['eye'][c1].mean():+.2f}/{d['eye'][c1].std():.2f}  eye|C=2 {d['eye'][c2].mean():+.2f}/{d['eye'][c2].std():.2f}")
    print(f"  sig2_vis|C=1 {d['sig2_vis'][c1].mean():.2f}  |C=2 {d['sig2_vis'][c2].mean():.2f}")
