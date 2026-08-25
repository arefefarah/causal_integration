"""Design SS4 Poisson validity check: at the largest sigma (smallest gain),
ML-decode unimodal visual trials; empirical decoder variance vs nominal sig2."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmsi.data.encoding import make_encoders, gaussian_code
from cmsi.utils.config import load_config

for name in ["default", "realistic", "matlab_match"]:
    cfg = load_config(str(Path(__file__).resolve().parents[2] / "configs" / f"{name}.yaml"))
    enc = cfg["encoding"]
    encs = make_encoders(enc, np.random.default_rng(cfg["seed"]))
    centers, width, K = encs["rf_centers"], enc["rf_width"], enc["gain_K"]
    rng = np.random.default_rng(1)
    grid = np.linspace(centers[0], centers[-1], 1601)
    F = np.exp(-0.5*((grid[:,None]-centers[None,:])/width)**2)  # (grid, units)
    print(f"--- {name} (K={K}) ---")
    for sig2 in [min(cfg['generative']['sigma2_vis_range']), max(cfg['generative']['sigma2_vis_range'])]:
        gain = K/sig2
        ntr = 4000
        # true retinal position fixed at 0 -> measurement x ~ N(0, sig2), then encode x
        x = rng.normal(0.0, np.sqrt(sig2), ntr)
        rates = gaussian_code(x, centers, width, np.full(ntr, gain))
        counts = rng.poisson(rates)
        # ML decode of x from counts (Poisson log-lik on grid)
        logF = np.log(np.clip(gain*F, 1e-12, None))    # (grid, units)
        sumF = (gain*F).sum(1)                          # (grid,)
        ll = counts @ logF.T - sumF[None,:]             # (ntr, grid)
        xhat = grid[ll.argmax(1)]
        dec_var = np.var(xhat - x)     # decoder noise around the true measurement
        tot_var = np.var(xhat)         # total = sig2 + decoder noise (approx)
        spikes = counts.sum(1).mean()
        print(f"  sig2={sig2:5.1f}  gain={gain:6.2f}  spikes/trial={spikes:7.1f}   "
          f"decode-noise var={dec_var:6.2f}  total var={tot_var:6.2f}  nominal={sig2:5.1f}  "
          f"inflation={(tot_var/sig2-1)*100:+.0f}%")
