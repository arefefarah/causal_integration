"""SS9.3: changing only p_common with the same seed must leave all other latents
identical (uncontaminated cross-prior comparison)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmsi.data.generative import sample_trials
from cmsi.utils.config import load_config

cfg = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"))
g1 = dict(cfg["generative"], p_common=0.5)
g2 = dict(cfg["generative"], p_common=0.7)
a = sample_trials(20000, g1, np.random.default_rng(0))
b = sample_trials(20000, g2, np.random.default_rng(0))
same_C = a["C"] == b["C"]
print("trials with same C:", same_C.mean())
for k in ["eye","sig2_vis","sig2_prop","sig2_eye"]:
    print(f"  {k} identical across priors: {np.array_equal(a[k], b[k])}")
for k in ["s_vis","s_prop","x_vis","x_prop","x_eye"]:
    print(f"  {k} identical on same-C trials: {np.array_equal(a[k][same_C], b[k][same_C])}")
