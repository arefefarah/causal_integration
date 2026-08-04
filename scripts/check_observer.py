"""Check this observer against the original src/causal_msi implementation.

    python scripts/check_observer.py

Runs both on identical draws and prints the largest disagreement. Worth running
after any edit to data/generative.py or data/encoding.py -- the equations are the
part of this project that must not drift.
"""

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

REPO = Path(__file__).resolve().parents[2]      # the outer causal_integration repo
sys.path.insert(0, str(REPO / "src"))

from cmsi.data import encoding as new_enc      # noqa: E402
from cmsi.data import generative as new        # noqa: E402
from cmsi.utils import load_config             # noqa: E402

try:
    from causal_msi import encoding as old_enc         # noqa: E402
    from causal_msi import generative as old           # noqa: E402
    from causal_msi.config import load_config as load_old_config   # noqa: E402
except ImportError as exc:                              # pragma: no cover
    raise SystemExit(f"the original package is not importable ({exc}); "
                     "install pydantic/pyyaml or skip this check")

N = 5000
TOLERANCE = 1e-10


def main():
    cfg = load_config()
    old_cfg = load_old_config(REPO / "configs" / "default.yaml")

    rng = np.random.default_rng(0)
    d = new.sample_trials(N, cfg["generative"], rng)
    d.update(new.observer(d, cfg["generative"]))

    latents = old.LatentBatch(
        C=d["C"].astype(np.int64), s_vis=d["s_vis"], s_prop=d["s_prop"],
        eye=d["eye"], retinal_source=d["retinal"], sigma2_vis=d["sig2_vis"],
        sigma2_prop=d["sig2_prop"], sigma2_eye=d["sig2_eye"],
    )
    meas = old.Measurements(x_vis=d["x_vis"], x_eye=d["x_eye"], x_prop=d["x_prop"])
    t = old.analytical_observer(meas, latents, old_cfg.generative)

    checks = {
        "mu_vis": (d["mu_vis"], t.mu_vis), "var_vis": (d["var_vis"], t.var_vis),
        "mu_prop": (d["mu_prop"], t.mu_prop), "var_prop": (d["var_prop"], t.var_prop),
        "p_common": (d["p_common"], t.p_common), "log_bf": (d["log_bf"], t.log_bf),
        "fused_mu": (d["fused_mu"], t.fused_mu), "fused_var": (d["fused_var"], t.fused_var),
        "seg_vis_mu": (d["seg_vis_mu"], t.seg_vis_mu),
        "seg_prop_mu": (d["seg_prop_mu"], t.seg_prop_mu),
    }

    encoders_new = new_enc.make_encoders(cfg["encoding"], np.random.default_rng(cfg["seed"]))
    encoders_old = old_enc.Encoders.build(np.random.default_rng(cfg["seed"]),
                                          old_cfg.encoding)
    checks["rf_centers"] = (encoders_new["rf_centers"], encoders_old.rf_centers)
    checks["prop_slope"] = (encoders_new["prop_slope"], encoders_old.prop_hand.slopes)
    checks["eye_intercept"] = (encoders_new["eye_intercept"], encoders_old.prop_eye.intercepts)
    checks["X"] = (
        new_enc.encode(d, encoders_new, cfg["encoding"], np.random.default_rng(7)),
        old_enc.assemble_inputs(np.random.default_rng(7), meas, latents,
                                old_cfg.encoding, encoders=encoders_old),
    )

    worst = 0.0
    for name, (a, b) in checks.items():
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        worst = max(worst, diff)
        print(f"{name:>14}  max|new - old| = {diff:.3e}")

    print(f"\nworst difference: {worst:.3e}")
    if worst >= TOLERANCE:
        raise SystemExit("MISMATCH -- the port has drifted from the original")
    print("OK -- numerically identical to src/causal_msi")


if __name__ == "__main__":
    main()
