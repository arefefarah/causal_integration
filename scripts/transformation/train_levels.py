"""Train the networks of the transformation-to-causality investigation and SAVE
them as runs (the repo's 07_eye_sweep.py trains the same way but keeps no
checkpoints).

    python scripts/transformation/train_levels.py --param eye_noise --levels 0.3 1 2 4 8 1000 --seeds 0 1 2 --twin
    python scripts/transformation/train_levels.py --param eye_sd    --levels 1 3 9 18 27     --seeds 0 1 2 --twin
    python scripts/transformation/train_levels.py --param eye_wide  --levels 16             --seeds 0 1 2 --twin
    python scripts/transformation/train_levels.py --param eye_ecc   --levels 0.01 0.03 0.1  --seeds 0 1 2 --twin

Levers (one generative change to configs/flagship.yaml each):
    eye_noise  the eye signal's noise sd (deg): sigma2_eye_range centred on sd^2 with the
               flagship's +-6 % width.  sd 0.3 = the near-perfect eye signal (R2),
               sd 1000 = an uninformative one (R3; the eye population fires nothing and
               the observer uses the prior).
    eye_sd     the eye position's spread (deg): eye_sigma_sq = sd^2 (as 07_eye_sweep.py --param eye_sd)
    eye_wide   a wide per-trial eye-noise range inside one network: sigma2_eye_range [1, level]
    eye_ecc    eccentricity-dependent eye noise: sig2_eye_trial = sig2_eye + c * eye^2, with the
               observer still exact (the dataset is rebuilt here, src/ is untouched)

Writes results/transformation_investigation/runs/<param>_<level>_s<seed>[_twin]/
    model.pt, config.yaml, dataset.txt           (what 02_train.py writes)
and data/transformation_investigation/<param>_<level>_s<seed>.npz (slim: X float32, no X_clean;
the twin's dataset.txt points at the causal dataset -- same trials, same encoders),
and appends to results/transformation_investigation/runs/manifest.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import DATA_INV, ROOT, RUNS, slim_dataset  # noqa: E402

from cmsi.data.dataset import TARGETS, make_dataset  # noqa: E402
from cmsi.data.encoding import encode  # noqa: E402
from cmsi.data.generative import observer  # noqa: E402
from cmsi.utils.config import load_config, save_config, tweak  # noqa: E402
from cmsi.utils.io import save_checkpoint, save_dataset  # noqa: E402

# torch (and the training loop) are imported only when training is asked for,
# so inv_common can import make_dataset_ecc from here on a machine without torch.

WIDTH = 4.2 / 3.7          # the flagship's sigma2_eye_range ratio


def level_config(cfg, param, level, seed, head="causal"):
    if param == "eye_noise":
        s2 = float(level) ** 2
        c = tweak(cfg, sigma2_eye_range=[float(s2 / np.sqrt(WIDTH)), float(s2 * np.sqrt(WIDTH))])
    elif param == "eye_sd":
        c = tweak(cfg, eye_sigma_sq=float(level) ** 2)
    elif param == "eye_wide":
        c = tweak(cfg, sigma2_eye_range=[1.0, float(level)])
    elif param == "eye_ecc":
        c = dict(cfg)      # the lever is applied in make_dataset_ecc
    else:
        raise KeyError(param)
    return tweak(c, seed=int(seed), head=head)


def make_dataset_ecc(cfg, c_ecc):
    """The flagship dataset with eccentricity-dependent eye noise: after the
    repo's sampler, the per-trial eye noise becomes sig2_eye + c * eye^2, the
    eye measurement is redrawn with it, the observer is re-run (it uses the
    per-trial sig2_eye, so it stays exact) and the input is re-encoded (the
    eye population's gain K / sig2_eye now falls with eccentricity)."""
    d = make_dataset(cfg)
    rng = np.random.default_rng(int(cfg["seed"]) + 7919)
    d["sig2_eye_base"] = d["sig2_eye"].copy()
    d["sig2_eye"] = d["sig2_eye"] + float(c_ecc) * d["eye"] ** 2
    d["x_eye"] = rng.normal(d["eye"], np.sqrt(d["sig2_eye"]))
    d.update(observer(d, cfg["generative"]))
    d["X"], d["X_clean"] = encode(d, d["encoders"], cfg["encoding"], rng, return_clean=True)
    names = TARGETS[cfg["model"]["head"]]
    d["target_names"] = names
    d["Y"] = np.stack([d[k] for k in names], axis=1)
    return d


def run_name(param, level, seed, twin=False):
    return f"{param}_{float(level):g}_s{int(seed)}" + ("_twin" if twin else "")


def train_one(cfg, param, level, seed, twin, epochs, n_trials, verbose=False):
    name = run_name(param, level, seed, twin)
    out = RUNS / name
    if (out / "model.pt").exists():
        return name, None
    c = level_config(cfg, param, level, seed, head="fused" if twin else "causal")
    c = tweak(c, n_trials=int(n_trials), epochs=int(epochs))
    from cmsi.models.training import train
    t0 = time.time()
    d = make_dataset_ecc(c, level) if param == "eye_ecc" else make_dataset(c)
    model, history, splits = train(d, c, verbose=verbose)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, c, history, splits, out / "model.pt")
    save_config(c, out / "config.yaml")
    ds_name = run_name(param, level, seed, twin=False)
    (out / "dataset.txt").write_text(ds_name)
    if not twin:
        DATA_INV.mkdir(parents=True, exist_ok=True)
        save_dataset(slim_dataset(d), c, DATA_INV / f"{ds_name}.npz")
    rec = {"name": name, "param": param, "level": float(level), "seed": int(seed), "twin": bool(twin),
           "best_epoch": int(history["best_epoch"]), "best_val": float(history["best_val"]),
           "rejection_rate": float(np.asarray(d.get("rejection_rate", [0.0])).ravel()[0]),
           "realized_c1": float(np.mean(d["C"] == 1)), "n_trials": int(len(d["X"])),
           "sigma2_eye_range": [float(v) for v in c["generative"]["sigma2_eye_range"]],
           "eye_sigma_sq": float(c["generative"]["eye_sigma_sq"]),
           "ecc_c": float(level) if param == "eye_ecc" else 0.0,
           "train_seconds": round(time.time() - t0, 1), "dataset": ds_name}
    return name, rec


def append_manifest(rec):
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / "manifest.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    rows = [r for r in rows if r["name"] != rec["name"]] + [rec]
    p.write_text(json.dumps(rows, indent=1))


def main(a):
    import torch
    torch.set_num_threads(int(a.threads))
    cfg = load_config(a.config)
    jobs = [(a.param, lv, s, tw) for s in a.seeds for lv in a.levels for tw in ((False, True) if a.twin else (False,))]
    print(f"{a.param}: {len(jobs)} networks", flush=True)
    for param, lv, s, tw in jobs:
        name, rec = train_one(cfg, param, lv, s, tw, a.epochs, a.n)
        if rec is None:
            print(f"  {name}: exists, skipped", flush=True)
            continue
        append_manifest(rec)
        print(f"  {name}: best epoch {rec['best_epoch']}, val {rec['best_val']:.4f}, rejected {100 * rec['rejection_rate']:.1f} %, {rec['train_seconds']} s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "flagship.yaml"))
    p.add_argument("--param", required=True, choices=["eye_noise", "eye_sd", "eye_wide", "eye_ecc"])
    p.add_argument("--levels", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--twin", action="store_true")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--threads", type=int, default=1)
    main(p.parse_args())
