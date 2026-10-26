"""Shared pieces of the transformation-to-causality investigation (2026-10-13).

Imports the repo's `cmsi` package and `scripts/neuronal/npmodel.py` unchanged
and writes only under results/transformation_investigation/ and
data/transformation_investigation/; nothing in results/<run>/ is touched.

Layout it reads and writes:
    results/<run>/                       the stored flagship, twin and seeds (as in the repo)
    results/transformation_investigation/runs/<name>/{model.pt, config.yaml, dataset.txt}
    data/transformation_investigation/<dataset>.npz   slim datasets (X float32, no X_clean)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_root(start):
    """The repo root: the first ancestor of this file that holds src/cmsi, so the
    code runs from scripts/transformation/ or from any other folder inside the
    repo."""
    for p in (start, *start.parents):
        if (p / "src" / "cmsi").is_dir():
            return p
    raise RuntimeError(f"no repo root (a folder holding src/cmsi) above {start}")


ROOT = _find_root(HERE)
SRC = ROOT / "src"
OUT = ROOT / "results" / "transformation_investigation"
RUNS = OUT / "runs"
DATA_INV = ROOT / "data" / "transformation_investigation"
for p in (str(SRC), str(ROOT / "scripts" / "neuronal")):
    if p not in sys.path:
        sys.path.insert(0, p)

# The analysis is torch-free: the forward pass is scripts/neuronal/npmodel.py's,
# and `stub_cmsi` lets the cmsi submodules load without cmsi/__init__ (which
# imports torch). Only train_levels.py needs torch.
from npmodel import NpNet, load_checkpoint_np, stub_cmsi  # noqa: E402

stub_cmsi()
from cmsi.analysis.causal import (binned_weight, hybrid_weight, mean_by_bin,  # noqa: E402,F401
                                  model_comparison, position_regression,
                                  transition_fit, variance_signature)   # re-exported
from cmsi.data.encoding import encode  # noqa: E402
from cmsi.data.generative import observer  # noqa: E402
from cmsi.utils.io import load_dataset  # noqa: E402

CLASS_COLORS = {"congruent": "#1f77b4", "opposite": "#d62728", "mixed": "#ccbb44"}
NEAR, FAR = 2.0, 15.0            # fusion |d| <= NEAR, segregation |d| >= FAR (as in the repo)
TRUE_NEAR = 2.0                  # "truly common-cause" trials: |s_vis - s_prop| <= TRUE_NEAR


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def sigma_out(control="pcommon1"):
    """The read-out noise scale the repo uses for runs without their own
    metrics.json: the control's residual_std (03_analyze --control)."""
    m = json.loads((ROOT / "results" / control / "metrics.json").read_text())
    return np.asarray(m["residual_std"], float)


def _dataset_path(name, cfg=None):
    """The dataset's npz, or -- for an investigation run whose slim dataset is not
    on disk -- the dataset regenerated from the run's own config (make_dataset is
    seeded by cfg["seed"], so the regenerated trials, encoders and Poisson draws
    are identical) and written to data/transformation_investigation/."""
    for folder in (DATA_INV, ROOT / "data"):
        p = folder / f"{name}.npz"
        if p.exists():
            return p
    if cfg is None:
        raise FileNotFoundError(f"dataset {name!r} not found in {DATA_INV} or {ROOT / 'data'}")
    from cmsi.data.dataset import make_dataset
    from cmsi.utils.io import save_dataset
    c = dict(cfg)
    c["model"] = dict(cfg["model"], head="causal")       # the twin's dataset.txt names the causal dataset
    if name.startswith("eye_ecc_"):
        sys.path.insert(0, str(HERE))
        from train_levels import make_dataset_ecc
        d = make_dataset_ecc(c, float(name.split("_")[2]))
    else:
        d = make_dataset(c)
    DATA_INV.mkdir(parents=True, exist_ok=True)
    save_dataset(slim_dataset(d), c, DATA_INV / f"{name}.npz")
    return DATA_INV / f"{name}.npz"


def load_run(path):
    """A run folder (results/<run> or results/transformation_investigation/runs/<name>)
    -> dict(net, cfg, ckpt, d (test split), d_full, encoders, names, sig_out).

    Twins trained by train_levels.py record the CAUSAL dataset in dataset.txt
    (same trials, same encoders); only their config says head: fused."""
    path = Path(path)
    if not path.exists() and (ROOT / "results" / str(path)).exists():
        path = ROOT / "results" / str(path)
    ckpt = load_checkpoint_np(path / "model.pt")
    cfg = ckpt["config"]
    name = (path / "dataset.txt").read_text().strip()
    d_full, _ = load_dataset(_dataset_path(name, cfg))
    idx = np.asarray(ckpt["splits"]["test"])
    n = len(d_full["X"])
    d = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
         for k, v in d_full.items()}
    net = NpNet(ckpt)
    return {"path": path, "name": path.name, "net": net, "cfg": cfg, "ckpt": ckpt, "d": d,
            "d_full": d_full, "encoders": d_full["encoders"], "names": list(d_full["target_names"]),
            "head": cfg["model"]["head"], "sig_out": sigma_out(), "n_test": int(len(idx))}


def slim_dataset(d):
    """The dataset with X as float32 and without X_clean (a quarter of the size)."""
    out = {}
    for k, v in d.items():
        if k == "X_clean":
            continue
        out[k] = v.astype(np.float32) if k == "X" else v
    return out


# --------------------------------------------------------------------------- #
# observers
# --------------------------------------------------------------------------- #
def observer_on(trials, gen, x_eye=None, sig2_eye=None, eye_known=False):
    """The repo's observer on a trial dict, optionally with a replaced eye
    signal (x_eye, sig2_eye) or with the TRUE eye position (eye_known=True:
    x_eye := eye, sig2_eye := 1e-9, i.e. var_eye -> 0, t_err -> 0 -- the
    transformation-free observer R1). Returns the observer dict."""
    t = dict(trials)
    if eye_known:
        t["x_eye"] = np.asarray(trials["eye"], float)
        t["sig2_eye"] = np.full(len(t["x_eye"]), 1e-9)
    if x_eye is not None:
        t["x_eye"] = np.asarray(x_eye, float)
    if sig2_eye is not None:
        t["sig2_eye"] = np.asarray(sig2_eye, float) * np.ones(len(t["x_eye"]))
    return observer(t, gen)


def eye_gain_k(gen, sig2_eye):
    return gen["eye_sigma_sq"] / (gen["eye_sigma_sq"] + np.asarray(sig2_eye, float))


def transformation_error(d, gen):
    """t_err = eye_hat - eye: the error of the observer's eye estimate, per trial."""
    k = eye_gain_k(gen, d["sig2_eye"])
    return k * d["x_eye"] + (1 - k) * gen["eye_mu"] - d["eye"], k


# --------------------------------------------------------------------------- #
# the network on (possibly manipulated) trials
# --------------------------------------------------------------------------- #
def encode_trials(trials, encoders, enc, poisson=True, seed=1):
    e = dict(enc, poisson_noise=bool(poisson))
    return encode(trials, encoders, e, np.random.default_rng(seed))


def predict_on(b, trials, poisson=True, seed=1):
    X = encode_trials(trials, b["encoders"], b["cfg"]["encoding"], poisson, seed)
    return b["net"].predict(X)


def hybrid_reads(b, pred, obs):
    """The visual and hand implied weights of a causal network's predictions
    against an observer dict `obs` (R0 on the same trials), exactly as
    03_analyze / scripts/neuronal/common.py compute them."""
    n = b["names"]
    iv, ip, ivv, ivp = (n.index(k) for k in ("mu_vis", "mu_prop", "var_vis", "var_prop"))
    so = b["sig_out"]
    dv = obs["fused_mu"] - obs["seg_vis_mu"]
    dp = obs["fused_mu"] - obs["seg_prop_mu"]
    w, _, flags = hybrid_weight(pred[:, ivv], obs["fused_var"], obs["seg_vis_var"], dv,
                                pred[:, iv], obs["seg_vis_mu"], so[iv], so[ivv])
    wp, _, _ = hybrid_weight(pred[:, ivp], obs["fused_var"], obs["seg_prop_var"], dp,
                             pred[:, ip], obs["seg_prop_mu"], so[ip], so[ivp])
    return w, wp, flags


def trial_dict(d, keys=("x_vis", "x_prop", "x_eye", "sig2_vis", "sig2_prop", "sig2_eye",
                        "eye", "s_vis", "s_prop", "C", "retinal")):
    return {k: np.asarray(d[k], float) if k != "C" else np.asarray(d[k]) for k in keys if k in d}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def crossing(x, y, step=1.0, lo=0.0, hi=31.0, min_n=30):
    """The |x| at which the binned mean of y first falls through 0.5."""
    ax, y = np.abs(np.asarray(x, float)), np.asarray(y, float)
    ok = np.isfinite(y)
    edges = np.arange(lo, hi + step, step)
    c, m = [], []
    for a, bb in zip(edges[:-1], edges[1:]):
        s = ok & (ax >= a) & (ax < bb)
        if s.sum() >= min_n:
            c.append(0.5 * (a + bb)); m.append(y[s].mean())
    for i in range(1, len(c)):
        if m[i - 1] >= 0.5 > m[i]:
            return float(c[i - 1] + (m[i - 1] - 0.5) / (m[i - 1] - m[i]) * (c[i] - c[i - 1]))
    return float("nan")


def lookup_r2(x, y, step=2.0, hi=32.0):
    """R2 of predicting y from a binned lookup on |x|."""
    ax, y = np.abs(np.asarray(x, float)), np.asarray(y, float)
    ok = np.isfinite(y)
    edges = np.arange(0, hi + step, step)
    c, m = [], []
    for a, bb in zip(edges[:-1], edges[1:]):
        s = ok & (ax >= a) & (ax < bb)
        if s.sum() >= 30:
            c.append(0.5 * (a + bb)); m.append(y[s].mean())
    if len(c) < 3:
        return float("nan")
    pred = np.interp(ax, c, m)
    return float(1 - np.var(y[ok] - pred[ok]) / np.var(y[ok]))


def binned_mean(x, y, edges, min_n=20):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    out = []
    for a, bb in zip(edges[:-1], edges[1:]):
        s = ok & (x >= a) & (x < bb)
        out.append({"lo": float(a), "hi": float(bb), "n": int(s.sum()),
                    "mean": float(y[s].mean()) if s.sum() >= min_n else float("nan")})
    return out


def peak_location(x, y, edges, smooth=2):
    """Location of the maximum of the binned mean of y over signed x, after a
    moving average over `smooth` neighbouring bins on each side."""
    rows = binned_mean(x, y, edges, min_n=15)
    c = np.array([0.5 * (r["lo"] + r["hi"]) for r in rows])
    m = np.array([r["mean"] for r in rows])
    ok = np.isfinite(m)
    if ok.sum() < 3:
        return float("nan")
    ms = np.full(len(m), np.nan)
    for i in range(len(m)):
        j = slice(max(0, i - smooth), min(len(m), i + smooth + 1))
        v = m[j][np.isfinite(m[j])]
        ms[i] = v.mean() if len(v) else np.nan
    i = int(np.nanargmax(ms))
    # parabolic refinement
    if 0 < i < len(ms) - 1 and np.isfinite(ms[i - 1]) and np.isfinite(ms[i + 1]):
        den = ms[i - 1] - 2 * ms[i] + ms[i + 1]
        off = 0.5 * (ms[i - 1] - ms[i + 1]) / den if den != 0 else 0.0
        return float(c[i] + off * (c[1] - c[0]))
    return float(c[i])


def lstsq_coefs(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1))
    return path
