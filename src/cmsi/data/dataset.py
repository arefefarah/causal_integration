"""Assembling a dataset: sample -> observe -> encode -> (X, Y).

    d = make_dataset(cfg)
    d["X"], d["Y"]        network input and targets
    d["p_common"], ...    everything the analytical observer computed
    d["C"], d["s_vis"]    ground truth, for checking only -- never an input
"""

import numpy as np

from cmsi.data.encoding import encode, make_encoders
from cmsi.data.generative import observer, sample_trials

TARGETS = {
    "causal": ["mu_vis", "var_vis", "mu_prop", "var_prop"],
    "fused": ["fused_mu", "fused_var"],
}


def make_dataset(cfg, n=None, seed=None):
    """Build one dataset as a flat dict of arrays.

    n and seed override cfg["training"]["n_trials"] and cfg["seed"] -- useful for
    a quick check, or for a second test set drawn from the same encoders.
    """
    n = int(n if n is not None else cfg["training"]["n_trials"])
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)

    d = sample_trials(n, cfg["generative"], rng)
    d.update(observer(d, cfg["generative"]))

    d["encoders"] = make_encoders(cfg["encoding"], np.random.default_rng(cfg["seed"]))
    d["X"] = encode(d, d["encoders"], cfg["encoding"], rng)

    names = TARGETS[cfg["model"]["head"]]
    d["target_names"] = names
    d["Y"] = np.stack([d[k] for k in names], axis=1)
    return d


def split_indices(n, split, rng):
    """Shuffled train/val/test index arrays."""
    idx = rng.permutation(n)
    n_train = int(round(split[0] * n))
    n_val = int(round(split[1] * n))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def subset(d, idx):
    """Slice every per-trial array in a dataset dict by idx (X, Y and all latents).

    Use this instead of indexing keys by hand, so an analysis on the test split
    can't accidentally mix in training trials.
    """
    n = len(d["X"])
    out = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            out[key] = value[idx]
        else:
            out[key] = value
    return out
