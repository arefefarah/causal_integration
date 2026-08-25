"""Assembling a dataset: sample -> observe -> encode -> (X, Y).

    d = make_dataset(cfg)
    d["X"], d["Y"]        network input and targets
    d["post_c1"], ...     everything the analytical observer computed
    d["C"], d["s_vis"]    ground truth, for checking only -- never an input
"""

import numpy as np

from cmsi.data.encoding import encode, make_encoders
from cmsi.data.generative import containment_bounds, observer, sample_trials

TARGETS = {
    "causal": ["mu_vis", "var_vis", "mu_prop", "var_prop"],
    "fused": ["fused_mu", "fused_var"],
}


def make_dataset(cfg, n=None, seed=None):
    """Build one dataset as a flat dict of arrays.

    n and seed override cfg["training"]["n_trials"] and cfg["seed"] -- useful for
    a quick check, or for a second test set drawn from the same encoders.

    Trials whose visual measurement falls outside the reliably encoded span
    (visual_field pulled in by 2*rf_width) are rejected and redrawn (SS3);
    the realised rate is stored as d["rejection_rate"]. The pre-Poisson rates
    are stored as d["X_clean"] (SS8.5).
    """
    n = int(n if n is not None else cfg["training"]["n_trials"])
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)

    contain = containment_bounds(cfg["encoding"])
    d = sample_trials(n, cfg["generative"], rng, contain=contain)
    d.update(observer(d, cfg["generative"]))

    d["encoders"] = make_encoders(cfg["encoding"], np.random.default_rng(cfg["seed"]))
    d["X"], d["X_clean"] = encode(d, d["encoders"], cfg["encoding"], rng,
                                  return_clean=True)

    names = TARGETS[cfg["model"]["head"]]
    d["target_names"] = names
    d["Y"] = np.stack([d[k] for k in names], axis=1)
    return d


def split_indices(n, split, rng, stratify=None, n_bins=10):
    """Train/val/test index arrays.

    With stratify=None: a plain shuffled split. With stratify = a per-trial
    scalar (use the analytical posterior post_c1), the split is stratified over
    its quantile bins (SS8.4): each bin contributes the same train/val/test
    proportions, so the test set covers every posterior decile with full power
    and the binned model comparison (SS7.4) is never starved in a bin.
    """
    if stratify is None:
        idx = rng.permutation(n)
        n_train = int(round(split[0] * n))
        n_val = int(round(split[1] * n))
        return {
            "train": idx[:n_train],
            "val": idx[n_train:n_train + n_val],
            "test": idx[n_train + n_val:],
        }

    stratify = np.asarray(stratify)
    assert len(stratify) == n, "stratify must have one value per trial"
    edges = np.quantile(stratify, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, stratify, side="right") - 1, 0, n_bins - 1)

    parts = {"train": [], "val": [], "test": []}
    for b in range(n_bins):
        members = rng.permutation(np.flatnonzero(bins == b))
        k_train = int(round(split[0] * len(members)))
        k_val = int(round(split[1] * len(members)))
        parts["train"].append(members[:k_train])
        parts["val"].append(members[k_train:k_train + k_val])
        parts["test"].append(members[k_train + k_val:])
    return {k: rng.permutation(np.concatenate(v)) for k, v in parts.items()}


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
