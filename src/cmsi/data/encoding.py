"""Scalar measurements -> population-code vectors (the network input).

Three groups, concatenated in this order:

    visual_hand  gaussian receptive fields over the RETINAL visual measurement
    prop_hand    linear push-pull code of the proprioceptive measurement
    prop_eye     linear push-pull code of the eye-position measurement

Each group's activity is scaled by the cue's reliability (gain = K / variance)
and then corrupted by Poisson spike-count noise. The network sees ONLY these
three groups -- never p(C=1), disparity, the true sources, eye position, or C.
"""

import numpy as np


def make_encoders(enc, rng):
    """Fixed tuning parameters, held constant across a dataset.

    Seed this from cfg["seed"] rather than the data generator: two datasets built
    from the same config then share an input basis, so a model trained on one can
    be evaluated on the other.
    """
    return {
        "rf_centers": np.linspace(*enc["visual_field"], enc["n_vis"]),
        "prop_slope": rng.uniform(*enc["slope_range"], enc["n_prop"]),
        "prop_intercept": rng.uniform(*enc["intercept_range"], enc["n_prop"]),
        "eye_slope": rng.uniform(*enc["slope_range"], enc["n_eye"]),
        "eye_intercept": rng.uniform(*enc["intercept_range"], enc["n_eye"]),
    }


def gaussian_code(x, centers, width, gain):
    """r[i,j] = gain[i] * exp(-(x[i] - c[j])^2 / 2w^2)  -> (n_trials, n_units)."""
    bumps = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / width) ** 2)
    return gain[:, None] * bumps


def push_pull_code(x, slope, intercept, gain):
    """r[i,j] = gain[i] * relu(slope[j]*x[i] + intercept[j]).

    Mixed-sign slopes give oppositely tuned (push vs pull) units; the rectifier
    keeps rates non-negative so Poisson noise is well defined.
    """
    drive = x[:, None] * slope[None, :] + intercept[None, :]
    return gain[:, None] * np.clip(drive, 0, None)


def encode_groups(d, encoders, enc, rng):
    """The three population codes, as a dict of (n_trials, n_units) arrays."""
    K = enc["gain_K"]
    groups = {
        "visual_hand": gaussian_code(
            d["x_vis"], encoders["rf_centers"], enc["rf_width"], K / d["sig2_vis"]
        ),
        "prop_hand": push_pull_code(
            d["x_prop"], encoders["prop_slope"], encoders["prop_intercept"],
            K / d["sig2_prop"]
        ),
        "prop_eye": push_pull_code(
            d["x_eye"], encoders["eye_slope"], encoders["eye_intercept"],
            K / d["sig2_eye"]
        ),
    }
    if enc["poisson_noise"]:
        groups = {k: rng.poisson(np.clip(v, 0, None)).astype(float)
                  for k, v in groups.items()}
    return groups


def encode(d, encoders, enc, rng):
    """Concatenated network input X, shape (n_trials, n_vis + n_prop + n_eye)."""
    g = encode_groups(d, encoders, enc, rng)
    return np.concatenate([g["visual_hand"], g["prop_hand"], g["prop_eye"]], axis=1)


def group_slices(enc):
    """Column slices of X per group -- handy when inspecting first-layer weights."""
    a, b, c = enc["n_vis"], enc["n_prop"], enc["n_eye"]
    return {
        "visual_hand": slice(0, a),
        "prop_hand": slice(a, a + b),
        "prop_eye": slice(a + b, a + b + c),
    }


def input_dim(enc):
    return enc["n_vis"] + enc["n_prop"] + enc["n_eye"]
