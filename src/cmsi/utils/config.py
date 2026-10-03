"""Loading and editing configs.

A config is a plain nested dict with five sections: generative, encoding, model,
training, analysis (plus a top-level seed). Functions take the section they need,
not the whole config, so you can always see what a function actually depends on.
"""

from copy import deepcopy
from pathlib import Path

import yaml

from cmsi.utils.paths import CONFIGS

SECTIONS = ("generative", "encoding", "model", "training", "analysis")


def load_config(path=None):
    """Load a yaml config (defaults to configs/default.yaml) and sanity-check it."""
    path = CONFIGS / "default.yaml" if path is None else path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    check(cfg)
    return cfg


def save_config(cfg, path):
    """Write a config back out -- used to record what produced a run."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def analysis_block(cfg, run_dir, path=None):
    """The `analysis` section a run's stages 3, 4 and the 06 experiment use,
    and where it came from -> (cfg with that section swapped in, source).

    The analysis block (disparity_grid, reliability_levels, ridge_alpha,
    decoder_test_size, ...) is not a training parameter, so it may change
    without retraining. It is taken, in order, from the yaml at `path` when
    one is given (03_analyze.py --config), else from results/<run>/config.yaml
    when that exists, else from the checkpoint's own config (`cfg`, the one
    the run was trained with). Stage 3 writes the block it used back into
    results/<run>/config.yaml, so one `--config` re-analysis makes every later
    stage-4 or 06 run follow it; the generative, encoding, model and training
    sections are always the checkpoint's.
    """
    cfg = deepcopy(cfg)
    stored = Path(run_dir) / "config.yaml"
    if path is not None:
        source = Path(path)
    elif stored.exists():
        source = stored
    else:
        return cfg, "checkpoint"
    with open(source, encoding="utf-8") as fh:
        block = yaml.safe_load(fh).get("analysis")
    if not block:
        raise KeyError(f"{source} has no analysis section")
    cfg["analysis"] = block
    return cfg, str(source)


def tweak(cfg, **overrides):
    """Copy a config with a few values replaced, e.g. tweak(cfg, p_common=0.8).

    Looks the key up in every section, so you don't have to remember which one
    it lives in. Unknown or ambiguous keys raise rather than silently doing
    nothing -- a typo in a sweep is otherwise invisible until the results are odd.
    """
    new = deepcopy(cfg)
    for key, value in overrides.items():
        if key in new and key not in SECTIONS:
            new[key] = value
            continue
        hits = [s for s in SECTIONS if key in new.get(s, {})]
        if not hits:
            raise KeyError(f"unknown config key {key!r}")
        if len(hits) > 1:
            raise KeyError(f"ambiguous key {key!r}, present in {hits}")
        new[hits[0]][key] = value
    return new


def check(cfg):
    """Catch the config mistakes that produce confusing results rather than errors."""
    missing = [s for s in SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"config is missing section(s): {missing}")

    gen, model, train = cfg["generative"], cfg["model"], cfg["training"]
    if not 0.0 <= gen["p_common"] <= 1.0:
        raise ValueError("p_common must be in [0, 1]")
    for key in ("sigma0_sq", "eye_sigma_sq"):
        # eye_sigma_sq is used by the observer as well as the sampler: it sets how
        # far the eye measurement is shrunk toward its prior in to_body_frame.
        # Use float("inf") for a flat prior on eye position.
        if not gen[key] > 0:
            raise ValueError(f"{key} must be positive, got {gen[key]}")
    for key in ("sigma2_vis_range", "sigma2_prop_range", "sigma2_eye_range"):
        lo, hi = gen[key]
        if not 0 < lo <= hi:
            raise ValueError(f"{key} must satisfy 0 < lo <= hi, got {gen[key]}")
    if abs(sum(train["split"]) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test split must sum to 1, got {train['split']}")
    if model["head"] not in ("causal", "fused"):
        raise ValueError(f"head must be 'causal' or 'fused', got {model['head']!r}")
    if train["optimizer"] == "rprop" and train["batch_size"] < train["n_trials"]:
        print("warning: rprop is a full-batch method; expect noisy mini-batch training")
    return cfg
