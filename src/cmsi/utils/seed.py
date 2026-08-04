"""One integer seeds the whole pipeline."""

import random

import numpy as np


def seed_everything(seed):
    """Seed python / numpy / torch and return a fresh numpy Generator.

    Prefer the returned generator over the global numpy RNG: passing it around
    explicitly is what makes a run reproducible regardless of import order.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
