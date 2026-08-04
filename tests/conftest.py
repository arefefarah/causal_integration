"""Shared fixtures. Also puts src/ on the path so pytest runs with no install."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmsi.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """The default config, shrunk so the whole suite runs in a few seconds."""
    c = load_config()
    c["training"]["n_trials"] = 2000
    c["training"]["epochs"] = 3
    return c


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def dataset(cfg):
    from cmsi.data import make_dataset
    return make_dataset(cfg)
