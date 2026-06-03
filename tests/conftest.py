"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal_msi.config import Config, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config() -> Config:
    """The default validated configuration."""
    return load_config(CONFIG_PATH)


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministically seeded numpy generator."""
    return np.random.default_rng(1234)
