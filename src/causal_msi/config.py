"""Typed configuration models (pydantic) mirroring ``configs/default.yaml``.

Loading a YAML file through :func:`load_config` validates every field and yields
a frozen, type-checked :class:`Config` object that is threaded through data
generation, training, and analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    """Base model: forbid unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerativeConfig(_Base):
    """Generative-model parameters (degrees and deg^2)."""

    p_common: float = Field(ge=0.0, le=1.0)
    mu0: float
    sigma0_sq: float = Field(gt=0.0)

    eye_mu: float
    eye_sigma_sq: float = Field(gt=0.0)

    sigma2_vis_range: tuple[float, float]
    sigma2_prop_range: tuple[float, float]
    sigma2_eye_range: tuple[float, float]

    disparity_grid: list[float]

    @field_validator("sigma2_vis_range", "sigma2_prop_range", "sigma2_eye_range")
    @classmethod
    def _check_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 < lo <= hi):
            raise ValueError(f"variance range must satisfy 0 < lo <= hi, got {v}")
        return v


class EncodingConfig(_Base):
    """Population-code encoding parameters."""

    n_units_visual_hand: int = Field(gt=0)
    n_units_prop_hand: int = Field(gt=0)
    n_units_prop_eye: int = Field(gt=0)

    visual_field: tuple[float, float]
    rf_width: float = Field(gt=0.0)

    gain_K: float = Field(gt=0.0)

    slope_range: tuple[float, float]
    intercept_range: tuple[float, float]

    poisson_noise: bool

    @property
    def input_dim(self) -> int:
        """Total network input dimension (sum of the three group sizes)."""
        return self.n_units_visual_hand + self.n_units_prop_hand + self.n_units_prop_eye

    @property
    def group_sizes(self) -> dict[str, int]:
        """Per-group unit counts, in concatenation order."""
        return {
            "visual_hand": self.n_units_visual_hand,
            "prop_hand": self.n_units_prop_hand,
            "prop_eye": self.n_units_prop_eye,
        }


class ModelConfig(_Base):
    """Network architecture parameters."""

    hidden_sizes: list[int] = Field(min_length=1)
    hidden_activation: Literal["sigmoid", "relu", "tanh"]
    head_type: Literal["causal", "integration_only"]

    @field_validator("hidden_sizes")
    @classmethod
    def _check_hidden(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError(f"hidden sizes must be positive, got {v}")
        return v


class LossWeights(_Base):
    """Per-component weights for the estimate loss (means and variances)."""

    est: float = Field(ge=0.0)
    var: float = Field(ge=0.0)


class TrainingConfig(_Base):
    """Training-loop parameters."""

    optimizer: Literal["rprop", "adam"]
    lr: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    n_trials: int = Field(gt=0)
    split: tuple[float, float, float]
    loss_weights: LossWeights
    n_seeds: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)

    @field_validator("split")
    @classmethod
    def _check_split(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"train/val/test split must sum to 1.0, got {v}")
        return v


class DecoderConfig(_Base):
    """Linear-decoder hyperparameters used across analyses."""

    alpha: float = Field(gt=0.0)
    cv_folds: int = Field(gt=1)
    test_size: float = Field(gt=0.0, lt=1.0)


class AnalysisConfig(_Base):
    """Analysis-suite parameters."""

    disparity_grid: list[float]
    reliability_levels: list[float]
    decoder: DecoderConfig


class Config(_Base):
    """Top-level configuration aggregating every section."""

    seed: int
    generative: GenerativeConfig
    encoding: EncodingConfig
    model: ModelConfig
    training: TrainingConfig
    analysis: AnalysisConfig

    @model_validator(mode="after")
    def _check_head_consistency(self) -> Config:
        # No cross-section constraints yet; hook left for future invariants.
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML file structured like ``configs/default.yaml``.

    Returns
    -------
    Config
        A validated, frozen configuration object.

    Raises
    ------
    pydantic.ValidationError
        If any field is missing, mistyped, or out of range.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
