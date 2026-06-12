"""Tests for the feedforward model and both heads (implemented -- expected to pass)."""

from __future__ import annotations

import torch

from causal_msi.config import ModelConfig
from causal_msi.models import build_model


def _causal_cfg() -> ModelConfig:
    return ModelConfig(hidden_sizes=[64, 64], hidden_activation="sigmoid", head_type="causal")


def _integration_cfg() -> ModelConfig:
    return ModelConfig(
        hidden_sizes=[64, 64], hidden_activation="sigmoid", head_type="integration_only"
    )


def test_causal_forward_shape() -> None:
    """The causal head produces 4 outputs per trial (Kording Eqs. 9/10 + vars)."""
    model = build_model(input_dim=130, cfg=_causal_cfg())
    out = model(torch.randn(8, 130))
    assert out.shape == (8, 4)
    assert model.output_dim == 4


def test_integration_forward_shape() -> None:
    """The integration-only head produces 2 outputs per trial."""
    model = build_model(input_dim=130, cfg=_integration_cfg())
    out = model(torch.randn(8, 130))
    assert out.shape == (8, 2)
    assert model.output_dim == 2


def test_variance_outputs_positive() -> None:
    """Softplus keeps the two variance outputs strictly positive."""
    model = build_model(input_dim=50, cfg=_causal_cfg())
    out = model(torch.randn(64, 50) * 10)
    assert torch.all(out[:, 1] > 0)
    assert torch.all(out[:, 3] > 0)


def test_no_pc_output() -> None:
    """The causal head has no p(C=1) output -- exactly 4 columns."""
    model = build_model(input_dim=50, cfg=_causal_cfg())
    out = model(torch.randn(64, 50) * 10)
    assert out.shape[1] == 4


def test_integration_variance_positive() -> None:
    """The twin's single variance output is positive."""
    model = build_model(input_dim=50, cfg=_integration_cfg())
    out = model(torch.randn(64, 50) * 10)
    assert torch.all(out[:, 1] > 0)


def test_activation_hooks_return_sil_msl() -> None:
    """The activation hooks expose SIL and MSL tensors of the right shape."""
    cfg = _causal_cfg()
    model = build_model(input_dim=40, cfg=cfg)
    cache = model.activations(torch.randn(7, 40))
    assert cache.sil is not None and cache.msl is not None
    assert cache.sil.shape == (7, cfg.hidden_sizes[0])
    assert cache.msl.shape == (7, cfg.hidden_sizes[1])


def test_return_cache_matches_plain_forward() -> None:
    """forward(return_cache=True) returns the same outputs as the plain call."""
    model = build_model(input_dim=20, cfg=_causal_cfg())
    model.eval()
    x = torch.randn(5, 20)
    with torch.no_grad():
        plain = model(x)
        out, _ = model(x, return_cache=True)
    torch.testing.assert_close(plain, out)
