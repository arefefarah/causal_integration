"""Tests for the integration analysis (CORE SCIENCE -- expected to xfail).

These encode the arithmetic the implicit-integration analysis must satisfy. They
xfail against ``NotImplementedError`` until the formulae in
``causal_msi.analysis.integration`` (and the related Bayes-factor inversion) are
implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_msi.analysis.bayes_factor import implied_log_bf
from causal_msi.analysis.integration import fusion_weight_curve
from causal_msi.generative import ObserverTargets, common_cause_posterior

science = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(science): not yet implemented", strict=False
)


def _toy_targets(n: int = 16) -> ObserverTargets:
    rng = np.random.default_rng(0)
    zeros = np.zeros(n)
    return ObserverTargets(
        mu_vis=rng.normal(size=n),
        var_vis=np.full(n, 2.0),
        mu_prop=rng.normal(size=n),
        var_prop=np.full(n, 3.0),
        seg_vis_mu=rng.normal(size=n),
        seg_vis_var=np.full(n, 2.0),
        seg_prop_mu=rng.normal(size=n),
        seg_prop_var=np.full(n, 3.0),
        fused_mu=rng.normal(size=n),
        fused_var=np.full(n, 1.0),
        p_common=rng.uniform(size=n),
        log_bf=zeros,
    )


@science
def test_model_averaging_endpoints() -> None:
    """Model averaging returns fused when p(C=1)=1 and segregated when p(C=1)=0."""
    from causal_msi.analysis.integration import analytical_model_averaged_estimate

    t = _toy_targets()
    # Force the two extremes by overriding p_common.
    t_common = ObserverTargets(**{**t.__dict__, "p_common": np.ones_like(t.p_common)})
    t_sep = ObserverTargets(**{**t.__dict__, "p_common": np.zeros_like(t.p_common)})
    np.testing.assert_allclose(analytical_model_averaged_estimate(t_common), t.fused_mu)
    np.testing.assert_allclose(analytical_model_averaged_estimate(t_sep), t.seg_vis_mu)


@science
def test_fusion_weight_in_unit_interval() -> None:
    """The empirical fusion weight lies in [0, 1] for estimates between the endpoints."""
    rng = np.random.default_rng(1)
    seg = rng.normal(size=100)
    fused = seg + rng.normal(size=100)
    w_true = rng.uniform(size=100)
    estimate = w_true * fused + (1 - w_true) * seg
    w = fusion_weight_curve(seg, fused, estimate)
    assert np.all((w >= -1e-6) & (w <= 1 + 1e-6))


@science
def test_fusion_weight_recovers_known_mixture() -> None:
    """The fusion weight recovers a known mixing coefficient."""
    seg = np.array([0.0, 0.0, 0.0])
    fused = np.array([10.0, 10.0, 10.0])
    estimate = np.array([2.5, 5.0, 7.5])
    w = fusion_weight_curve(seg, fused, estimate)
    np.testing.assert_allclose(w, [0.25, 0.5, 0.75])


@science
def test_implied_log_bf_round_trip() -> None:
    """Inverting the posterior recovers the log Bayes factor that produced it."""
    p_common = 0.5
    log_bf = np.linspace(-5, 5, 25)
    p = common_cause_posterior(log_bf, p_common)
    recovered = implied_log_bf(p, p_common)
    np.testing.assert_allclose(recovered, log_bf, atol=1e-6)


def test_inferred_common_cause_recovers_analytical(config) -> None:
    """Feeding the analytical optimal estimates back in recovers the analytical p(C=1).

    If the network were perfect (outputs == Kording Eqs. 9/10), inverting its mean
    outputs must return the analytical common-cause posterior on non-degenerate
    trials (where the cues are separated enough to give leverage on the weight).
    """
    import numpy as np

    from causal_msi.analysis.common_cause import inferred_common_cause_posterior
    from causal_msi.generative import build_dataset

    rng = np.random.default_rng(0)
    ds = build_dataset(rng, config, n_trials=4000)
    implied = inferred_common_cause_posterior(ds.Y, ds.targets)

    # Restrict to trials with appreciable disparity (degenerate ones are NaN/ill-posed).
    disparity = np.abs(ds.targets.fused_mu - ds.targets.seg_prop_mu)
    keep = np.isfinite(implied) & (disparity > 1.0)
    assert keep.sum() > 100
    np.testing.assert_allclose(implied[keep], ds.targets.p_common[keep], atol=1e-2)
