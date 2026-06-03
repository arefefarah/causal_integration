"""Tests for the population-code encoders (implemented -- expected to pass)."""

from __future__ import annotations

import numpy as np

from causal_msi.encoding import (
    Encoders,
    PushPullParams,
    assemble_inputs,
    encode_groups,
    gaussian_rf_code,
    make_rf_centers,
    poisson_noise,
    push_pull_code,
    reliability_gain,
)
from causal_msi.generative import LatentBatch, Measurements, sample_latents


def test_gaussian_rf_peaks_at_center() -> None:
    """A Gaussian RF unit responds maximally when the stimulus equals its centre."""
    centers = np.array([0.0])
    x = np.linspace(-20, 20, 41)
    gain = np.ones_like(x)
    resp = gaussian_rf_code(x, centers, width=5.0, gain=gain)
    assert resp.shape == (41, 1)
    assert np.argmax(resp[:, 0]) == np.argmin(np.abs(x - 0.0))


def test_gaussian_rf_width_controls_spread() -> None:
    """A wider RF gives a higher response at a fixed offset from the centre."""
    x = np.array([5.0])
    gain = np.ones(1)
    narrow = gaussian_rf_code(x, np.array([0.0]), width=2.0, gain=gain)
    wide = gaussian_rf_code(x, np.array([0.0]), width=10.0, gain=gain)
    assert wide[0, 0] > narrow[0, 0]


def test_push_pull_is_monotonic_and_nonnegative() -> None:
    """Push-pull units are monotonic in the stimulus and never negative."""
    x = np.linspace(-10, 10, 21)
    params = PushPullParams(slopes=np.array([1.0, -1.0]), intercepts=np.array([5.0, 5.0]))
    resp = push_pull_code(x, params, gain=np.ones_like(x))
    assert resp.shape == (21, 2)
    assert np.all(resp >= 0.0)
    # Positive-slope unit increases, negative-slope unit decreases (where unrectified).
    assert resp[-1, 0] >= resp[0, 0]
    assert resp[0, 1] >= resp[-1, 1]


def test_reliability_gain_scales_with_inverse_variance() -> None:
    """Gain scales with 1/variance: halving the variance doubles the gain."""
    g = reliability_gain(np.array([2.0, 4.0, 8.0]), gain_K=10.0)
    np.testing.assert_allclose(g, np.array([5.0, 2.5, 1.25]))
    # Higher reliability (lower variance) -> higher gain.
    assert g[0] > g[1] > g[2]


def test_poisson_mean_matches_lambda() -> None:
    """Poisson sample mean converges to the rate (lambda) over many trials."""
    rng = np.random.default_rng(0)
    rates = np.full((20000, 3), [1.0, 5.0, 10.0])
    counts = poisson_noise(rng, rates)
    np.testing.assert_allclose(counts.mean(axis=0), [1.0, 5.0, 10.0], atol=0.1)


def test_group_dimensions(config) -> None:
    """Encoded input dimensions match the configured per-group unit counts."""
    rng = np.random.default_rng(0)
    latents = sample_latents(rng, 32, config.generative)
    meas = Measurements(
        x_vis=rng.normal(size=32), x_eye=rng.normal(size=32), x_prop=rng.normal(size=32)
    )
    encoders = Encoders.build(rng, config.encoding)
    groups = encode_groups(rng, meas, latents, config.encoding, encoders)
    assert groups["visual_hand"].shape == (32, config.encoding.n_units_visual_hand)
    assert groups["prop_hand"].shape == (32, config.encoding.n_units_prop_hand)
    assert groups["prop_eye"].shape == (32, config.encoding.n_units_prop_eye)


def test_assembled_input_dim(config) -> None:
    """The assembled input width matches EncodingConfig.input_dim."""
    rng = np.random.default_rng(0)
    latents = sample_latents(rng, 16, config.generative)
    meas = Measurements(
        x_vis=rng.normal(size=16), x_eye=rng.normal(size=16), x_prop=rng.normal(size=16)
    )
    x = assemble_inputs(rng, meas, latents, config.encoding, p_common=config.generative.p_common)
    assert x.shape == (16, config.encoding.input_dim)


def test_rf_centers_span_visual_field(config) -> None:
    """RF centres span the configured visual field inclusively."""
    centers = make_rf_centers(config.encoding.n_units_visual_hand, config.encoding.visual_field)
    assert centers[0] == config.encoding.visual_field[0]
    assert centers[-1] == config.encoding.visual_field[1]


def test_sample_latents_common_sources_match(config) -> None:
    """When C == 1 the visual and proprioceptive world sources coincide."""
    rng = np.random.default_rng(5)
    latents: LatentBatch = sample_latents(rng, 500, config.generative)
    common = latents.C == 1
    np.testing.assert_allclose(latents.s_vis[common], latents.s_prop[common])
