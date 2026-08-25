"""Do the population codes carry what they are supposed to carry?"""

import numpy as np
import pytest

from cmsi.data.dataset import make_dataset, split_indices, subset
from cmsi.data.encoding import (
    encode_groups,
    gaussian_code,
    group_slices,
    input_dim,
    make_encoders,
    push_pull_code,
)


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
def test_gaussian_bump_peaks_at_the_stimulus():
    centers = np.linspace(-40, 40, 50)
    x = np.array([-12.0, 0.0, 25.0])
    rates = gaussian_code(x, centers, 8.0, np.ones(3))
    peaks = centers[rates.argmax(1)]
    assert np.all(np.abs(peaks - x) <= (centers[1] - centers[0]))


def test_push_pull_units_are_oppositely_tuned():
    slope = np.array([1.0, -1.0])
    intercept = np.array([10.0, 10.0])
    x = np.array([-5.0, 5.0])
    rates = push_pull_code(x, slope, intercept, np.ones(2))
    assert rates[1, 0] > rates[0, 0], "positive-slope unit rises with x"
    assert rates[1, 1] < rates[0, 1], "negative-slope unit falls with x"


def test_rates_are_never_negative():
    rates = push_pull_code(np.array([-1000.0]), np.array([1.0]),
                           np.array([0.0]), np.ones(1))
    assert np.all(rates >= 0), "Poisson noise needs non-negative rates"


def test_gain_scales_with_reliability():
    centers = np.linspace(-40, 40, 50)
    x = np.zeros(2)
    rates = gaussian_code(x, centers, 8.0, 10.0 / np.array([1.0, 10.0]))
    assert rates[0].sum() == pytest.approx(10 * rates[1].sum())


# --------------------------------------------------------------------------- #
# encoders and assembly
# --------------------------------------------------------------------------- #
def test_encoders_are_reproducible_from_the_seed(cfg):
    """Two datasets from one config must share an input basis, or a model
    trained on the first sees a scrambled input space on the second."""
    a = make_encoders(cfg["encoding"], np.random.default_rng(cfg["seed"]))
    b = make_encoders(cfg["encoding"], np.random.default_rng(cfg["seed"]))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_input_has_the_declared_width(cfg, dataset):
    assert dataset["X"].shape[1] == input_dim(cfg["encoding"])


def test_group_slices_partition_the_input(cfg, dataset):
    slices = group_slices(cfg["encoding"])
    covered = sum(s.stop - s.start for s in slices.values())
    assert covered == dataset["X"].shape[1]


def test_poisson_noise_is_unbiased(cfg, rng):
    """Spike counts are noisy but their mean returns the underlying rate."""
    d = {"x_vis": np.zeros(20000), "x_prop": np.zeros(20000), "x_eye": np.zeros(20000),
         "sig2_vis": np.full(20000, 2.0), "sig2_prop": np.full(20000, 2.0),
         "sig2_eye": np.full(20000, 2.0)}
    encoders = make_encoders(cfg["encoding"], np.random.default_rng(0))

    clean = dict(cfg["encoding"], poisson_noise=False)
    noisy = dict(cfg["encoding"], poisson_noise=True)
    rates = encode_groups(d, encoders, clean, rng)["visual_hand"]
    counts = encode_groups(d, encoders, noisy, rng)["visual_hand"]

    assert np.allclose(counts.mean(0), rates.mean(0), rtol=0.05, atol=0.1)


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #
def test_targets_match_their_names(dataset):
    for i, name in enumerate(dataset["target_names"]):
        assert np.array_equal(dataset["Y"][:, i], dataset[name])


def test_dataset_is_reproducible(cfg):
    a, b = make_dataset(cfg, n=300), make_dataset(cfg, n=300)
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["Y"], b["Y"])


def test_splits_are_disjoint_and_complete(rng):
    splits = split_indices(1000, [0.7, 0.15, 0.15], rng)
    joined = np.concatenate(list(splits.values()))
    assert len(joined) == 1000
    assert len(np.unique(joined)) == 1000


def test_subset_keeps_trials_aligned(dataset):
    """Every per-trial array must be sliced together, or an analysis silently
    compares one trial's prediction to another trial's ground truth."""
    idx = np.array([5, 1, 99, 3])
    part = subset(dataset, idx)
    assert np.array_equal(part["X"], dataset["X"][idx])
    assert np.array_equal(part["post_c1"], dataset["post_c1"][idx])
    assert np.array_equal(part["mu_vis"], dataset["Y"][idx, 0])
    assert part["target_names"] == dataset["target_names"]
