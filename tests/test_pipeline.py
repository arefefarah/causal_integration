"""The stage boundaries: what one script writes, the next must be able to read."""

import numpy as np

from cmsi.data import make_dataset
from cmsi.models import predict, train
from cmsi.utils import (
    load_checkpoint,
    load_dataset,
    load_json,
    save_checkpoint,
    save_dataset,
    save_json,
)


def test_dataset_survives_a_round_trip(cfg, tmp_path):
    d = make_dataset(cfg, n=200)
    save_dataset(d, cfg, tmp_path / "d.npz")
    loaded, loaded_cfg = load_dataset(tmp_path / "d.npz")

    assert loaded_cfg == cfg
    assert loaded["target_names"] == d["target_names"]
    assert np.array_equal(loaded["X"], d["X"])
    assert np.array_equal(loaded["p_common"], d["p_common"])
    assert all(np.array_equal(loaded["encoders"][k], v)
               for k, v in d["encoders"].items())


def test_checkpoint_reloads_to_the_same_predictions(cfg, tmp_path):
    """A reloaded model must include the standardiser, or predictions silently
    change between training and analysis."""
    d = make_dataset(cfg, n=400)
    model, history, splits = train(d, cfg, verbose=False)
    before = predict(model, d["X"][:50])

    save_checkpoint(model, cfg, history, splits, tmp_path / "model.pt")
    reloaded, cfg_back, _, splits_back = load_checkpoint(tmp_path / "model.pt")

    assert np.allclose(predict(reloaded, d["X"][:50]), before, atol=1e-6)
    assert cfg_back == cfg
    assert np.array_equal(splits_back["test"], splits["test"])


def test_metrics_json_handles_numpy_values(tmp_path):
    metrics = {"r2": np.float64(0.9), "n": np.int64(5), "curve": np.arange(3)}
    save_json(metrics, tmp_path / "m.json")
    assert load_json(tmp_path / "m.json") == {"r2": 0.9, "n": 5, "curve": [0, 1, 2]}
