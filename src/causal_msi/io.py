"""Persistence helpers for datasets, checkpoints, and results.

Everything is written under ``outputs/`` (or a caller-supplied directory):

- datasets as compressed ``.npz`` (inputs, targets, latents, measurements),
- checkpoints as torch ``.pth`` (model state, config, metrics),
- results as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from causal_msi.config import Config
from causal_msi.generative import Dataset, LatentBatch, Measurements, ObserverTargets


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Save a :class:`Dataset` to a compressed ``.npz`` file.

    Parameters
    ----------
    dataset
        The dataset to save.
    path
        Destination path (``.npz`` appended if missing). Parent dirs are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"X": dataset.X, "Y": dataset.Y}
    for name, dc in (
        ("lat", dataset.latents),
        ("meas", dataset.measurements),
        ("tgt", dataset.targets),
    ):
        for key, val in asdict(dc).items():
            arrays[f"{name}__{key}"] = np.asarray(val)
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path


def load_dataset(path: str | Path) -> Dataset:
    """Load a :class:`Dataset` previously written by :func:`save_dataset`.

    Parameters
    ----------
    path
        Path to the ``.npz`` file.

    Returns
    -------
    Dataset
        The reconstructed dataset.
    """
    data = np.load(Path(path))

    def group(prefix: str) -> dict[str, np.ndarray]:
        plen = len(prefix) + 2
        return {k[plen:]: data[k] for k in data.files if k.startswith(prefix + "__")}

    latents = LatentBatch(**group("lat"))
    measurements = Measurements(**group("meas"))
    targets = ObserverTargets(**group("tgt"))
    return Dataset(
        X=data["X"],
        Y=data["Y"],
        latents=latents,
        measurements=measurements,
        targets=targets,
    )


def save_checkpoint(
    model: nn.Module,
    config: Config,
    metrics: dict[str, Any],
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint with its config and metrics.

    Parameters
    ----------
    model
        Trained model.
    config
        The configuration used for training.
    metrics
        Training/validation metrics to embed.
    path
        Destination ``.pth`` path. Parent dirs are created.
    extra
        Optional additional payload (e.g. seed, input_dim).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.suffix != ".pth":
        path = path.with_suffix(".pth")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": config.model_dump(),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | None = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload.

    Parameters
    ----------
    path
        Path to the ``.pth`` file.
    map_location
        Torch ``map_location`` for deserialisation.

    Returns
    -------
    dict
        The payload dict (``state_dict``, ``config``, ``metrics``, extras).
    """
    payload: dict[str, Any] = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    return payload


def save_results(results: dict[str, Any], path: str | Path) -> Path:
    """Write a results dict to JSON.

    Parameters
    ----------
    results
        JSON-serialisable results.
    path
        Destination ``.json`` path. Parent dirs are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:  # noqa: ANN401 -- generic JSON encoder hook
    """Fallback JSON encoder for numpy scalars/arrays."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
