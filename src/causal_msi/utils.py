"""Seeding and device helpers.

A single seeding utility seeds both numpy and torch so every stage of the
pipeline (data generation, training, analysis) is reproducible from one integer.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> np.random.Generator:
    """Seed Python, numpy, and torch RNGs and return a numpy Generator.

    Parameters
    ----------
    seed
        Integer seed applied to ``random``, ``numpy``, and ``torch`` (CPU + CUDA).
    deterministic
        If True, request deterministic cuDNN/torch algorithms. This can slow
        training but guarantees bit-reproducibility across runs.

    Returns
    -------
    numpy.random.Generator
        A freshly seeded :class:`numpy.random.Generator` (PCG64) to thread through
        the generative model. Prefer this over the global numpy RNG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer
        Optional explicit device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        If None, prefer CUDA, then Apple MPS, then CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_tensor(array: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """Convert a numpy array to a float32 torch tensor on ``device``.

    Parameters
    ----------
    array
        Input array of any shape.
    device
        Target device; defaults to :func:`get_device`.

    Returns
    -------
    torch.Tensor
        float32 tensor with the same shape as ``array``.
    """
    if device is None:
        device = get_device()
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
