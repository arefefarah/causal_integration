r"""Ablation: silence opposite units and test for a causal-readout dissociation.

Silences (zeros) a set of MSL units -- typically the "opposite" units from
:mod:`causal_msi.analysis.congruent_opposite` -- and measures whether the causal
read-out (output 5, ``p(C=1)``) degrades MORE than the estimate read-outs
(outputs 1-4). A selective deficit is evidence that opposite units specifically
support the causal inference.

The forward-pass-with-mask plumbing is implemented; the dissociation statistic is
``TODO(science)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from causal_msi.models import FeedforwardMSI

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def forward_with_msl_mask(model: FeedforwardMSI, x: FloatArray, silence: BoolArray) -> FloatArray:
    """Forward pass with selected MSL units silenced (zeroed).

    Parameters
    ----------
    model
        Trained network.
    x
        Inputs, shape ``(N, input_dim)``.
    silence
        Boolean mask over MSL units, shape ``(h2,)``; True units are zeroed before
        the read-out.

    Returns
    -------
    numpy.ndarray
        Head outputs with the ablation applied, shape ``(N, output_dim)``.
    """
    model.eval()
    mask = torch.as_tensor(~silence, dtype=torch.float32)
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    with torch.no_grad():
        h = xt
        for i, module in enumerate(model.hidden):
            h = module(h)
            if i == model._msl_idx:
                h = h * mask
        raw = model.readout(h)
        out = model._apply_head(raw)
    return out.cpu().numpy()


@dataclass(frozen=True)
class DissociationResult:
    """Change in each read-out's error after ablation."""

    delta_estimate_error: float  # outputs 1-4 (means + vars)
    delta_pc_error: float  # output 5 (p(C=1))
    dissociation: float  # pc degradation relative to estimate degradation


def ablation_dissociation(
    baseline: FloatArray,
    ablated: FloatArray,
    target: FloatArray,
) -> DissociationResult:
    """Quantify whether ablation degrades the causal readout more than estimates.

    Parameters
    ----------
    baseline
        Intact-network outputs, shape ``(N, 5)``.
    ablated
        Post-ablation outputs, shape ``(N, 5)``.
    target
        Analytical targets, shape ``(N, 5)``.

    Returns
    -------
    DissociationResult
        Change in estimate-readout error, change in ``p(C=1)`` error, and their
        dissociation.

    Notes
    -----
    TODO(science): define the dissociation statistic -- e.g. the increase in
    ``p(C=1)`` error (output 5) divided by the increase in estimate error (outputs
    1-4) after silencing opposite units. A dissociation >> 1 supports a selective
    role for opposite units in causal inference. Tested in the ablation tests.
    """
    raise NotImplementedError("TODO(science): ablation dissociation statistic")
