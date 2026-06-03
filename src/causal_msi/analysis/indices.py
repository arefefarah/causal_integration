r"""Prior-project response indices: additivity, enhancement, additivity, gain.

Reuses the metrics from the prior multisensory-integration project so the causal
network can be compared to the divisive-normalization account:

- AI  -- additivity index (multisensory vs sum of unisensory responses),
- RE  -- response enhancement (multisensory vs max unisensory),
- RA  -- response additivity (multisensory vs linear sum),
- gain index -- reliability-driven gain change per unit.

Each function takes the relevant unisensory/multisensory responses; the exact
index formulae are ``TODO(science)``. Responses are activity (rate-like).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def additivity_index(r_visual: FloatArray, r_prop: FloatArray, r_multi: FloatArray) -> FloatArray:
    """Additivity index (AI) per unit.

    Parameters
    ----------
    r_visual
        Visual-only responses, shape ``(n_units,)`` or ``(N, n_units)``.
    r_prop
        Proprioception-only responses, same shape.
    r_multi
        Multisensory (both cues) responses, same shape.

    Returns
    -------
    numpy.ndarray
        AI per unit.

    Notes
    -----
    TODO(science): ``AI = (r_multi - (r_visual + r_prop)) / (r_visual + r_prop)``
    (or the project's exact normalisation). Positive = super-additive.
    """
    raise NotImplementedError("TODO(science): additivity index formula")


def response_enhancement(
    r_visual: FloatArray, r_prop: FloatArray, r_multi: FloatArray
) -> FloatArray:
    """Response enhancement (RE): multisensory vs the strongest unisensory cue.

    Parameters
    ----------
    r_visual, r_prop, r_multi
        Unisensory and multisensory responses (see :func:`additivity_index`).

    Returns
    -------
    numpy.ndarray
        RE per unit.

    Notes
    -----
    TODO(science): ``RE = (r_multi - max(r_visual, r_prop)) / max(r_visual, r_prop)``.
    """
    raise NotImplementedError("TODO(science): response-enhancement formula")


def response_additivity(
    r_visual: FloatArray, r_prop: FloatArray, r_multi: FloatArray
) -> FloatArray:
    """Response additivity (RA): multisensory vs the linear sum.

    Parameters
    ----------
    r_visual, r_prop, r_multi
        Unisensory and multisensory responses.

    Returns
    -------
    numpy.ndarray
        RA per unit.

    Notes
    -----
    TODO(science): the project's RA definition (e.g. ``r_multi / (r_visual +
    r_prop)``); RA = 1 is exact additivity.
    """
    raise NotImplementedError("TODO(science): response-additivity formula")


def gain_index(r_high_reliability: FloatArray, r_low_reliability: FloatArray) -> FloatArray:
    """Gain index: response change from low- to high-reliability stimulation.

    Parameters
    ----------
    r_high_reliability
        Responses under high reliability (low variance), shape ``(n_units,)``.
    r_low_reliability
        Responses under low reliability (high variance), same shape.

    Returns
    -------
    numpy.ndarray
        Gain index per unit.

    Notes
    -----
    TODO(science): ``gain_index = (r_high - r_low) / (r_high + r_low)`` (or the
    project's normalisation). Links to the reliability-gain modulation in encoding.
    """
    raise NotImplementedError("TODO(science): gain-index formula")
