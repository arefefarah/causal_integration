r"""Receptive-field shift vs eye position, per unit and per layer.

Estimates how each unit's preferred (visual) location shifts as eye position
changes -- the signature of a reference-frame transform. A purely retinal unit's
body-frame preferred location shifts one-for-one with eye position; a purely
body-frame unit does not shift.

Tuning-peak estimation by binning is implemented; the shift-gain summary and its
frame interpretation are ``TODO(science)``. Units: deg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RFShift:
    """Per-unit receptive-field shift summary."""

    shift_gain: FloatArray  # d(preferred location) / d(eye position), shape (h,)
    preferred_by_eye: dict[float, FloatArray]  # eye level -> preferred loc per unit


def preferred_location_by_eye(
    activations: FloatArray,
    stimulus_location: FloatArray,
    eye_position: FloatArray,
    eye_levels: FloatArray,
    loc_grid: FloatArray,
) -> dict[float, FloatArray]:
    """Estimate each unit's preferred stimulus location at each eye-position level.

    Parameters
    ----------
    activations
        Layer activations, shape ``(N, h)``.
    stimulus_location
        Per-trial stimulus location (deg), shape ``(N,)``.
    eye_position
        Per-trial eye position (deg), shape ``(N,)``.
    eye_levels
        Eye-position levels to group by (deg).
    loc_grid
        Location-bin centres (deg) over which to find the tuning peak.

    Returns
    -------
    dict
        Mapping eye level -> per-unit preferred location array of shape ``(h,)``.
    """
    h = activations.shape[1]
    result: dict[float, FloatArray] = {}
    tol = (eye_levels[1] - eye_levels[0]) / 2 if len(eye_levels) > 1 else 1.0
    for level in eye_levels:
        sel = np.abs(eye_position - level) <= tol
        if not sel.any():
            result[float(level)] = np.full(h, np.nan)
            continue
        preferred = np.empty(h)
        loc = stimulus_location[sel]
        idx = np.clip(np.digitize(loc, loc_grid) - 1, 0, len(loc_grid) - 1)
        for j in range(h):
            tuning = np.array(
                [
                    activations[sel, j][idx == b].mean() if (idx == b).any() else -np.inf
                    for b in range(len(loc_grid))
                ]
            )
            preferred[j] = loc_grid[int(np.argmax(tuning))]
        result[float(level)] = preferred
    return result


def rf_shift_gain(preferred_by_eye: dict[float, FloatArray]) -> FloatArray:
    """Per-unit shift gain: slope of preferred location vs eye position.

    Parameters
    ----------
    preferred_by_eye
        Mapping eye level -> per-unit preferred location, from
        :func:`preferred_location_by_eye`.

    Returns
    -------
    numpy.ndarray
        Shift gain per unit, shape ``(h,)``.

    Notes
    -----
    TODO(science): regress each unit's preferred location on eye position; the
    slope is the shift gain. Interpret ~1 as retinal coding, ~0 as body-frame
    coding, intermediate as a partial transform. Aggregate per layer (SIL vs MSL)
    to show the retinal->body transition. Tested in the rf-shift tests.
    """
    raise NotImplementedError("TODO(science): RF shift-gain regression and frame interpretation")
