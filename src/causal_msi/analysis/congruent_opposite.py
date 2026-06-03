r"""Congruent vs opposite MSL units.

Classifies multisensory-layer units by the sign of their cross-modal tuning
(congruent = same-sign tuning to vision and proprioception; opposite =
opposite-sign), then tests whether the congruent/opposite activity balance
predicts the causal read-out, and cross-references units that show opposite RF
shifts with eye position.

Tuning-slope estimation by linear regression is implemented; the sign-based
classification threshold and the balance->readout link are ``TODO(science)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CrossModalTuning:
    """Per-unit cross-modal tuning slopes."""

    visual_slope: FloatArray
    prop_slope: FloatArray


def cross_modal_tuning(
    msl_activations: FloatArray,
    x_vis_body: FloatArray,
    x_prop: FloatArray,
) -> CrossModalTuning:
    """Estimate per-unit tuning slopes to the visual and proprioceptive locations.

    Parameters
    ----------
    msl_activations
        MSL activations, shape ``(N, h2)``.
    x_vis_body
        Body-frame visual measurement, shape ``(N,)`` (deg).
    x_prop
        Proprioceptive measurement, shape ``(N,)`` (deg).

    Returns
    -------
    CrossModalTuning
        Per-unit regression slopes against each modality, each shape ``(h2,)``.
    """
    h2 = msl_activations.shape[1]
    vis_slope = np.empty(h2)
    prop_slope = np.empty(h2)
    for j in range(h2):
        vis_slope[j] = np.polyfit(x_vis_body, msl_activations[:, j], 1)[0]
        prop_slope[j] = np.polyfit(x_prop, msl_activations[:, j], 1)[0]
    return CrossModalTuning(visual_slope=vis_slope, prop_slope=prop_slope)


def classify_units(tuning: CrossModalTuning, min_abs_slope: float = 0.0) -> dict[str, BoolArray]:
    """Classify MSL units as congruent or opposite by tuning-slope signs.

    Parameters
    ----------
    tuning
        Per-unit cross-modal tuning slopes.
    min_abs_slope
        Minimum absolute slope (both modalities) to consider a unit tuned.

    Returns
    -------
    dict
        ``{"congruent": mask, "opposite": mask, "untuned": mask}`` boolean arrays
        of shape ``(h2,)``.

    Notes
    -----
    TODO(science): define the classification rule -- congruent = ``sign(visual) ==
    sign(prop)`` (same-sign tuning), opposite = opposite signs, both above
    ``min_abs_slope``; units below threshold are untuned. Tested in the
    congruent/opposite tests.
    """
    raise NotImplementedError("TODO(science): congruent/opposite classification rule")


def balance_predicts_readout(
    msl_activations: FloatArray,
    congruent: BoolArray,
    opposite: BoolArray,
    pred_pc: FloatArray,
) -> dict[str, float]:
    """Test whether the congruent/opposite activity balance predicts ``p(C=1)``.

    Parameters
    ----------
    msl_activations
        MSL activations, shape ``(N, h2)``.
    congruent, opposite
        Unit masks from :func:`classify_units`, each shape ``(h2,)``.
    pred_pc
        Network common-cause output, shape ``(N,)``.

    Returns
    -------
    dict
        Association statistics between the balance index and ``p(C=1)``.

    Notes
    -----
    TODO(science): define a balance index (e.g. mean congruent activity minus mean
    opposite activity, or their ratio) and correlate it with ``p(C=1)``; the
    hypothesis is that opposite units encode disparity/segregation evidence while
    congruent units encode fusion evidence.
    """
    raise NotImplementedError("TODO(science): congruent/opposite balance -> readout link")
