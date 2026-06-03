r"""Representational geometry of the MSL code.

Asks whether ``p(C=1)`` is an explicit low-dimensional axis or distributed, via
RSA and participation-ratio dimensionality, and compares representational geometry
across reference frames. The reference-frame-of-causality analysis asks in which
frame (retinal / body / intermediate) the causal read-out is invariant.

PCA / dimensionality / RDM plumbing is implemented; the RSA-significance and
invariance criteria are ``TODO(science)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def participation_ratio(activations: FloatArray) -> float:
    """Effective dimensionality (participation ratio) of an activation matrix.

    Parameters
    ----------
    activations
        Layer activations, shape ``(N, h)``.

    Returns
    -------
    float
        ``(sum lambda_i)^2 / sum(lambda_i^2)`` over the covariance eigenvalues
        ``lambda_i`` -- the participation ratio (1 = one dimension, h = isotropic).
    """
    centered = activations - activations.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    denom = float(np.sum(eig**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(eig) ** 2 / denom)


def representational_dissimilarity(
    activations: FloatArray, metric: str = "correlation"
) -> FloatArray:
    """Representational dissimilarity matrix (RDM) between condition-mean patterns.

    Parameters
    ----------
    activations
        Condition-mean activations, shape ``(n_conditions, h)``.
    metric
        Dissimilarity metric passed to :func:`scipy.spatial.distance.pdist`.

    Returns
    -------
    numpy.ndarray
        Square RDM, shape ``(n_conditions, n_conditions)``.
    """
    from scipy.spatial.distance import pdist, squareform

    rdm: FloatArray = squareform(pdist(activations, metric=metric))
    return rdm


def pc_axis_explicitness(activations: FloatArray, p_common: FloatArray) -> dict[str, float]:
    """Test whether ``p(C=1)`` lies on an explicit low-dimensional axis.

    Parameters
    ----------
    activations
        Layer activations, shape ``(N, h)``.
    p_common
        Analytical or network ``p(C=1)``, shape ``(N,)``.

    Returns
    -------
    dict
        Summary of how concentrated the ``p(C=1)`` encoding is (e.g. variance
        explained by the best single axis vs total decodable variance).

    Notes
    -----
    TODO(science): define "explicit low-dimensional" -- e.g. compare the fraction
    of ``p(C=1)`` variance captured by the top decoding axis to the participation
    ratio, and test against a distributed-code null.
    """
    raise NotImplementedError("TODO(science): explicit-vs-distributed p(C=1) axis criterion")


def reference_frame_of_causality(
    activations_by_frame: dict[str, FloatArray],
    pred_pc: FloatArray,
) -> dict[str, float]:
    """Identify the reference frame in which the causal read-out is invariant.

    Parameters
    ----------
    activations_by_frame
        Mapping frame name (``"retinal"``, ``"body"``, ``"intermediate"``) ->
        activation matrix ``(N, h)`` for matched conditions across eye positions.
    pred_pc
        Network common-cause output, shape ``(N,)``.

    Returns
    -------
    dict
        Per-frame invariance score of the causal read-out.

    Notes
    -----
    TODO(science): define the invariance metric -- in which frame does ``p(C=1)``
    (and its decoding axis) stay constant as eye position varies? Lower
    across-eye-position variance of the decoded ``p(C=1)`` in a frame = that frame
    is the reference frame of causality.
    """
    raise NotImplementedError("TODO(science): reference-frame-of-causality invariance metric")
