r"""Bayes-factor mechanism analysis.

Computes the analytical log Bayes factor per trial, inverts network output 5
(given the prior ``p_common``) to an IMPLIED log-BF, and compares the two. Also
provides hooks to ask which sub-population's activity tracks log-BF / body-frame
disparity.

Inversion and decoding plumbing are implemented; the closed-form inversion is
``TODO(science)`` (it mirrors the posterior in :mod:`causal_msi.generative`).
Units: log-BF in nats, disparity in deg.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from causal_msi.analysis._decoders import DecodeResult, fit_linear_decoder
from causal_msi.config import DecoderConfig

FloatArray = NDArray[np.float64]


def implied_log_bf(pred_pc: FloatArray, p_common: float) -> FloatArray:
    """Invert the network ``p(C=1)`` to the implied log Bayes factor.

    Parameters
    ----------
    pred_pc
        Network common-cause probability, shape ``(N,)`` in ``(0, 1)``.
    p_common
        Prior ``P(C=1)`` used by the observer.

    Returns
    -------
    numpy.ndarray
        Implied log Bayes factor per trial, shape ``(N,)`` (nats).

    Notes
    -----
    TODO(science): invert ``p = BF*pc / (BF*pc + (1-pc))`` for ``log BF``
        ``log BF = logit(p) - logit(p_common)``
    i.e. ``log(p/(1-p)) - log(p_common/(1-p_common))``. Should recover the
    analytical log-BF up to network error. Tested in
    ``tests/test_integration.py`` (round-trip with the posterior).
    """
    raise NotImplementedError("TODO(science): invert p(C=1) -> implied log Bayes factor")


def compare_log_bf(analytical_log_bf: FloatArray, implied: FloatArray) -> dict[str, float]:
    """Compare analytical and network-implied log Bayes factors.

    Parameters
    ----------
    analytical_log_bf
        Analytical log-BF, shape ``(N,)`` (nats).
    implied
        Network-implied log-BF, shape ``(N,)`` (nats).

    Returns
    -------
    dict
        ``{"slope", "r2", "rmse"}`` of implied vs analytical.
    """
    from sklearn.metrics import r2_score

    slope = float(np.polyfit(analytical_log_bf, implied, 1)[0])
    return {
        "slope": slope,
        "r2": float(r2_score(analytical_log_bf, implied)),
        "rmse": float(np.sqrt(np.mean((analytical_log_bf - implied) ** 2))),
    }


def population_tracks_log_bf(
    activations: FloatArray,
    analytical_log_bf: FloatArray,
    cfg: DecoderConfig,
    seed: int = 0,
) -> DecodeResult:
    """Decode the analytical log Bayes factor from a layer's activations.

    Parameters
    ----------
    activations
        Layer activations, shape ``(N, h)``.
    analytical_log_bf
        Analytical log-BF, shape ``(N,)`` (nats).
    cfg
        Decoder configuration.
    seed
        Split seed.

    Returns
    -------
    DecodeResult
        Held-out decodability of log-BF (which sub-population tracks it).
    """
    return fit_linear_decoder(activations, analytical_log_bf, cfg, seed)
