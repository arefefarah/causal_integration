"""Shared linear-decoder helpers (plumbing reused across analyses).

These wrap scikit-learn ridge regression / classification with a fixed train/test
split so analyses can report where a latent (e.g. the analytical ``p(C=1)`` or the
model-averaged estimate) is linearly decodable from a layer's activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from causal_msi.config import DecoderConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DecodeResult:
    """Result of fitting a linear decoder.

    Attributes
    ----------
    r2
        Held-out coefficient of determination.
    slope
        Slope of held-out predictions regressed on targets (1.0 = unbiased).
    coef
        Fitted decoder weights, shape ``(n_features,)``.
    predictions
        Held-out predictions, shape ``(n_test,)``.
    targets
        Held-out targets, shape ``(n_test,)``.
    """

    r2: float
    slope: float
    coef: FloatArray
    predictions: FloatArray
    targets: FloatArray


def fit_linear_decoder(
    features: FloatArray,
    target: FloatArray,
    cfg: DecoderConfig,
    seed: int = 0,
) -> DecodeResult:
    """Fit a ridge linear decoder from ``features`` to a scalar ``target``.

    Parameters
    ----------
    features
        Layer activations, shape ``(N, n_features)``.
    target
        Scalar regression target, shape ``(N,)``.
    cfg
        Decoder configuration (ridge alpha, held-out test size).
    seed
        Split seed for reproducibility.

    Returns
    -------
    DecodeResult
        Held-out R^2, fit slope, weights, and predictions.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, target, test_size=cfg.test_size, random_state=seed
    )
    model = Ridge(alpha=cfg.alpha)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    r2 = float(r2_score(y_te, pred))
    # Slope of predictions vs targets (unbiasedness check).
    slope = float(np.polyfit(y_te, pred, 1)[0]) if np.ptp(y_te) > 0 else float("nan")
    return DecodeResult(
        r2=r2, slope=slope, coef=np.asarray(model.coef_), predictions=pred, targets=y_te
    )
