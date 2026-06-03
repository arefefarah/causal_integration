r"""Layer-wise decodability of the analytical ``p(C=1)`` + emergent-vs-imposed control.

Two questions:

1. Where does the common-cause latent emerge? Fit a linear decoder of the
   analytical ``p(C=1|x)`` from the SIL activations and from the MSL activations
   and compare held-out R^2 across layers.
2. Emergent vs imposed: load the ``integration_only`` twin (never trained on
   ``p(C=1)``) and test whether ``p(C=1)`` is nonetheless linearly decodable from
   ITS MSL -- i.e. whether the causal latent emerges from the always-fuse task or
   is imposed only by the causal read-out objective.

Decoder fitting is implemented (ridge); the interpretive comparison is plumbing
plus a ``TODO(science)`` for the formal emergence criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from causal_msi.analysis._decoders import DecodeResult, fit_linear_decoder
from causal_msi.config import DecoderConfig
from causal_msi.models import FeedforwardMSI

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LayerDecoding:
    """Decodability of a target from each named layer."""

    sil: DecodeResult
    msl: DecodeResult


def extract_activations(model: FeedforwardMSI, x: FloatArray) -> dict[str, FloatArray]:
    """Run the model and return SIL/MSL activations as numpy arrays.

    Parameters
    ----------
    model
        Trained network.
    x
        Inputs, shape ``(N, input_dim)``.

    Returns
    -------
    dict
        ``{"sil": (N, h1), "msl": (N, h2)}`` activation matrices.
    """
    cache = model.activations(torch.as_tensor(np.asarray(x), dtype=torch.float32))
    assert cache.sil is not None and cache.msl is not None
    return {"sil": cache.sil.cpu().numpy(), "msl": cache.msl.cpu().numpy()}


def decode_pc_by_layer(
    model: FeedforwardMSI,
    x: FloatArray,
    analytical_pc: FloatArray,
    cfg: DecoderConfig,
    seed: int = 0,
) -> LayerDecoding:
    """Decode the analytical ``p(C=1)`` from SIL and MSL.

    Parameters
    ----------
    model
        Trained network (causal or integration-only twin).
    x
        Inputs, shape ``(N, input_dim)``.
    analytical_pc
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    cfg
        Decoder configuration.
    seed
        Split seed.

    Returns
    -------
    LayerDecoding
        Held-out decoding results for both layers.
    """
    acts = extract_activations(model, x)
    return LayerDecoding(
        sil=fit_linear_decoder(acts["sil"], analytical_pc, cfg, seed),
        msl=fit_linear_decoder(acts["msl"], analytical_pc, cfg, seed),
    )


def emergent_vs_imposed(
    causal_model: FeedforwardMSI,
    twin_model: FeedforwardMSI,
    x: FloatArray,
    analytical_pc: FloatArray,
    cfg: DecoderConfig,
    seed: int = 0,
) -> dict[str, LayerDecoding]:
    """Compare ``p(C=1)`` decodability between the causal model and the twin.

    Parameters
    ----------
    causal_model
        Model trained with the 5-output causal head.
    twin_model
        ``integration_only`` model never trained on ``p(C=1)``.
    x
        Shared inputs, shape ``(N, input_dim)``.
    analytical_pc
        Analytical ``p(C=1|x)``, shape ``(N,)``.
    cfg
        Decoder configuration.
    seed
        Split seed.

    Returns
    -------
    dict
        ``{"causal": LayerDecoding, "imposed_twin": LayerDecoding}``.

    Notes
    -----
    TODO(science): define the formal "emergence" criterion -- e.g. the twin's MSL
    R^2 exceeding a chance/null threshold (shuffled-label baseline) is evidence the
    latent emerges from the integration task rather than being imposed by the
    read-out. Add the null model here.
    """
    return {
        "causal": decode_pc_by_layer(causal_model, x, analytical_pc, cfg, seed),
        "imposed_twin": decode_pc_by_layer(twin_model, x, analytical_pc, cfg, seed),
    }
