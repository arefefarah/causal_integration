r"""The additive feedforward network.

Architecture (matching the prior multisensory-integration model):

    input -> SIL (sensory-integration layer, 64, sigmoid)
          -> MSL (multisensory layer, 64, sigmoid)
          -> linear read-out head

Two heads are supported:

- ``head_type="causal"`` (4 outputs): raw read-out
  ``[mu_vis_raw, var_vis_raw, mu_prop_raw, var_prop_raw]`` is mapped to
  ``[mu_vis, var_vis, mu_prop, var_prop]`` by applying *softplus* to the two
  variance outputs and identity to the two means. These four targets are the
  Bayesian causal-inference *optimal* position estimates of Kording et al. (2007):
  the model-averaged visual/proprioceptive estimates (Eqs. 9/10) and their
  posterior variances. ``p(C=1)`` is computed internally by the analytical
  observer and is NEITHER an input NOR an output.

- ``head_type="integration_only"`` (2 outputs): a Project-1-style always-fuse
  control twin producing ``[mu_fused, raw_var]`` (softplus on the variance), i.e.
  the C=1 forced-fusion estimate (Eq. 12). Used for the emergent-vs-imposed
  analysis.

The named hidden layers (SIL, MSL) are exposed via forward hooks so analyses can
read their activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_msi.config import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

_HEAD_DIM: dict[str, int] = {"causal": 4, "integration_only": 2}


@dataclass
class ForwardCache:
    """Activations captured during a forward pass.

    Attributes
    ----------
    sil
        Sensory-integration-layer activations, shape ``(N, hidden_sizes[0])``.
    msl
        Multisensory-layer activations, shape ``(N, hidden_sizes[1])``. For deeper
        networks this is the last hidden layer.
    """

    sil: Tensor | None = None
    msl: Tensor | None = None


class FeedforwardMSI(nn.Module):
    """Additive feedforward network for causal multisensory inference.

    Parameters
    ----------
    input_dim
        Dimensionality of the concatenated population-code input.
    cfg
        Model configuration (hidden sizes, activation, head type).
    """

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cfg = cfg
        self.head_type = cfg.head_type
        self.output_dim = _HEAD_DIM[cfg.head_type]

        act_cls = _ACTIVATIONS[cfg.hidden_activation]

        layers: list[nn.Module] = []
        prev = input_dim
        self._hidden_linear: list[nn.Linear] = []
        for h in cfg.hidden_sizes:
            lin = nn.Linear(prev, h)
            layers.append(lin)
            layers.append(act_cls())
            self._hidden_linear.append(lin)
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.readout = nn.Linear(prev, self.output_dim)

        # Indices (into self.hidden) of the activation outputs we name SIL / MSL.
        # Each hidden block is [Linear, Activation]; the activation is at 2*i + 1.
        self._sil_idx = 1
        self._msl_idx = 2 * (len(cfg.hidden_sizes) - 1) + 1

    def _apply_head(self, raw: Tensor) -> Tensor:
        """Apply the output non-linearities for the configured head.

        Parameters
        ----------
        raw
            Linear read-out, shape ``(N, output_dim)``.

        Returns
        -------
        torch.Tensor
            Transformed outputs in the documented output order.
        """
        if self.head_type == "causal":
            mu_vis = raw[:, 0:1]
            var_vis = nn.functional.softplus(raw[:, 1:2])
            mu_prop = raw[:, 2:3]
            var_prop = nn.functional.softplus(raw[:, 3:4])
            return torch.cat([mu_vis, var_vis, mu_prop, var_prop], dim=1)
        # integration_only
        mu = raw[:, 0:1]
        var = nn.functional.softplus(raw[:, 1:2])
        return torch.cat([mu, var], dim=1)

    def forward(
        self, x: Tensor, return_cache: bool = False
    ) -> Tensor | tuple[Tensor, ForwardCache]:
        """Run a forward pass.

        Parameters
        ----------
        x
            Input batch, shape ``(N, input_dim)``.
        return_cache
            If True, also return a :class:`ForwardCache` with SIL/MSL activations.

        Returns
        -------
        torch.Tensor or tuple
            The head outputs of shape ``(N, output_dim)``, or ``(outputs, cache)``
            when ``return_cache`` is True.
        """
        cache = ForwardCache()
        h = x
        for i, module in enumerate(self.hidden):
            h = module(h)
            if return_cache and i == self._sil_idx:
                cache.sil = h
            if return_cache and i == self._msl_idx:
                cache.msl = h
        raw = self.readout(h)
        out = self._apply_head(raw)
        if return_cache:
            return out, cache
        return out

    @torch.no_grad()
    def activations(self, x: Tensor) -> ForwardCache:
        """Return SIL and MSL activations for ``x`` without tracking gradients.

        Parameters
        ----------
        x
            Input batch, shape ``(N, input_dim)``.

        Returns
        -------
        ForwardCache
            The captured hidden activations.
        """
        self.eval()
        _, cache = self.forward(x, return_cache=True)
        return cache


def build_model(input_dim: int, cfg: ModelConfig) -> FeedforwardMSI:
    """Build a :class:`FeedforwardMSI` model (factory).

    Parameters
    ----------
    input_dim
        Network input dimension.
    cfg
        Model configuration.

    Returns
    -------
    FeedforwardMSI
        An initialised model on the CPU (move with ``.to(device)``).
    """
    return FeedforwardMSI(input_dim, cfg)
