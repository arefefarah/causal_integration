r"""Training objectives.

The causal head is trained with a weighted sum of:

- MSE on the two mean outputs (``mu_vis``, ``mu_prop``) -- the Kording (2007)
  model-averaged optimal estimates (Eqs. 9/10),
- MSE on the two (softplus) variance outputs (``var_vis``, ``var_prop``) -- the
  posterior variances of those estimates.

There is no ``p(C=1)`` term: the common-cause posterior is computed internally by
the analytical observer and never appears as an output.

The integration-only twin is trained with MSE on its fused mean and variance.

Each loss returns the scalar total plus a dict of detached per-component values for
logging. Component weights come from :class:`causal_msi.config.LossWeights`.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from causal_msi.config import LossWeights

# Output-column indices for the causal head.
IDX_MU_VIS, IDX_VAR_VIS, IDX_MU_PROP, IDX_VAR_PROP = 0, 1, 2, 3


def causal_loss(
    pred: Tensor,
    target: Tensor,
    weights: LossWeights,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted multi-output loss for the causal head.

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 4)`` in order
        ``[mu_vis, var_vis, mu_prop, var_prop]``.
    target
        Analytical targets, same shape and order (Kording Eqs. 9/10 + variances).
    weights
        Per-component weights (``est`` for the two means, ``var`` for the two
        variances).

    Returns
    -------
    tuple
        ``(total, components)`` where ``components`` maps
        ``{"est", "var", "total"}`` to detached floats for logging.
    """
    est = F.mse_loss(pred[:, [IDX_MU_VIS, IDX_MU_PROP]], target[:, [IDX_MU_VIS, IDX_MU_PROP]])
    var = F.mse_loss(pred[:, [IDX_VAR_VIS, IDX_VAR_PROP]], target[:, [IDX_VAR_VIS, IDX_VAR_PROP]])

    total = weights.est * est + weights.var * var
    components = {
        "est": float(est.detach()),
        "var": float(var.detach()),
        "total": float(total.detach()),
    }
    return total, components


def integration_loss(
    pred: Tensor,
    target: Tensor,
    weights: LossWeights,
) -> tuple[Tensor, dict[str, float]]:
    """Loss for the integration-only twin (always-fuse control).

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 2)`` = ``[mu_fused, var_fused]``.
    target
        Targets, same shape and order.
    weights
        Per-component weights; reuses ``est`` for the mean and ``var`` for the
        variance.

    Returns
    -------
    tuple
        ``(total, components)`` with ``{"est", "var", "total"}`` floats.
    """
    est = F.mse_loss(pred[:, 0], target[:, 0])
    var = F.mse_loss(pred[:, 1], target[:, 1])
    total = weights.est * est + weights.var * var
    components = {
        "est": float(est.detach()),
        "var": float(var.detach()),
        "total": float(total.detach()),
    }
    return total, components
