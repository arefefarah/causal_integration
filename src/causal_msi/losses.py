r"""Training objectives.

The causal head is trained with a weighted sum of:

- MSE on the two mean outputs (``mu_vis``, ``mu_prop``),
- MSE on the two (softplus) variance outputs (``var_vis``, ``var_prop``),
- BCE or MSE on the common-cause output ``p(C=1)`` against the GRADED analytical
  posterior (output 5 is regressed/BCE'd against ``p(C=1|x)``, not binary ``C``).

The integration-only twin is trained with MSE on its fused mean and variance.

Each loss returns the scalar total plus a dict of detached per-component values for
logging. Component weights come from :class:`causal_msi.config.LossWeights`.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from causal_msi.config import LossWeights

# Output-column indices for the causal head.
IDX_MU_VIS, IDX_VAR_VIS, IDX_MU_PROP, IDX_VAR_PROP, IDX_PC = 0, 1, 2, 3, 4


def causal_loss(
    pred: Tensor,
    target: Tensor,
    weights: LossWeights,
    pc_loss: str = "bce",
) -> tuple[Tensor, dict[str, float]]:
    """Weighted multi-output loss for the causal head.

    Parameters
    ----------
    pred
        Network outputs, shape ``(N, 5)`` in order
        ``[mu_vis, var_vis, mu_prop, var_prop, p_common]``.
    target
        Analytical targets, same shape and order. ``target[:, 4]`` is the graded
        posterior ``p(C=1|x)`` in ``[0, 1]``.
    weights
        Per-component weights (``est``, ``var``, ``pc``).
    pc_loss
        ``"bce"`` (binary cross-entropy against the graded posterior) or ``"mse"``.

    Returns
    -------
    tuple
        ``(total, components)`` where ``components`` maps
        ``{"est", "var", "pc", "total"}`` to detached floats for logging.
    """
    est = F.mse_loss(pred[:, [IDX_MU_VIS, IDX_MU_PROP]], target[:, [IDX_MU_VIS, IDX_MU_PROP]])
    var = F.mse_loss(pred[:, [IDX_VAR_VIS, IDX_VAR_PROP]], target[:, [IDX_VAR_VIS, IDX_VAR_PROP]])

    pc_pred = pred[:, IDX_PC]
    pc_tgt = target[:, IDX_PC]
    if pc_loss == "bce":
        # Both predicted and target are probabilities in [0, 1]; BCE on soft labels.
        pc = F.binary_cross_entropy(pc_pred.clamp(1e-6, 1 - 1e-6), pc_tgt)
    elif pc_loss == "mse":
        pc = F.mse_loss(pc_pred, pc_tgt)
    else:
        raise ValueError(f"unknown pc_loss {pc_loss!r}")

    total = weights.est * est + weights.var * var + weights.pc * pc
    components = {
        "est": float(est.detach()),
        "var": float(var.detach()),
        "pc": float(pc.detach()),
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
