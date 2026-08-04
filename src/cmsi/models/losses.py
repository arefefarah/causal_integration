"""The training objective.

Plain per-column MSE, optionally reweighted. The reweighting is not cosmetic:
the outputs live on different scales (means ~ +-10 deg, variances ~ 5-25 deg^2),
so an unweighted sum lets the large-scale columns dominate the gradient and the
shared trunk simply starves mu_prop -- even though a dedicated decoder reaches
R^2 ~ 0.97 on it.
"""

import torch


def output_weights(Y_train, balance=True):
    """1 / var(target) per output column, or None for an unweighted loss."""
    if not balance:
        return None
    return 1.0 / Y_train.var(0).clamp_min(1e-8)


def mse_loss(pred, target, weights=None):
    """Returns (total, per_output). per_output is detached, for logging."""
    per_output = ((pred - target) ** 2).mean(0)
    if weights is not None:
        per_output = per_output * weights
    return per_output.sum(), per_output.detach()


@torch.no_grad()
def evaluate(model, X, Y, weights=None):
    """Loss of a trained model on a batch of trials."""
    model.eval()
    total, per_output = mse_loss(model(X), Y, weights)
    return float(total), per_output.cpu().numpy()
