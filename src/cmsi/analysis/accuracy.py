"""Does the readout match the analytical observer, output by output?"""

import numpy as np
from sklearn.metrics import r2_score


def accuracy(pred, target, names):
    """Per-output slope / R^2 / RMSE / bias, as a list of dicts.

    slope ~ 1 and bias ~ 0 mean unbiased. R^2 is scale-free but harsh on
    small-range outputs, so read RMSE (in deg or deg^2) alongside it.
    """
    rows = []
    for i, name in enumerate(names):
        y, yhat = target[:, i], pred[:, i]
        slope, intercept = np.polyfit(y, yhat, 1)
        err = yhat - y
        rows.append({
            "output": name,
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2_score(y, yhat)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(err.mean()),
        })
    return rows


def errors(pred, target, names):
    """Per-output error arrays (network - analytical)."""
    return {name: pred[:, i] - target[:, i] for i, name in enumerate(names)}


def generalization(pred, target, names, in_range):
    """Accuracy inside vs outside a mask -- e.g. trained-range disparities.

        in_range = np.abs(d["disparity"]) <= 20
    """
    return {"in_range": accuracy(pred[in_range], target[in_range], names),
            "out_of_range": accuracy(pred[~in_range], target[~in_range], names)}


def print_accuracy(rows, title=None):
    if title:
        print(f"\n{title}")
    print(f"{'output':>10} {'slope':>8} {'r2':>8} {'rmse':>8} {'bias':>8}")
    for r in rows:
        print(f"{r['output']:>10} {r['slope']:8.3f} {r['r2']:8.3f} "
              f"{r['rmse']:8.3f} {r['bias']:8.3f}")
