"""What does the population carry, independent of the readout?

A separate linear decoder fitted from a hidden layer to some latent answers a
different question from the readout's accuracy: not "was it trained to output
this?" but "is it represented at all?". Running it on the always-fuse twin, which
was never asked for anything causal, is the emergent-vs-imposed test.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def decode(activations, target, alpha=1.0, test_size=0.25, seed=0):
    """Ridge-decode a scalar target from a layer. Held-out r2, slope, predictions.

    The returned predictions cover only the decoder's own test split, so they do
    not line up with the trials you passed in -- use `index` if you need to match
    them back up.
    """
    idx = np.arange(len(target))
    x_tr, x_te, y_tr, y_te, _, idx_te = train_test_split(
        activations, target, idx, test_size=test_size, random_state=seed
    )
    ridge = Ridge(alpha=alpha).fit(x_tr, y_tr)
    pred = ridge.predict(x_te)
    slope = float(np.polyfit(y_te, pred, 1)[0]) if np.ptp(y_te) > 0 else np.nan
    return {"r2": float(r2_score(y_te, pred)), "slope": slope,
            "coef": ridge.coef_, "pred": pred, "target": y_te, "index": idx_te}


def decode_by_layer(acts, target, alpha=1.0, test_size=0.25, seed=0):
    """Decode the same target from every hidden layer -> {"layer0": result, ...}.

    A latent that is weak in layer0 and strong in the last layer is being built
    by the network rather than handed to it by the input encoding.
    """
    return {name: decode(a, target, alpha, test_size, seed)
            for name, a in acts.items() if name.startswith("layer")}


def summarise(results):
    """Just the held-out R^2 per layer -- what goes into metrics.json."""
    return {name: float(r["r2"]) for name, r in results.items()}
