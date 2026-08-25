"""Behavioral signatures bridging to Kording et al. (2007) (design SS7.6).

`bias_vs_disparity` is the Fig. 2e analog: how far each report is pulled
toward the other cue as a function of the (signed) body-frame disparity.
Under model averaging the pull is w * (fused - seg), so it rises with
disparity while p(C=1) is high and collapses back toward zero as the
disparity itself argues for two causes -- the signature non-monotonic curve.

`conditioned_bias` is the Fig. 3b-c analog: split trials by the observer's
own inferred cause (implied weight or decoded posterior > 0.5) and look at
the residual bias in each branch. Conditioning on "inferred two causes"
selects trials whose noise happened to exaggerate the disparity, which
produces the counter-intuitive NEGATIVE bias (push away from the other cue)
that Kording report -- a fingerprint of inference over causal structure, not
of any fixed-weight scheme.
"""

import numpy as np

from cmsi.analysis.causal import mean_by_bin


def _drop_thin_bins(centres, means, counts, min_count):
    """Bins holding almost no trials swing wildly and read as structure.

    Conditioning on the inferred cause empties exactly the bins where the two
    branches are most interesting (few large-disparity trials are judged
    common), so this filter is what keeps the panel honest.
    """
    keep = counts >= min_count
    return centres[keep], means[keep], counts[keep]


def bias_vs_disparity(estimate, segregated, disparity, grid, prediction=None,
                      min_count=30):
    """Mean pull toward the other cue, binned by signed disparity.

    bias = estimate - segregated: what the report gains over the single-cue
    (segregation) solution. Pass prediction = w_opt * (fused - segregated) to
    overlay the Bayes-optimal curve. Bins with fewer than min_count trials are
    dropped. Returns dict of binned curves.
    """
    bias = np.asarray(estimate) - np.asarray(segregated)
    c_net, m_net, n_net = _drop_thin_bins(*mean_by_bin(disparity, bias, grid),
                                          min_count)
    out = {"centres": c_net, "bias_net": m_net, "count": n_net}
    if prediction is not None:
        c_opt, m_opt, n_opt = _drop_thin_bins(
            *mean_by_bin(disparity, np.asarray(prediction), grid), min_count)
        out["centres_opt"], out["bias_opt"] = c_opt, m_opt
    return out


def conditioned_bias(estimate, segregated, disparity, inferred_common, grid,
                     min_count=30):
    """Bias vs |disparity|, split by the network's own causal judgment.

    inferred_common: boolean per trial (implied weight or decoded posterior
    > 0.5). The C=2-judged branch is predicted to show the negative-bias /
    truncation effect at small-to-mid disparities.
    """
    bias = np.asarray(estimate) - np.asarray(segregated)
    ad = np.abs(np.asarray(disparity))
    inferred_common = np.asarray(inferred_common, bool)
    grid = np.unique(np.abs(np.asarray(grid, float)))
    out = {}
    for label, mask in (("common", inferred_common), ("separate", ~inferred_common)):
        if mask.sum() == 0:
            out[label] = {"centres": np.array([]), "bias": np.array([]),
                          "count": np.array([])}
            continue
        c, m, n = _drop_thin_bins(*mean_by_bin(ad[mask], bias[mask], grid),
                                  min_count)
        out[label] = {"centres": c, "bias": m, "count": n}
    seen = out["separate"]["bias"] < 0
    out["negative_bias_seen"] = bool(seen.any()) if seen.size else False
    return out
