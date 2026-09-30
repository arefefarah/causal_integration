"""Comparisons between the network and the analytical observer.

    accuracy   does the readout match the observer, per output
    causal     the implied fusion weight (the hybrid read of each channel),
               the position- and variance-domain regressions,
               reliability-within-disparity, variance signature, model comparison
    decoding   what the hidden layers carry, independent of the readout
    units      congruent/opposite classification, balance, lesion, RF shifts
    behavior   Kording-style bias curves and inferred-cause conditioning
    todo       placeholders for the analyses not yet written
"""

from cmsi.analysis.accuracy import accuracy, errors, generalization, print_accuracy
from cmsi.analysis.behavior import bias_vs_disparity, conditioned_bias
from cmsi.analysis.causal import (
    binned_weight,
    by_reliability,
    compare,
    hybrid_weight,
    implied_log_bf,
    mean_by_bin,
    mixture_variance,
    model_comparison,
    position_regression,
    reliability_within_disparity,
    sigma_out,
    strategy_fit,
    transition_fit,
    variance_regression,
    variance_signature,
    variance_weight,
    weight_by_posterior,
    weight_consistency,
    weight_regression,
)
from cmsi.analysis.decoding import decode, decode_by_layer, summarise
from cmsi.analysis.units import (
    balance,
    congruency,
    lesion,
    lesion_comparison,
    rf_shift,
    tuning_curves,
)

__all__ = [
    "accuracy", "errors", "generalization", "print_accuracy",
    "implied_log_bf", "compare", "mean_by_bin", "binned_weight", "weight_by_posterior",
    "transition_fit", "by_reliability", "strategy_fit",
    "position_regression", "sigma_out",
    "mixture_variance", "variance_weight", "hybrid_weight", "variance_regression",
    "weight_regression", "weight_consistency", "reliability_within_disparity",
    "variance_signature", "model_comparison",
    "decode", "decode_by_layer", "summarise",
    "congruency", "balance", "lesion", "lesion_comparison", "rf_shift",
    "tuning_curves",
    "bias_vs_disparity", "conditioned_bias",
]
