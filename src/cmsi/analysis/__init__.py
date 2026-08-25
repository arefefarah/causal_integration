"""Comparisons between the network and the analytical observer.

    accuracy   does the readout match the observer, per output
    causal     implied fusion weight, position-domain regression, sigma_w,
               reliability-within-disparity, variance signature, model comparison
    decoding   what the hidden layers carry, independent of the readout
    units      congruent/opposite classification, balance, lesion, RF shifts
    behavior   Kording-style bias curves and inferred-cause conditioning
    todo       placeholders for the analyses not yet written
"""

from cmsi.analysis.accuracy import accuracy, errors, generalization, print_accuracy
from cmsi.analysis.behavior import bias_vs_disparity, conditioned_bias
from cmsi.analysis.causal import (
    by_reliability,
    compare,
    fusion_weight,
    implied_log_bf,
    joint_fusion_weight,
    mean_by_bin,
    model_comparison,
    position_regression,
    reliability_within_disparity,
    sigma_out,
    sigma_w,
    strategy_fit,
    transition_fit,
    variance_signature,
    weight_consistency,
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
    "fusion_weight", "implied_log_bf", "compare", "mean_by_bin",
    "transition_fit", "by_reliability", "strategy_fit",
    "position_regression", "sigma_out", "sigma_w", "joint_fusion_weight",
    "weight_consistency", "reliability_within_disparity",
    "variance_signature", "model_comparison",
    "decode", "decode_by_layer", "summarise",
    "congruency", "balance", "lesion", "lesion_comparison", "rf_shift",
    "tuning_curves",
    "bias_vs_disparity", "conditioned_bias",
]
