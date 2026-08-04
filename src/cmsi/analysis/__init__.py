"""Comparisons between the network and the analytical observer.

    accuracy   does the readout match the observer, per output
    causal     implied fusion weight, transition curves, decision strategy
    decoding   what the hidden layers carry, independent of the readout
    todo       placeholders for the analyses not yet written
"""

from cmsi.analysis.accuracy import accuracy, errors, generalization, print_accuracy
from cmsi.analysis.causal import (
    by_reliability,
    compare,
    fusion_weight,
    implied_log_bf,
    mean_by_bin,
    strategy_fit,
    transition_fit,
)
from cmsi.analysis.decoding import decode, decode_by_layer, summarise

__all__ = [
    "accuracy", "errors", "generalization", "print_accuracy",
    "fusion_weight", "implied_log_bf", "compare", "mean_by_bin",
    "transition_fit", "by_reliability", "strategy_fit",
    "decode", "decode_by_layer", "summarise",
]
