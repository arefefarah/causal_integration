"""Analysis suite for the trained causal-inference network.

Modules
-------
- :mod:`~causal_msi.analysis.performance`        per-output regression, calibration,
  proportion-common-cause vs disparity/reliability, generalization.
- :mod:`~causal_msi.analysis.decoding`           layer-wise decodability of p(C=1)
  (SIL vs MSL) + emergent-vs-imposed control (integration-only twin).
- :mod:`~causal_msi.analysis.congruent_opposite` congruent/opposite MSL units.
- :mod:`~causal_msi.analysis.bayes_factor`        analytical vs implied log Bayes factor.
- :mod:`~causal_msi.analysis.indices`             AI / RE / RA / gain indices.
- :mod:`~causal_msi.analysis.integration`         the sharper integration analysis.
- :mod:`~causal_msi.analysis.geometry`            RSA / dimensionality / RF-of-causality.
- :mod:`~causal_msi.analysis.rf_shifts`           RF shift vs eye position per layer.
- :mod:`~causal_msi.analysis.ablation`            opposite-unit silencing dissociation.

Scientific computations are left as ``TODO(science)`` with stated formulas and
matching failing tests; plumbing (decoder fitting, activation extraction) is
implemented.
"""

from __future__ import annotations

__all__ = [
    "ablation",
    "bayes_factor",
    "congruent_opposite",
    "decoding",
    "geometry",
    "indices",
    "integration",
    "performance",
]
