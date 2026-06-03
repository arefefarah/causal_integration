"""causal_msi: Bayesian causal inference across reference frames.

An additive feedforward neural network is trained to perform Bayesian causal
inference on population-coded sensory cues (visual hand position, proprioceptive
hand position) together with a proprioceptive eye-position signal. The network
estimates each cue's source and uncertainty (in a common body frame) and infers
the probability that the two cues share a common cause.

The package is organised as:

- :mod:`causal_msi.generative` -- generative model + analytical (Bayesian) observer.
- :mod:`causal_msi.encoding`   -- population-code encoders (network inputs).
- :mod:`causal_msi.models`     -- the feedforward network (causal / integration heads).
- :mod:`causal_msi.losses`     -- training objectives.
- :mod:`causal_msi.training`   -- training loop, early stopping, multi-seed runner.
- :mod:`causal_msi.analysis`   -- the analysis suite.

Units convention: all spatial quantities are in DEGREES; all variances are in
DEGREES^2 (deg^2).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
