"""Side experiments that live outside the main pipeline.

Everything here writes under results/experiments/<experiment>/ and never
touches results/<run>/, so an architecture or an input encoding can be tried
freely; once one is chosen it goes into configs/ and the whole pipeline is
re-run through the numbered stages.

    implied_weight   how the implied fusion weight, sigma_out and sigma_w
                     depend on the read-out noise, the cue reliabilities and
                     the network (scripts/06_implied_weight.py)
"""

from cmsi.experiments import implied_weight

__all__ = ["implied_weight"]
