"""Side experiments that live outside the main pipeline.

Everything here writes under results/experiments/<experiment>/ and never
touches results/<run>/, so a configuration can be tried freely; once one is
chosen it goes into configs/ and the whole pipeline is re-run through the
numbered stages.

    implied_weight   the implied fusion weight (the hybrid read of each
                     channel) on any pipeline run: the per-run figures, the
                     weight at many reliability levels of each input, and
                     `train`, which takes a new configuration through the
                     whole pipeline with its p_common = 1 control
                     (scripts/06_implied_weight.py)
"""

from cmsi.experiments import implied_weight

__all__ = ["implied_weight"]
