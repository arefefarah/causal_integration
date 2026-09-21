"""Side experiments that live outside the main pipeline.

Everything here writes under results/experiments/<experiment>/ and never
touches results/<run>/, so an architecture or an input encoding can be tried
freely; once one is chosen it goes into configs/ and the whole pipeline is
re-run through the numbered stages.

    implied_weight   how the implied fusion weight, sigma_out and sigma_w
                     depend on the read-out noise and the cue reliabilities,
                     on any pipeline run (scripts/06_implied_weight.py)
    design           what the trials of a configuration allow the per-trial
                     weight to show, before training, and the same measured
                     on trained runs side by side (06 design / compare)
    architecture     the same analysis across hidden-layer sizes, each
                     variant with its own control (scripts/07_architecture.py)
    fixed_variance   the same analysis with every input's noise pinned to one
                     value, swept over values (scripts/08_fixed_variance.py)
    sweep            the same analysis across the values of any one config
                     key, each variant with its own datasets and control
                     (scripts/09_sweep.py)
"""

from cmsi.experiments import architecture, design, fixed_variance, implied_weight, sweep

__all__ = ["implied_weight", "design", "architecture", "fixed_variance", "sweep"]
