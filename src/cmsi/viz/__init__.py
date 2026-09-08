"""Figures, grouped the same way the results folder is.

    viz.inputs     what the network is shown   -> figures/inputs
    viz.training   did it converge             -> figures/training
    viz.results    network vs observer         -> figures/model

Every function returns a Figure and saves nothing; `style.save_figures` writes a
whole group at once.
"""

from cmsi.viz import inputs, prior_sweep, results, training
from cmsi.viz.style import (
    COLORS,
    DPI,
    FORMATS,
    SIZE,
    apply_style,
    check_plos,
    label_panels,
    save_figures,
    save_svg,
    save_tiff,
)

__all__ = ["inputs", "training", "results", "prior_sweep", "apply_style",
           "save_figures", "save_tiff", "save_svg", "check_plos", "label_panels",
           "COLORS", "DPI", "FORMATS", "SIZE"]
