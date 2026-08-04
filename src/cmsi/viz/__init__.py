"""Figures, grouped the same way the results folder is.

    viz.inputs     what the network is shown   -> figures/inputs
    viz.training   did it converge             -> figures/training
    viz.results    network vs observer         -> figures/model

Every function returns a Figure and saves nothing; `style.save_figures` writes a
whole group at once.
"""

from cmsi.viz import inputs, results, training
from cmsi.viz.style import COLORS, apply_style, save_figures

__all__ = ["inputs", "training", "results", "apply_style", "save_figures", "COLORS"]
