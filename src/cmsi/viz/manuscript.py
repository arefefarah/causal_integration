"""THE STANDARD for every panel in the manuscript, in one place.

A panel is a cell of fixed size whose axes sit inside fixed ABSOLUTE margins
(PANEL_MARGIN, in inches), with its text at PANEL_FONT and 12-pt bold panel
letters (`label_panels`). Three cell widths share one height and one set of
margins:

    CELL_SQUARE   2.5  x 2.5 in   axes 1.84 x 1.75    the default
    CELL_WIDE     3.75 x 2.5 in   axes 3.09 x 1.75    rectangular, two to a row
    CELL_FULL     7.5  x 2.5 in   axes 6.84 x 1.75    one panel across the page

Every manuscript figure is an arrangement of cells, so its size follows from
its layout: 1 x 3 squares, 1 x 2 wides or 1 full = 7.5 x 2.5 in; 2 x 2 squares
= 5.0 x 5.0 in; a full over two wides = 7.5 x 5.0 in. Any two panels, in any
two figures, therefore have the same margins and fonts by construction rather
than by adjustment, and their axes line up when figures are stacked. The
margins are sized for the widest labels any panel draws ("2000" on a y axis,
a 9-pt title) and are the same for every cell whether or not a given panel
needs them. Fonts are inside the PLOS 8-12 pt window; superscripts are Unicode
glyphs, never mathtext, which would set them at 70 % of the label size.

Files are saved at exactly their frame (`exact_frame`), never cropped to
content, so panels tile edge to edge. Every manuscript figure gets its own
folder under `manuscript_dir()`, holding every format of it.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from cmsi.utils.paths import RESULTS
from cmsi.viz.style import SIZE, exact_frame, label_panels

CELL_SQUARE = SIZE["third"]                                 # (2.5, 2.5) in
CELL_WIDE = (SIZE["third"][0] * 1.5, SIZE["third"][1])      # (3.75, 2.5) in
CELL_FULL = (SIZE["third"][0] * 3, SIZE["third"][1])        # (7.5, 2.5) in
PANEL_MARGIN = {"left": 0.5625, "bottom": 0.4375, "right": 0.1, "top": 0.3125}
PANEL_FONT = {"tick": 8, "label": 9, "title": 9, "legend": 8}
PANEL_LW, PANEL_MS = 1.2, 3.0
LETTER_OFFSET = {"dx": -34, "dy": 2}                        # points, from axes top-left


def panel_rect(cell=CELL_SQUARE):
    """(left, bottom, width, height) in figure fractions of a cell, from the
    absolute margins. For the square cell this is (0.225, 0.175, 0.735, 0.70)."""
    w, h = cell
    m = PANEL_MARGIN
    return (m["left"] / w, m["bottom"] / h,
            (w - m["left"] - m["right"]) / w, (h - m["bottom"] - m["top"]) / h)


PANEL_RECT = panel_rect(CELL_SQUARE)                        # the square case, by name


def cell_axes(fig, cell, origin=(0.0, 0.0), reserve_right=0.0):
    """Add an axes for a `cell`-sized panel whose bottom-left corner is at
    `origin` (inches) in `fig`. `reserve_right` (inches) narrows the axes to
    leave room inside the cell for a colourbar."""
    W, H = fig.get_size_inches()
    m = PANEL_MARGIN
    x0, y0 = origin
    left = x0 + m["left"]
    bottom = y0 + m["bottom"]
    width = cell[0] - m["left"] - m["right"] - reserve_right
    height = cell[1] - m["bottom"] - m["top"]
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    ax.tick_params(labelsize=PANEL_FONT["tick"])
    return ax


def panel_axes(ax=None, cell=CELL_SQUARE):
    """A fixed-frame single-cell figure with its axes at panel_rect(cell), or
    the axes handed in (a composed figure supplies its own)."""
    if ax is None:
        fig = plt.figure(figsize=cell)
        ax = fig.add_axes(panel_rect(cell))
        exact_frame(fig, check=False)       # a component, not a submission file
    ax.tick_params(labelsize=PANEL_FONT["tick"])
    return ax


def finish_panel(ax, xlabel, ylabel, title, legend_loc=None):
    ax.set_xlabel(xlabel, fontsize=PANEL_FONT["label"])
    ax.set_ylabel(ylabel, fontsize=PANEL_FONT["label"])
    ax.set_title(title, fontsize=PANEL_FONT["title"])
    if legend_loc:
        ax.legend(fontsize=PANEL_FONT["legend"], loc=legend_loc,
                  handlelength=1.6, borderaxespad=0.4)
    return ax.figure


def letter(axes, letters="ABCDEFGH"):
    """Standard panel letters on the given axes, in order."""
    label_panels(axes, letters=letters, **LETTER_OFFSET)


def manuscript_grid(draw, ncols, letters="ABCDEFGH", cell=CELL_SQUARE):
    """Standard panels tiled into an ncols-wide grid, lettered in reading order.

    `draw` is a list of callables taking `ax`. Every cell is one `cell`-sized
    panel with its axes at panel_rect(cell), so the figure is (cell width *
    ncols) by (cell height * nrows) inches and each axes is exactly the size
    and position it would have in a standalone panel file. Row-major order:
    A B / C D.
    """
    ncols = int(ncols)
    nrows = int(np.ceil(len(draw) / ncols))
    cw, ch = cell
    fig = plt.figure(figsize=(cw * ncols, ch * nrows))
    axes = []
    for i, fn in enumerate(draw):
        r, c = divmod(i, ncols)
        ax = cell_axes(fig, cell, origin=(c * cw, (nrows - 1 - r) * ch))
        fn(ax)
        axes.append(ax)
    letter(axes, letters)
    exact_frame(fig)
    return fig


def manuscript_dir(run="flagship"):
    """Where manuscript figures go: results/manuscript/ for the flagship run
    (the one the paper reports), results/manuscript_<run>/ for any other, so
    re-rendering a satellite never overwrites the paper's figures."""
    return RESULTS / ("manuscript" if run == "flagship" else f"manuscript_{run}")


# --------------------------------------------------------------------------- #
# The manuscript's figures, numbered as in the paper
# --------------------------------------------------------------------------- #
# One folder per figure under manuscript_dir(), named fig<N>_<content>; the
# composed figure inside carries the same name (png, tif, svg) and
# `save_figures` drops a flat copy of its png beside the folder, so the
# manuscript's media/ can be filled from one listing. Panels are A_..., B_...
#
# Renumber or rename HERE and nowhere else: the code refers to a figure by
# what it shows (FIG_WEIGHT, FIG_UNITS), never by its number, so moving a
# figure in the paper is a one-line change. Who writes each:
#
#   fig1   task, generative model, encoding, network   drawn by hand (not rendered
#                                                       by the pipeline; see GUIDE 9.4)
#   fig2   output scatter 2 x 2                          04_figures.py (results.manuscript_panels)
#   fig3   error histograms 2 x 2                        04_figures.py
#   fig4   fusion weight A-C                             04_figures.py
#   fig5   decision bias                                 scripts/neuronal/run.py
#   fig6   variance hump                                 04_figures.py
#   fig7   prior sweep                                   05_prior_sweep.py
#   fig8   model comparison                              04_figures.py
#   fig9   decoding + RF shift                           04_figures.py
#   fig10  units                                         scripts/neuronal/run.py
#   fig11  lesion behaviour                              scripts/neuronal/run.py
FIG_TASK = "fig1_task_model_network"
FIG_SCATTER = "fig2_output_scatter"
FIG_ERRORS = "fig3_error_histograms"
FIG_WEIGHT = "fig4_fusion_weight"
FIG_BIAS = "fig5_decision_bias"
FIG_HUMP = "fig6_variance_hump"
FIG_SWEEP = "fig7_prior_sweep"
FIG_MODEL = "fig8_model_comparison"
FIG_DECODING = "fig9_decoding_rf_shift"
FIG_UNITS = "fig10_units"
FIG_LESION = "fig11_lesion_behaviour"
FIGURES = (FIG_TASK, FIG_SCATTER, FIG_ERRORS, FIG_WEIGHT, FIG_BIAS, FIG_HUMP,
           FIG_SWEEP, FIG_MODEL, FIG_DECODING, FIG_UNITS, FIG_LESION)


def figure_number(name):
    """The paper's number of a figure folder name ('fig9_decoding_rf_shift' -> 9)."""
    return int(name.split("_", 1)[0][3:])


def folder_of(path):
    """The manuscript figure folder a saved file belongs to (its parent)."""
    return Path(path).parent.name
