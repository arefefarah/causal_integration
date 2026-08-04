"""One place for figure defaults, so every panel in the thesis matches.

Figure functions return a matplotlib Figure and never save or show; the caller
decides where things go. `save_figures` writes a whole group at once.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # scripts run headless; notebooks override this themselves
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150

COLORS = {
    "network": "#1f77b4",
    "analytical": "#111111",
    "visual": "#d62728",
    "prop": "#2ca02c",
    "eye": "#9467bd",
}


def apply_style():
    """Sensible publication-ish defaults. Call once at the top of a script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def save_figures(figs, outdir, close=True):
    """figs: {name: Figure}. Writes {outdir}/{name}.png. Returns the paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=DPI)
        paths.append(path)
        if close:
            plt.close(fig)
    return paths
