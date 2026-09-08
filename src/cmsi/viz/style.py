"""One place for figure defaults, so every panel in the manuscript matches.

Figure functions return a matplotlib Figure and never save or show; the caller
decides where things go. `save_figures` writes a whole group at once.

Everything here is pinned to the PLOS ONE figure specification
(journals.plos.org/plosone/s/figures), because that is where the figures are
going:

    width      2.63 - 7.5 in   (789 - 2250 px at 300 dpi)
               <= 5.2 in aligns with the text column; 7.5 in is full width
    height     <= 8.75 in      (2625 px)
    resolution 300 - 600 dpi
    fonts      Arial, Times or Symbol only, 8 - 12 pt AT FINAL SIZE
    format     TIFF, flattened, no alpha channel, LZW; RGB 8-bit; < 10 MB

Every figure is also written as SVG -- not for the journal, which does not take
it, but for you: text stays text (editable in Inkscape or Illustrator, and it
renders in Arial on a machine that has it), lines and axes stay vectors, and
the dense point clouds are embedded as 300-dpi images so a 50,000-trial scatter
does not become a 50,000-element file. Output is deterministic (no timestamp,
fixed element ids) so a regenerated SVG diffs cleanly.

The point that decides everything else: figures are drawn at their FINAL
printed size. A 5.2-inch-wide figure with 10 pt text is submitted as a
5.2-inch-wide file with 10 pt text -- nothing is scaled at layout, so the font
sizes in `apply_style` are literal, not nominal. Drawing at 11 inches and
letting the journal shrink it would turn 10 pt into 7 pt, below the floor.

Arial is not installed on Linux; `Liberation Sans` is metrically identical to
it, so a figure rendered on a Linux box lays out exactly as it will on the Mac
that has Arial. The stack below tries Arial first and falls through.
"""

import io
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")   # scripts run headless; notebooks override this themselves
import matplotlib.pyplot as plt  # noqa: E402

# ---- PLOS ONE limits -------------------------------------------------------- #
DPI = 300
WIDTH_TEXT = 5.2        # single-panel figures: aligns with the text column
WIDTH_FULL = 7.5        # multi-panel figures: the maximum
WIDTH_MIN = 2.63
HEIGHT_MAX = 8.75
MAX_BYTES = 10 * 1024 * 1024
BORDER_IN = 2 / 72      # the recommended 2-pt white border, in inches

# Common figure sizes, (width, height) in inches. Named so a figure function
# says what kind of figure it is rather than repeating numbers.
SIZE = {
    "single":   (WIDTH_TEXT, 3.9),      # one panel
    "single_tall": (WIDTH_TEXT, 4.4),   # one panel, square-ish scatter
    "pair":     (WIDTH_FULL, 3.5),      # 1 x 2
    "triple":   (WIDTH_FULL, 2.9),      # 1 x 3
    "quad":     (WIDTH_FULL, 6.4),      # 2 x 2
    "quad_short": (WIDTH_FULL, 5.6),    # 2 x 2 of histograms
    "composite": (WIDTH_FULL, 6.0),     # the three-panel sweep figure
}

FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

COLORS = {
    "network": "#1f77b4",
    "analytical": "#111111",
    "visual": "#d62728",
    "prop": "#2ca02c",
    "eye": "#9467bd",
}


def apply_style():
    """PLOS-compliant defaults. Call once at the top of a script."""
    plt.rcParams.update({
        "figure.dpi": 110,                 # screen only; files use savefig.dpi
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": BORDER_IN,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        # fonts: Arial (or its metric twin), every size inside 8-12 pt
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        # math text in the same face, so $p_{common}$ is not set in DejaVu
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans",           # default is 'cursive', absent on Linux
        "mathtext.sf": "sans",
        # embed TrueType in vector output (PLOS EPS rule; harmless for PDF)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # SVG: keep text as <text> (editable, searchable) rather than outlines,
        # and hash element ids from a fixed salt so output is reproducible
        "svg.fonttype": "none",
        "svg.hashsalt": "cmsi",
        # geometry: lines heavy enough to survive print at 300 dpi
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def label_panels(axes, letters="ABCDEFGH", dx=-30, dy=4):
    """Bold 12-pt panel letters, top-left of each axes, offset in POINTS so
    the placement does not depend on the axes size. PLOS requires lettered
    labels on every multi-panel figure."""
    for ax, letter in zip(np.ravel(axes), letters, strict=False):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=12, fontweight="bold", ha="left", va="bottom",
                    annotation_clip=False)


# ---- export ------------------------------------------------------------------ #
def _to_flat_rgb(fig):
    """Render at DPI and return a flattened RGB PIL image (no alpha channel).

    matplotlib's own TIFF writer emits RGBA, which PLOS rejects. Going through
    a PNG buffer and compositing onto white gives a file that is exactly what
    the PNG shows, with the alpha channel gone.
    """
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=im.getchannel("A"))
    return flat


def save_tiff(fig, path):
    """PLOS ONE TIFF: RGB, flattened, LZW, 300 dpi metadata."""
    _to_flat_rgb(fig).save(path, format="TIFF", compression="tiff_lzw",
                           dpi=(DPI, DPI))
    return path


_FONT_STACK_CSS = ", ".join(f"'{f}'" for f in FONT_STACK) + ", sans-serif"


def save_svg(fig, path):
    """Vector SVG with text kept as text and no timestamp.

    matplotlib writes the full font stack on ordinary text but only the font it
    resolved (e.g. 'Liberation Sans' on Linux) on math text; the post-process
    below gives every element the same stack, so the file renders in Arial on
    a machine that has Arial, whichever machine drew it. `rasterized=True`
    artists are embedded as images at DPI.
    """
    import re
    path = Path(path)
    fig.savefig(path, format="svg", dpi=DPI, metadata={"Date": None})
    text = path.read_text(encoding="utf-8")
    resolved = "|".join(re.escape(f) for f in FONT_STACK)
    text = re.sub(rf"font-family: ?'({resolved})'(?=[;\"])",
                  f"font-family: {_FONT_STACK_CSS}", text)
    # browsers and Inkscape collapse runs of spaces in <text> unless told not
    # to, which would turn the "A   title" panel headings into "A title"
    text = text.replace("<svg ", '<svg xml:space="preserve" ', 1)
    path.write_text(text, encoding="utf-8")
    return path


def _svg_size(path):
    """(width_in, height_in) from the root element's pt attributes."""
    import re
    head = Path(path).read_text(encoding="utf-8")[:2000]
    m = re.search(r'<svg[^>]*\swidth="([\d.]+)pt"[^>]*\sheight="([\d.]+)pt"', head)
    if m is None:
        return None, None
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def check_plos(path):
    """Read a written figure back and test it against the PLOS ONE limits.

    Returns a dict with the measured size and a list of problems (empty when
    the file complies). Used by `save_figures` to warn, and by the tests.
    Raster files are measured in pixels at 300 dpi; an SVG is measured from
    its declared size in points (the journal does not take SVG, but the file
    should still be the size the raster is).
    """
    from PIL import Image
    path = Path(path)
    problems = []
    if path.suffix.lower() == ".svg":
        w_in, h_in = _svg_size(path)
        if w_in is None:
            problems.append("no width/height on the <svg> root")
            w_in = h_in = float("nan")
        w_px, h_px = round(w_in * DPI), round(h_in * DPI)
        dpi, mode, compression = None, "vector", None
    else:
        with Image.open(path) as im:
            dpi = im.info.get("dpi", (None, None))[0]
            w_px, h_px = im.size
            mode = im.mode
            compression = im.info.get("compression")
        w_in = w_px / DPI
        h_in = h_px / DPI
        if dpi is None or abs(dpi - DPI) > 1:
            problems.append(f"dpi metadata {dpi} != {DPI}")
    if w_in < WIDTH_MIN - 0.01:
        problems.append(f"width {w_in:.2f} in < {WIDTH_MIN}")
    if w_in > WIDTH_FULL + 0.01:
        problems.append(f"width {w_in:.2f} in > {WIDTH_FULL}")
    if h_in > HEIGHT_MAX + 0.01:
        problems.append(f"height {h_in:.2f} in > {HEIGHT_MAX}")
    if path.suffix.lower() in (".tif", ".tiff"):
        if mode != "RGB":
            problems.append(f"mode {mode}, PLOS wants flattened RGB")
        if compression != "tiff_lzw":
            problems.append(f"compression {compression}, not LZW")
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"file {size / 2**20:.1f} MB > 10 MB")
    return {"path": str(path), "width_px": w_px, "height_px": h_px,
            "width_in": round(w_in, 2), "height_in": round(h_in, 2),
            "dpi": dpi, "mode": mode, "bytes": size, "problems": problems}


FORMATS = ("png", "tif", "svg")


def save_figures(figs, outdir, formats=FORMATS, close=True, check=True):
    """figs: {name: Figure}. Writes {outdir}/{name}.{fmt} for each format.

    "png" is the working copy (300 dpi, for the guide and for looking at);
    "tif" is the submission copy (flattened RGB, LZW); "svg" is the editable
    vector copy; "pdf" is vector too, for a LaTeX draft. Every png/tif/svg is
    read back and checked against the PLOS limits; a violation is a warning,
    not an error, so a figure that runs a little wide still gets written and
    the message says by how much. Returns the paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        for fmt in formats:
            path = outdir / f"{name}.{fmt}"
            if fmt in ("tif", "tiff"):
                save_tiff(fig, path)
            elif fmt == "svg":
                save_svg(fig, path)
            else:
                fig.savefig(path, dpi=DPI)
            paths.append(path)
            if check and fmt in ("png", "tif", "tiff", "svg"):
                rep = check_plos(path)
                for p in rep["problems"]:
                    warnings.warn(f"{path.name}: {p}", stacklevel=2)
        if close:
            plt.close(fig)
    return paths
