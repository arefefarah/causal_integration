"""Figure export against the PLOS ONE specification, plus the SVG contract.

The journal's limits are numbers, so they are tested as numbers: every file
`save_figures` writes must be 789-2250 px wide and at most 2625 px tall at
300 dpi, and the TIFF must be flattened RGB with LZW compression. The SVG must
be the same physical size, keep its text as text, carry the Arial-first font
stack on every element, embed rasterized artists as images, and be
reproducible. These tests draw small synthetic figures; the real figures are
checked at write time by `save_figures` itself, which warns on a violation.
"""

import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from cmsi.viz import (
    DPI,
    FORMATS,
    SIZE,
    apply_style,
    check_plos,
    label_panels,
    save_figures,
)
from cmsi.viz.style import HEIGHT_MAX, WIDTH_FULL, WIDTH_MIN


@pytest.fixture(autouse=True)
def _style():
    apply_style()


def _fig(size):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    ax.plot(np.arange(10), np.arange(10) ** 2, "o-")
    ax.set(xlabel="x (deg)", ylabel="y (deg$^2$)", title="a title that stays")
    fig.tight_layout()
    return fig


def test_style_is_inside_the_plos_font_window():
    from matplotlib import pyplot as plt
    rc = plt.rcParams
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        assert 8 <= float(rc[key]) <= 12, key
    assert rc["savefig.dpi"] == DPI == 300
    assert rc["font.sans-serif"][0] == "Arial"


def test_named_sizes_are_inside_the_plos_envelope():
    for name, (w, h) in SIZE.items():
        if name == "third":
            # a component: three of them make one full-width figure
            assert abs(3 * w - WIDTH_FULL) < 1e-9
            continue
        assert WIDTH_MIN <= w <= WIDTH_FULL, name
        assert h <= HEIGHT_MAX, name


@pytest.mark.parametrize("kind", ["single", "pair", "quad", "composite"])
def test_saved_files_meet_the_limits(tmp_path, kind):
    paths = save_figures({"f": _fig(SIZE[kind])}, tmp_path)
    assert FORMATS == ("png", "tif", "svg")
    assert {p.suffix for p in paths} == {".png", ".tif", ".svg"}
    for p in paths:
        rep = check_plos(p)
        assert rep["problems"] == [], rep
        assert 789 <= rep["width_px"] <= 2250
        assert rep["height_px"] <= 2625
    # the svg declares the same physical size the raster has
    by = {p.suffix: check_plos(p) for p in paths}
    assert abs(by[".svg"]["width_in"] - by[".png"]["width_in"]) < 0.02
    assert abs(by[".svg"]["height_in"] - by[".png"]["height_in"]) < 0.02


def test_tiff_is_flattened_rgb_lzw_at_300_dpi(tmp_path):
    (tif,) = [p for p in save_figures({"f": _fig(SIZE["single"])}, tmp_path)
              if p.suffix == ".tif"]
    with Image.open(tif) as im:
        assert im.mode == "RGB"                       # no alpha channel
        assert im.info["compression"] == "tiff_lzw"
        assert im.info["dpi"] == (DPI, DPI)
        assert im.n_frames == 1                       # one page, no layers


def test_svg_keeps_text_as_text_with_the_font_stack(tmp_path):
    (svg,) = [p for p in save_figures({"f": _fig(SIZE["single"])}, tmp_path)
              if p.suffix == ".svg"]
    text = svg.read_text(encoding="utf-8")
    root = ET.fromstring(text)                                  # well-formed
    assert root.tag.endswith("svg")
    assert text.count("<text") >= 5                             # not outlines
    assert "a title that stays" in text                         # searchable
    families = set(re.findall(r"font-family: ?([^;\"]*)", text))
    assert families and all(f.startswith("'Arial'") for f in families), families
    assert 'xml:space="preserve"' in text[:500]
    # every text element keeps its spaces, in both the attribute form Inkscape
    # writes and the CSS form browsers honour (Chromium ignores the root's)
    texts = re.findall(r"<text\b[^>]*>", text)
    assert texts and all('xml:space="preserve"' in t for t in texts)
    assert all("white-space: pre" in t for t in texts)
    assert "<dc:date>" not in text                              # reproducible


def test_svg_is_deterministic_and_embeds_rasterized_artists(tmp_path):
    from matplotlib import pyplot as plt

    def cloud():
        fig, ax = plt.subplots(figsize=SIZE["single"])
        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(size=5000), rng.normal(size=5000), s=2,
                   alpha=0.2, rasterized=True)
        ax.plot([0, 1], [0, 1])
        return fig

    a = [p for p in save_figures({"a": cloud()}, tmp_path / "1") if p.suffix == ".svg"][0]
    b = [p for p in save_figures({"a": cloud()}, tmp_path / "2") if p.suffix == ".svg"][0]
    ta, tb = a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8")
    assert ta == tb                                             # byte-identical
    assert ta.count("<image") == 1                              # the cloud
    assert ta.count("<use") < 100                               # not 5000 markers
    assert a.stat().st_size < 1_500_000


def test_oversize_figure_is_written_but_warned(tmp_path):
    fig = _fig((9.0, 3.0))                            # wider than 7.5 in
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        paths = save_figures({"wide": fig}, tmp_path)
    assert all(p.exists() for p in paths)
    assert any("width" in str(x.message) for x in w)


def test_label_panels_adds_one_letter_per_axes():
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=SIZE["triple"])
    label_panels(axes)
    letters = [t.get_text() for ax in axes for t in ax.texts]
    assert letters == ["A", "B", "C"]
    assert all(t.get_fontsize() == 12 for ax in axes for t in ax.texts)
    plt.close(fig)


# ---- the manuscript trio: same frame, same axes, same fonts ------------------ #
def _trio(tmp_path):
    """Draw the three panels from synthetic data and return their SVG texts."""
    from cmsi.viz import results
    rng = np.random.default_rng(0)
    n = 3000
    disparity = rng.uniform(-45, 45, n)
    post = 1 / (1 + np.exp((np.abs(disparity) - 8) / 3))
    fused, seg = rng.normal(size=n), rng.normal(size=n) + 4 * np.sign(disparity)
    pred = seg + post * (fused - seg) * 0.86 + rng.normal(scale=0.6, size=n)
    w = post + rng.normal(scale=0.05, size=n)
    grid = np.linspace(-40, 40, 17)
    reg = {"slope": 0.86, "intercept": 0.0, "slope_ci95": [0.83, 0.88]}
    centres = np.linspace(-40, 40, 17)
    saved = {"bias_centres": centres, "bias_net": np.sin(centres / 8),
             "bias_centres_opt": centres, "bias_opt": np.sin(centres / 8) * 1.1}
    figs = {"A": results.panel_fusion_weight(disparity, w, post, grid, w_prop=w),
            "B": results.panel_position_regression(pred, seg, fused, post, reg),
            "C": results.panel_bias_vs_disparity(saved)}
    paths = save_figures(figs, tmp_path, formats=("svg", "png"))
    return {p.stem: p for p in paths if p.suffix == ".svg"}, figs


def test_manuscript_panels_share_frame_axes_and_fonts(tmp_path):
    svgs, figs = _trio(tmp_path)
    sizes, rects, fonts = set(), set(), set()
    for path in svgs.values():
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        sizes.add((root.get("width"), root.get("height")))
        axes_g = next(g for g in root.iter("{http://www.w3.org/2000/svg}g")
                      if g.get("id", "").startswith("axes_"))
        rects.add(axes_g.find(".//{http://www.w3.org/2000/svg}path").get("d"))
        fonts.add(tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))))
    assert sizes == {("180pt", "180pt")}                    # 2.5 in, exactly
    assert len(rects) == 1                                  # one axes rectangle
    assert fonts == {("8", "9")}                            # one font spec
    # and the raster twins are exactly 750 x 750 px, no tight cropping
    for stem in svgs:
        with Image.open(tmp_path / f"{stem}.png") as im:
            assert im.size == (750, 750)


def test_manuscript_panels_are_not_clipped(tmp_path):
    _, figs = _trio(tmp_path)
    for name, fig in figs.items():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        assert bb.x0 >= -0.01 and bb.y0 >= -0.01, name
        assert bb.x1 <= w + 0.01 and bb.y1 <= h + 0.01, name


def test_manuscript_row_matches_the_panels(tmp_path):
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    row = results.manuscript_row([lambda ax: ax.plot([0, 1]) for _ in range(3)])
    assert tuple(row.get_size_inches()) == (7.5, 2.5)
    axes = row.axes
    widths = {round(ax.get_position().width * 7.5, 4) for ax in axes}
    heights = {round(ax.get_position().height * 2.5, 4) for ax in axes}
    assert widths == {round(results.PANEL_RECT[2] * 2.5, 4)}     # same axes width
    assert heights == {round(results.PANEL_RECT[3] * 2.5, 4)}    # same axes height
    letters = [t.get_text() for ax in axes for t in ax.texts]
    assert letters == ["A", "B", "C"]
    plt.close(row)


def test_manuscript_grid_tiles_standard_panels(tmp_path):
    """A 2 x 2 grid is 5.0 x 5.0 in and every cell's axes is the standard
    rectangle -- the same one a 1 x 3 row and a standalone panel use."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    rng = np.random.default_rng(0)
    draw = [lambda ax, k=k: results.panel_error_histogram(rng.normal(size=2000), f"o{k}", ax=ax)
            for k in range(4)]
    grid = results.manuscript_grid(draw, ncols=2)
    assert tuple(grid.get_size_inches()) == (5.0, 5.0)
    (svg,) = [p for p in save_figures({"g": grid}, tmp_path, formats=("svg",))]
    text = svg.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert (root.get("width"), root.get("height")) == ("360pt", "360pt")
    rects = set()
    for g in root.iter("{http://www.w3.org/2000/svg}g"):
        if g.get("id", "").startswith("axes_"):
            d = g.find(".//{http://www.w3.org/2000/svg}path").get("d")
            xs = [float(v) for v in re.findall(r"[ML] ([\d.]+) ", d)]
            ys = [float(v) for v in re.findall(r"[ML] [\d.]+ ([\d.]+)", d)]
            # rectangle relative to its own 180-pt cell
            rects.add((round(min(xs) % 180, 2), round(min(ys) % 180, 2),
                       round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)))
    assert len(rects) == 1                                       # 4 cells, 1 rect
    (rect,) = rects
    left, bottom, width, height = results.PANEL_RECT
    assert rect[2] == round(width * 180, 2) and rect[3] == round(height * 180, 2)
    assert rect[0] == round(left * 180, 2)
    letters = [t.get_text() for ax in grid.axes for t in ax.texts]
    assert letters == ["A", "B", "C", "D"]                       # reading order
    assert tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))) == ("12", "8", "9")
    plt.close(grid)


def test_wide_cell_shares_margins_and_saves_into_its_folder(tmp_path):
    """A wide (3.75 x 2.5 in) cell has the same absolute margins as the square
    one, so its axes start at the same offset and have the same height; a
    'folder/name' key puts every format of the figure in that folder."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    m = results.PANEL_MARGIN
    sq, wd = results.panel_rect(results.CELL_SQUARE), results.panel_rect(results.CELL_WIDE)
    assert abs(sq[0] * 2.5 - m["left"]) < 1e-9                   # same left margin, in inches,
    assert abs(wd[0] * 3.75 - m["left"]) < 1e-9                  # for square and wide alike
    assert abs(sq[1] - wd[1]) < 1e-9 and abs(sq[3] - wd[3]) < 1e-9   # same bottom/height
    assert abs(wd[2] * 3.75 - (3.75 - m["left"] - m["right"])) < 1e-9

    sig = {"centres": np.linspace(0.05, 0.95, 10), "mixture": np.linspace(12, 4, 10),
           "net": np.linspace(11, 4.5, 10), "between": np.linspace(1, 0.2, 10),
           "fixed": np.full(10, 7.5)}
    draw = [lambda ax: results.panel_variance_hump(sig, "var_vis", ax=ax),
            lambda ax: results.panel_variance_hump(sig, "var_prop", legend=False, ax=ax)]
    row = results.manuscript_grid(draw, ncols=2, cell=results.CELL_WIDE)
    assert tuple(row.get_size_inches()) == (7.5, 2.5)           # same outer size as a 1 x 3
    paths = save_figures({"fig4_variance_hump/row_AB": row}, tmp_path, formats=("svg", "png"))
    assert all(p.parent == tmp_path / "fig4_variance_hump" for p in paths)
    text = (tmp_path / "fig4_variance_hump" / "row_AB.svg").read_text(encoding="utf-8")
    assert tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))) == ("12", "8", "9")
    assert "d\u00b2" in text and "deg\u00b2" in text          # glyph, not mathtext
    plt.close(row)


def test_sweep_manuscript_figure_is_a_full_over_two_wides(tmp_path):
    """A (CELL_FULL) over B + C (CELL_WIDE): 7.5 x 5.0 in, every axes on the
    standard margins and height, the colourbar inside A's cell, fonts
    {8, 9, 12}, and the file in its own folder under the manuscript dir."""
    from matplotlib import pyplot as plt

    from cmsi.viz import manuscript, prior_sweep
    priors = [0.2, 0.5, 0.8]
    rows = [{"p_common": p, "midpoint_net": 12 * p + 0.5, "midpoint_opt": 12 * p,
             "midpoint_net_sem": 0.2, "posreg_slope_vis": 0.8 + 0.1 * p,
             "posreg_ci_vis": [0.75 + 0.1 * p, 0.85 + 0.1 * p], "n_seeds": 3}
            for p in priors]
    c = np.linspace(-30, 30, 13)
    curves = {}
    for p in priors:
        w = 1 / (1 + np.exp((np.abs(c) - 12 * p) / 3))
        curves.update({f"p{p:.2f}_centres_net": c, f"p{p:.2f}_w_net": w,
                       f"p{p:.2f}_se_net": np.full_like(c, 0.02),
                       f"p{p:.2f}_centres_opt": c, f"p{p:.2f}_w_opt": w * 1.05})
    fig = prior_sweep.manuscript_figure(rows, curves)
    assert tuple(fig.get_size_inches()) == (7.5, 5.0)
    W, H = fig.get_size_inches()
    m = manuscript.PANEL_MARGIN
    axA, axB, axC, cax = fig.axes
    for ax, cell_x0, cell_y0 in ((axA, 0, 2.5), (axB, 0, 0), (axC, 3.75, 0)):
        pos = ax.get_position()
        assert abs(pos.x0 * W - (cell_x0 + m["left"])) < 1e-6
        assert abs(pos.y0 * H - (cell_y0 + m["bottom"])) < 1e-6
        assert abs(pos.height * H - (2.5 - m["bottom"] - m["top"])) < 1e-6
    assert abs(axB.get_position().width - axC.get_position().width) < 1e-9
    assert cax.get_position().x1 * W < 7.5 - m["right"] + 1e-6      # colourbar inside A's cell
    letters = [t.get_text() for ax in (axA, axB, axC) for t in ax.texts]
    assert letters == ["A", "B", "C"]
    paths = save_figures({f"{prior_sweep.FOLDER}/prior_sweep_ABC": fig}, tmp_path, formats=("svg",))
    text = paths[0].read_text(encoding="utf-8")
    assert paths[0].parent.name == prior_sweep.FOLDER
    assert tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))) == ("12", "8", "9")
    assert "$" not in text.replace("$", "")  # no stray mathtext markup survives into the file
    plt.close(fig)


def test_manuscript_dir_protects_the_flagship_figures():
    from cmsi.utils.paths import RESULTS
    from cmsi.viz.manuscript import manuscript_dir
    assert manuscript_dir() == RESULTS / "manuscript"
    assert manuscript_dir("flagship") == RESULTS / "manuscript"
    assert manuscript_dir("pcommon07") == RESULTS / "manuscript_pcommon07"
