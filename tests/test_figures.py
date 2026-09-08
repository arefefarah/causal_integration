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
