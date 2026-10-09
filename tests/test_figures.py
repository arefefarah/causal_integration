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
from pathlib import Path

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
    w = post + rng.normal(scale=0.05, size=n)
    flags = (np.abs(disparity) > 6).astype(int)        # ratio where the gap is wide
    grid = np.linspace(-40, 40, 17)
    reg = {"slope": 0.98, "intercept": 0.01, "slope_ci95": [0.97, 0.99]}
    centres = np.linspace(-40, 40, 17)
    saved = {"bias_centres": centres, "bias_net": np.sin(centres / 8),
             "bias_centres_opt": centres, "bias_opt": np.sin(centres / 8) * 1.1}
    figs = {"A": results.panel_fusion_weight(disparity, w, post, grid, w_prop=w),
            "B": results.panel_weight_regression(w, post, reg, flags=flags),
            "C": results.panel_bias_vs_disparity(saved)}
    paths = save_figures(figs, tmp_path, formats=("svg", "png"))
    return {p.stem: p for p in paths if p.suffix == ".svg"}, figs


def test_weight_regression_panel_has_both_legends_and_the_shares(tmp_path):
    """Figure 4B: the hybrid read against the posterior, sources in one legend
    (lower right, with their shares of all trials), the two lines in the
    other (upper left), on the standard panel with panel A's y range."""
    from cmsi.viz import results
    rng = np.random.default_rng(1)
    n = 2000
    post = rng.uniform(0, 1, n)
    w = post + rng.normal(scale=0.1, size=n)
    flags = np.where(post > 0.6, 0, 1)
    reg = {"slope": 0.97, "intercept": 0.0, "slope_ci95": [0.96, 0.98]}
    fig = results.panel_weight_regression(w, post, reg, flags=flags, n_show=500)
    ax = fig.axes[0]
    legends = [art for art in ax.get_children() if art.__class__.__name__ == "Legend"]
    assert len(legends) == 2
    texts = [t.get_text() for leg in legends for t in leg.get_texts()]
    assert f"var_vis root: {100 * np.mean(flags == 0):.0f}%" in texts
    assert f"mu_vis ratio: {100 * np.mean(flags == 1):.0f}%" in texts
    assert any(t.startswith("fit: slope 0.97") for t in texts)
    assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (-0.25, 1.4)
    assert tuple(fig.get_size_inches()) == tuple(results.CELL_SQUARE)
    # the scatter is subsampled for drawing; the shares are not
    drawn = sum(len(c.get_offsets()) for c in ax.collections)
    assert drawn == 500


def test_manuscript_panels_figure_4_is_weight_regression(tmp_path):
    """The manuscript's figure 4 is A fusion weight, B the hybrid read on the
    posterior (08v panel B), C bias vs disparity; the position regression is
    no longer a manuscript panel."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    rng = np.random.default_rng(2)
    n = 600
    names = ["mu_vis", "var_vis", "mu_prop", "var_prop"]
    post = rng.uniform(0, 1, n)
    d = {"disparity": rng.uniform(-30, 30, n), "post_c1": post,
         "mu_vis": rng.normal(size=n), "var_vis": rng.uniform(2, 6, n),
         "mu_prop": rng.normal(size=n), "var_prop": rng.uniform(2, 6, n)}
    pred = np.stack([d[k] for k in names], axis=1) + rng.normal(scale=0.1, size=(n, 4))
    centres = np.linspace(-30, 30, 7)
    saved = {"bias_centres": centres, "bias_net": np.sin(centres / 8),
             "bias_centres_opt": centres, "bias_opt": np.sin(centres / 8),
             "hybrid_flags_vis": np.where(post > 0.5, 0, 1)}
    metrics = {"weight_regression_vis": {"slope": 0.97, "intercept": 0.0,
                                         "slope_ci95": [0.96, 0.98]}}
    w = post + rng.normal(scale=0.1, size=n)
    figs = results.manuscript_panels(pred, d, names, {"disparity_grid": centres.tolist()},
                                     w, w, saved, metrics)
    F = results.FIG_WEIGHT
    keys = {k for k in figs if k.startswith(F)}
    assert keys == {f"{F}/A_fusion_weight", f"{F}/B_weight_regression",
                    f"{F}/C_bias_vs_disparity", f"{F}/{F}"}
    row = figs[f"{results.FIG_WEIGHT}/{results.FIG_WEIGHT}"]
    assert [ax.get_title() for ax in row.axes] == [
        "fusion-segregation transition", "implied weight vs posterior", "bias vs disparity"]
    for fig in figs.values():
        plt.close(fig)


def test_decoding_panel_is_thin_labelled_bars_on_the_wide_cell():
    """Figure 9A: held-out R² per hidden layer, the causal network beside its
    twin, narrow bars carrying their values, the layers named as the paper
    names them, and headroom above the tallest bar for the legend."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    causal = {"layer0": 0.888, "layer1": 0.973}
    twin = {"layer0": 0.365, "layer1": 0.288}
    fig = results.panel_decoding(causal, twin)
    ax = fig.axes[0]
    assert tuple(fig.get_size_inches()) == tuple(results.CELL_WIDE)
    bars = [p for p in ax.patches if p.get_width() > 0]
    assert len(bars) == 4 and {round(b.get_width(), 3) for b in bars} == {0.2}
    assert [t.get_text() for t in ax.get_xticklabels()] == ["SIL", "MSL"]
    values = {t.get_text() for t in ax.texts}
    assert {"0.89", "0.97", "0.36", "0.29"} <= values           # every bar labelled
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["causal network", "always-fuse twin"]
    assert ax.get_ylim() == (0, 1.45) and max(ax.get_yticks()) == 1.0
    # the twin's bars are grey, the network's the network colour
    colours = {b.get_facecolor() for b in bars}
    assert len(colours) == 2
    plt.close(fig)
    # without a twin: two bars, no legend entry for it
    fig = results.panel_decoding(causal)
    assert len([p for p in fig.axes[0].patches if p.get_width() > 0]) == 2
    plt.close(fig)


def test_model_decoding_figures_are_drawn_like_the_manuscript_panel():
    """06_decoding / 07_emergent_vs_imposed share figure 9A's drawing: narrow
    bars carrying their values, SIL/MSL names, the twin in grey, headroom
    for the legend; on the single-panel figure size."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    from cmsi.viz.style import COLORS
    causal = {"layer0": 0.888, "layer1": 0.973}
    twin = {"layer0": 0.365, "layer1": 0.288}
    fig = results.decoding_comparison({"causal network": causal, "always-fuse twin": twin})
    ax = fig.axes[0]
    assert tuple(fig.get_size_inches()) == tuple(SIZE["single"])
    bars = [p for p in ax.patches if p.get_width() > 0]
    assert len(bars) == 4 and {round(b.get_width(), 3) for b in bars} == {0.2}
    assert [t.get_text() for t in ax.get_xticklabels()] == ["SIL", "MSL"]
    assert {"0.89", "0.97", "0.36", "0.29"} <= {t.get_text() for t in ax.texts}
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "causal network", "always-fuse twin"]
    from matplotlib.colors import to_hex
    assert {to_hex(b.get_facecolor()) for b in bars} == {COLORS["network"], COLORS["twin"]}
    assert ax.get_ylim() == (0, 1.3) and max(ax.get_yticks()) == 1.0
    plt.close(fig)
    fig = results.decoding_bars(causal)
    ax = fig.axes[0]
    assert len([p for p in ax.patches if p.get_width() > 0]) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["SIL", "MSL"]
    assert {"0.89", "0.97"} <= {t.get_text() for t in ax.texts}
    assert ax.get_legend() is None
    plt.close(fig)


def test_rf_shift_panel_marks_both_reference_frames():
    """Figure 9B: the gains in 0.1-wide bins aligned to zero, the x range
    always holding both the spatial (0) and the retinal (+1) marks, the
    median in the title."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    rng = np.random.default_rng(0)
    g = np.clip(rng.normal(-0.1, 0.3, 64), -1.2, 0.4)
    fig = results.panel_rf_shift({"rf_shift_gain": g})
    ax = fig.axes[0]
    assert tuple(fig.get_size_inches()) == tuple(results.CELL_WIDE)
    lo, hi = ax.get_xlim()
    assert lo <= min(-0.5, g.min()) and hi >= 1.3                 # both marks inside
    widths = {round(p.get_width(), 6) for p in ax.patches}
    assert widths == {0.1}
    assert all(round(p.get_x() / 0.1, 6) == round(p.get_x() / 0.1) for p in ax.patches)
    marks = sorted(ln.get_xdata()[0] for ln in ax.lines)
    assert marks == [0.0, 1.0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["spatial code (0)", "retinal code (+1)"]
    assert f"median {np.median(g):.2f}" in ax.get_title()
    counts = sum(p.get_height() for p in ax.patches)
    assert counts == 64 and ax.get_ylim()[1] > max(p.get_height() for p in ax.patches)
    plt.close(fig)


def test_manuscript_panels_figure_9_is_decoding_and_rf_shift(tmp_path):
    """The manuscript's figure 9 is A the posterior decoded by layer, B the RF
    shift gains, two wide cells in one 7.5 x 2.5 in row; it is skipped
    cleanly when the run has no decoding or no RF sweep."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    rng = np.random.default_rng(3)
    n = 400
    names = ["mu_vis", "var_vis", "mu_prop", "var_prop"]
    post = rng.uniform(0, 1, n)
    d = {"disparity": rng.uniform(-30, 30, n), "post_c1": post,
         "mu_vis": rng.normal(size=n), "var_vis": rng.uniform(2, 6, n),
         "mu_prop": rng.normal(size=n), "var_prop": rng.uniform(2, 6, n)}
    pred = np.stack([d[k] for k in names], axis=1) + rng.normal(scale=0.1, size=(n, 4))
    centres = np.linspace(-30, 30, 7)
    saved = {"bias_centres": centres, "bias_net": np.sin(centres / 8),
             "bias_centres_opt": centres, "bias_opt": np.sin(centres / 8),
             "hybrid_flags_vis": np.where(post > 0.5, 0, 1),
             "rf_shift_gain": rng.normal(0, 0.3, 64)}
    metrics = {"weight_regression_vis": {"slope": 0.97, "intercept": 0.0,
                                         "slope_ci95": [0.96, 0.98]},
               "post_c1_decoding_r2": {"layer0": 0.9, "layer1": 0.97},
               "twin_post_c1_decoding_r2": {"layer0": 0.3, "layer1": 0.3}}
    w = post + rng.normal(scale=0.1, size=n)
    figs = results.manuscript_panels(pred, d, names, {"disparity_grid": centres.tolist()},
                                     w, w, saved, metrics)
    keys = {k for k in figs if k.startswith(results.FIG_DECODING)}
    assert keys == {f"{results.FIG_DECODING}/A_decoding", f"{results.FIG_DECODING}/B_rf_shift",
                    f"{results.FIG_DECODING}/{results.FIG_DECODING}"}
    row = figs[f"{results.FIG_DECODING}/{results.FIG_DECODING}"]
    assert tuple(row.get_size_inches()) == (7.5, 2.5)
    assert [ax.get_title()[:22] for ax in row.axes] == [
        "where the causal poste", "reference frame of MSL"]
    assert [t.get_text() for ax in row.axes for t in ax.texts if t.get_text() in "AB"] == ["A", "B"]
    F = results.FIG_DECODING
    paths = save_figures({f"{F}/{F}": row}, tmp_path, formats=("svg", "png"))
    assert all(p.parent == tmp_path / F for p in paths)
    # the composed figure, named as its folder, is also copied flat beside it
    assert (tmp_path / f"{F}.png").exists()
    text = (tmp_path / F / f"{F}.svg").read_text(encoding="utf-8")
    assert tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))) == ("12", "8", "9")
    for fig in figs.values():
        plt.close(fig)
    # no decoding in metrics -> no figure 9, nothing else lost
    metrics.pop("post_c1_decoding_r2")
    figs = results.manuscript_panels(pred, d, names, {"disparity_grid": centres.tolist()},
                                     w, w, saved, metrics)
    assert not any(k.startswith(results.FIG_DECODING) for k in figs)
    assert f"{results.FIG_WEIGHT}/{results.FIG_WEIGHT}" in figs
    for fig in figs.values():
        plt.close(fig)


def test_model_comparison_panels_are_wide_cells_with_the_house_style(tmp_path):
    """Figure 8: per-decile implied weight (network + five strategies) and the
    per-decile RMSE of each strategy on a log axis, two WIDE cells, 7.5 x 2.5
    in, panel letters A B, fonts 8/9/12 and a flat copy of the composed png."""
    from matplotlib import pyplot as plt

    from cmsi.viz import results
    c = np.linspace(0.05, 0.95, 10)
    strategies = ("averaging", "integration", "segregation", "selection", "fixed")
    mc = {"bin_centres": c.tolist(), "bin_weight_net": (0.95 * c).tolist(),
          "bin_weight": {k: (c if k == "averaging" else np.where(c > 0.5, 1.0, 0.0)
                             if k == "selection" else np.full(10, {"integration": 1.0,
                                                                   "segregation": 0.0,
                                                                   "fixed": 0.05}[k])).tolist()
                         for k in strategies},
          "bin_rmse": {k: np.linspace(0.5 if k == "averaging" else 1.0,
                                      8.0 if k == "integration" else 1.2, 10).tolist()
                       for k in strategies},
          "best": "averaging"}
    draw = [lambda ax: results.panel_model_weight(mc, ax=ax),
            lambda ax: results.panel_model_rmse(mc, ax=ax)]
    row = results.manuscript_grid(draw, ncols=2, cell=results.CELL_WIDE)
    assert tuple(row.get_size_inches()) == (7.5, 2.5)
    assert [ax.get_title() for ax in row.axes] == ["weight by posterior decile",
                                                   "which strategy explains the network"]
    assert row.axes[1].get_yscale() == "log"
    labels = [t.get_text() for t in row.axes[0].get_legend().get_texts()]
    assert labels == ["model averaging", "model selection", "full integration",
                      "full segregation", "best fixed weight", "network"]
    assert [t.get_text() for ax in row.axes for t in ax.texts if t.get_text() in "AB"] == ["A", "B"]
    F = results.FIG_MODEL
    paths = save_figures({f"{F}/{F}": row}, tmp_path, formats=("svg", "png"))
    assert all(p.parent == tmp_path / F for p in paths)
    assert (tmp_path / f"{F}.png").exists()
    text = (tmp_path / F / f"{F}.svg").read_text(encoding="utf-8")
    assert tuple(sorted(set(re.findall(r"font-size: ?([\d.]+)px", text)))) == ("12", "8", "9")
    plt.close(row)


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
    F = results.FIG_HUMP
    paths = save_figures({f"{F}/{F}": row}, tmp_path, formats=("svg", "png"))
    assert all(p.parent == tmp_path / F for p in paths)
    assert (tmp_path / f"{F}.png").exists()                 # flat copy of the composed figure
    text = (tmp_path / F / f"{F}.svg").read_text(encoding="utf-8")
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
    F = prior_sweep.FOLDER
    paths = save_figures({f"{F}/{F}": fig}, tmp_path, formats=("svg",))
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


def test_figure_registry_numbers_the_paper_once():
    """viz/manuscript.py is the one place a figure is numbered: the names run
    fig1..fig11 without a gap, in the paper's order, and every writer (the
    per-run panels, the sweep, the neuronal scripts) takes its name from
    there. Figure 1 is the drawn schematic, which the pipeline never renders."""
    import re

    from cmsi.viz import manuscript, prior_sweep, results
    names = manuscript.FIGURES
    assert [manuscript.figure_number(n) for n in names] == list(range(1, len(names) + 1))
    assert all(re.fullmatch(r"fig\d+_[a-z_]+", n) for n in names)
    assert names[0] == manuscript.FIG_TASK
    assert prior_sweep.FOLDER == manuscript.FIG_SWEEP
    rendered = {results.FIG_SCATTER, results.FIG_ERRORS, results.FIG_WEIGHT, results.FIG_HUMP,
                results.FIG_MODEL, results.FIG_DECODING}
    neuronal = {manuscript.FIG_BIAS, manuscript.FIG_UNITS, manuscript.FIG_LESION}
    assert rendered | neuronal | {manuscript.FIG_SWEEP, manuscript.FIG_TASK} == set(names)
    assert manuscript.FIG_TASK not in rendered | neuronal
    # the figure name does not leak into the code except through the registry
    src = (Path(results.__file__).read_text(encoding="utf-8")
           + Path(prior_sweep.__file__).read_text(encoding="utf-8"))
    assert not re.search(r'"fig\d+_', src)


def test_weight_figures_draw_the_hybrid_read():
    """04, 08v, 15 and 16: the pipeline's weight figures, all from the hybrid
    read of a channel -- the curve through zero disparity, the variance-domain
    headline beside the weight on the posterior (points coloured by source),
    the weight in posterior bins, and its distribution against the
    posterior's with the overflow piled into the edge bins."""
    from cmsi.analysis.causal import (
        hybrid_weight,
        mixture_variance,
        variance_regression,
        weight_by_posterior,
        weight_regression,
    )
    from cmsi.viz import results
    from cmsi.viz.style import COLORS

    rng = np.random.default_rng(1)
    n = 3000
    disp = rng.normal(0, 12, n)
    post = 1 / (1 + np.exp(0.4 * (np.abs(disp) - 8)))
    fused_var, seg_var = rng.uniform(2, 3, n), rng.uniform(5, 9, n)
    delta = 0.6 * disp
    var_out = mixture_variance(post, fused_var, seg_var, delta) + rng.normal(0, 0.05, n)
    mu_out = post * delta + rng.normal(0, 0.3, n)
    w, _, flags = hybrid_weight(var_out, fused_var, seg_var, delta, mu_out, np.zeros(n))
    grid = [-40, -30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30, 40]

    fig = results.fusion_weight_curve(disp, w, post, grid, w_prop=w)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert sum("implied" in lab for lab in labels) == 2
    assert any("analytical" in lab for lab in labels)
    # the curve carries through zero disparity: the network's line has a
    # point at every grid centre, the origin included
    net = [line for line in ax.get_lines() if "vis" in line.get_label()][0]
    assert 0.0 in np.round(net.get_xdata(), 6)
    assert ax.get_ylim() == (-0.1, 1.1)

    reg_v = variance_regression(var_out, fused_var, seg_var, delta, post)
    fig2 = results.variance_regression_figure(var_out, fused_var, seg_var, delta, post, reg_v,
                                              w, weight_regression(w, post), "var_vis",
                                              flags=flags)
    assert len(fig2.axes) == 2
    assert "variance-domain regression" in fig2.axes[0].get_title()
    assert "implied weight on the posterior" in fig2.axes[1].get_title()
    labels = [t.get_text() for t in fig2.axes[1].get_legend().get_texts()]
    assert any("variance root" in lab for lab in labels)
    assert any("position ratio" in lab for lab in labels)
    fit = [line for line in fig2.axes[1].get_lines() if line.get_label().startswith("fit")][0]
    assert fit.get_color() == COLORS["hybrid"]

    fig3 = results.weight_vs_posterior(weight_by_posterior(w, post), "vis")
    ax = fig3.axes[0]
    assert ax.get_xlim() == (0, 1)
    assert any("all 3000 trials" in t.get_text() for t in ax.texts)

    w_wide = w.copy()
    w_wide[:30] = 5.0                                   # thirty trials beyond the axis
    fig4 = results.hybrid_weight_distribution(w_wide, w, post, flags_vis=flags,
                                              flags_prop=flags)
    assert len(fig4.axes) == 2
    assert fig4.axes[0].get_xlim() == (-1.0, 2.0)
    texts = " ".join(t.get_text() for t in fig4.axes[0].texts)
    assert f"{100 * 30 / n:.1f}% of trials outside" in texts
    assert "read from the mu_vis ratio" in texts
    assert "vis: median" in fig4.axes[0].get_title()
    assert "prop: median" in fig4.axes[1].get_title()
    labels = [t.get_text() for t in fig4.axes[0].get_legend().get_texts()]
    assert any(lab.startswith("network, all") for lab in labels)
    assert any("analytical posterior" in lab for lab in labels)
    from matplotlib import pyplot as plt
    plt.close("all")
