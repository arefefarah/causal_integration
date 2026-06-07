"""``visualize`` CLI: render diagnostic figures into ``results/``.

Produces four families of figures so you can sanity-check the pipeline end to end:

1. ENCODING -- tuning curves, single-trial population vectors, code heatmaps,
   reliability-gain and Poisson-noise checks (verify the sensory encoding).
2. MODEL QUALITY -- training/validation loss curves, per-output R^2 bars,
   common-cause decision accuracy.
3. OUTPUT ASSESSMENT -- decoded-vs-analytical scatter for all five outputs (with
   R^2), per-output error histograms, common-cause calibration.
4. ANALYSIS -- proportion-common-cause vs disparity (network vs analytical) and
   the implicit-integration comparison (reconstructed/decoded vs analytical).

By default a small model is trained quickly for the demo; pass ``--checkpoint`` to
visualise a pre-trained model instead (skips training and the loss-curve figure).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import typer  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from causal_msi import viz  # noqa: E402
from causal_msi.analysis import integration as integ  # noqa: E402
from causal_msi.analysis.decoding import extract_activations  # noqa: E402
from causal_msi.analysis.performance import (  # noqa: E402
    OUTPUT_NAMES,
    error_distributions,
    pc_calibration,
    per_output_regression,
)
from causal_msi.config import Config, load_config  # noqa: E402
from causal_msi.encoding import (  # noqa: E402
    Encoders,
    gaussian_rf_code,
    poisson_noise,
    push_pull_code,
    reliability_gain,
)
from causal_msi.generative import Dataset, build_dataset, transform_visual_to_body  # noqa: E402
from causal_msi.io import load_checkpoint  # noqa: E402
from causal_msi.models import FeedforwardMSI, build_model  # noqa: E402
from causal_msi.training import TrainResult, train_one  # noqa: E402
from causal_msi.utils import seed_everything  # noqa: E402

app = typer.Typer(add_completion=False, help="Render diagnostic figures into results/.")

FloatArray = np.ndarray


def _save(fig: Figure, results_dir: Path, name: str) -> None:
    """Save and close a figure under ``results_dir``."""
    path = results_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# 1. Encoding figures
# --------------------------------------------------------------------------- #
def _encoding_figures(results_dir: Path, cfg: Config) -> None:
    """Render encoding-verification figures from a freshly seeded encoder set."""
    rng = seed_everything(cfg.seed)
    enc = cfg.encoding
    encoders = Encoders.build(rng, enc)
    field = np.linspace(enc.visual_field[0], enc.visual_field[1], 200)
    unit_gain = np.ones_like(field)

    vis = gaussian_rf_code(field, encoders.rf_centers, enc.rf_width, unit_gain)
    ph = push_pull_code(field, encoders.prop_hand, unit_gain)
    pe = push_pull_code(field, encoders.prop_eye, unit_gain)

    # (a) Tuning curves per group.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.tuning_curves(field, vis, axes[0], "visual_hand (Gaussian RF)", max_units=25)
    viz.tuning_curves(field, ph, axes[1], "prop_hand (push-pull)", max_units=25)
    viz.tuning_curves(field, pe, axes[2], "prop_eye (push-pull)", max_units=25)
    fig.suptitle("Encoding tuning curves (gain=1, no Poisson)")
    _save(fig, results_dir, "01_encoding_tuning_curves.png")

    # (b) Single-trial population vectors at three stimulus values.
    picks = [enc.visual_field[0] / 2, 0.0, enc.visual_field[1] / 2]
    pick_arr = np.array(picks)
    g1 = np.ones_like(pick_arr)
    vis_p = gaussian_rf_code(pick_arr, encoders.rf_centers, enc.rf_width, g1)
    ph_p = push_pull_code(pick_arr, encoders.prop_hand, g1)
    pe_p = push_pull_code(pick_arr, encoders.prop_eye, g1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.population_vectors(picks, [vis_p[i] for i in range(3)], axes[0], "visual_hand vector")
    viz.population_vectors(picks, [ph_p[i] for i in range(3)], axes[1], "prop_hand vector")
    viz.population_vectors(picks, [pe_p[i] for i in range(3)], axes[2], "prop_eye vector")
    fig.suptitle("Single-trial population vectors (bump moves; ramps shift)")
    _save(fig, results_dir, "02_encoding_population_vectors.png")

    # (c) Code heatmaps: stimulus on rows (sorted), units on columns.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.input_heatmap(vis, axes[0], "visual_hand", ylabel="stimulus (low->high)")
    viz.input_heatmap(ph, axes[1], "prop_hand", ylabel="stimulus (low->high)")
    viz.input_heatmap(pe, axes[2], "prop_eye", ylabel="stimulus (low->high)")
    fig.suptitle("Population code vs stimulus (mean rate)")
    _save(fig, results_dir, "03_encoding_heatmaps.png")

    # (d) Reliability gain + Poisson-mean checks.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vis_lo, vis_hi = cfg.generative.sigma2_vis_range
    variances = np.linspace(vis_lo, vis_hi, 60)
    base = gaussian_rf_code(
        np.zeros_like(variances), encoders.rf_centers, enc.rf_width, np.ones_like(variances)
    ).sum(axis=1)
    total = base * reliability_gain(variances, enc.gain_K)
    axes[0].plot(1.0 / variances, total, "o-", ms=3)
    axes[0].set_xlabel("1 / variance  (reliability, 1/deg^2)")
    axes[0].set_ylabel("total visual_hand activity")
    axes[0].set_title("Reliability gain ~ 1/variance")

    rng2 = np.random.default_rng(0)
    lambdas = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    samples = poisson_noise(rng2, np.tile(lambdas, (20000, 1)))
    axes[1].plot(lambdas, samples.mean(axis=0), "o", ms=6, label="sample mean")
    axes[1].plot(lambdas, lambdas, "k--", lw=1, label="identity (mean=lambda)")
    axes[1].set_xlabel("rate lambda")
    axes[1].set_ylabel("Poisson sample mean")
    axes[1].set_title("Poisson noise: mean = lambda")
    axes[1].legend(fontsize=8)
    _save(fig, results_dir, "04_encoding_gain_poisson.png")


# --------------------------------------------------------------------------- #
# Train or load
# --------------------------------------------------------------------------- #
def _get_model(cfg: Config, checkpoint: Path | None) -> tuple[FeedforwardMSI, TrainResult | None]:
    """Train a quick model or load a checkpoint; return model (on CPU) + result."""
    if checkpoint is not None:
        payload = load_checkpoint(checkpoint)
        model = build_model(int(payload["input_dim"]), cfg.model)
        model.load_state_dict(payload["state_dict"])
        return model.to("cpu").eval(), None
    rng = seed_everything(cfg.seed)
    dataset = build_dataset(rng, cfg)
    result = train_one(dataset, cfg, seed=cfg.seed, progress=False)
    return result.model.to("cpu").eval(), result


# --------------------------------------------------------------------------- #
# 2. Model-quality figures
# --------------------------------------------------------------------------- #
def _quality_figures(
    results_dir: Path, result: TrainResult | None, pred: FloatArray, target: FloatArray
) -> None:
    """Loss curves, per-output R^2 bars, and common-cause accuracy."""
    if result is not None and result.history:
        train_total = np.array([e.train_total for e in result.history])
        val_total = np.array([e.val_total for e in result.history])
        comps = {
            k: np.array([e.components.get(k, np.nan) for e in result.history])
            for k in ("est", "var", "pc")
        }
        fig, ax = plt.subplots(figsize=(7, 4.5))
        viz.loss_curves(train_total, val_total, ax, comps)
        ax.set_title(f"Training (best val={result.best_val:.3f} @ epoch {result.best_epoch})")
        _save(fig, results_dir, "05_training_loss.png")

    regs = per_output_regression(pred, target)
    acc = float(np.mean((pred[:, 4] > 0.5) == (target[:, 4] > 0.5)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    viz.bar_metrics(
        [r.name for r in regs],
        [r.r2 for r in regs],
        ax,
        ylabel="held-out R^2",
        title=f"Per-output R^2  |  common-cause accuracy={acc:.2f}",
        ylim=(0.0, 1.05),
    )
    _save(fig, results_dir, "06_output_r2_bars.png")


# --------------------------------------------------------------------------- #
# 3. Output-assessment figures
# --------------------------------------------------------------------------- #
def _output_figures(results_dir: Path, pred: FloatArray, target: FloatArray) -> None:
    """Decoded-vs-analytical scatter (5 outputs + calibration) and error histograms."""
    regs = per_output_regression(pred, target)
    sub = np.random.default_rng(0).choice(
        pred.shape[0], size=min(2000, pred.shape[0]), replace=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat = axes.ravel()
    for i, (name, reg) in enumerate(zip(OUTPUT_NAMES, regs, strict=True)):
        viz.regression_scatter(target[sub, i], pred[sub, i], flat[i], label=name)
        flat[i].set_title(f"{name}: R^2={reg.r2:.3f}, slope={reg.slope:.2f}")
    xs, ys = pc_calibration(pred[:, 4], target[:, 4])
    flat[5].plot([0, 1], [0, 1], "k--", lw=1)
    flat[5].plot(xs, ys, "o-")
    flat[5].set_title("p(C=1) calibration")
    flat[5].set_xlabel("predicted p(C=1)")
    flat[5].set_ylabel("analytical p(C=1)")
    fig.suptitle("Decoded vs analytical (all outputs)")
    _save(fig, results_dir, "07_output_decoding_scatter.png")

    errs = error_distributions(pred, target)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, name in enumerate(OUTPUT_NAMES):
        viz.error_histogram(errs[name], axes.ravel()[i])
        axes.ravel()[i].set_title(f"{name} error  (mean={errs[name].mean():.2f})")
    axes.ravel()[5].axis("off")
    fig.suptitle("Per-output error distributions (decoded - analytical)")
    _save(fig, results_dir, "08_output_error_histograms.png")


# --------------------------------------------------------------------------- #
# 4. Analysis figures
# --------------------------------------------------------------------------- #
def _bin_means(x: FloatArray, y: FloatArray, edges: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean of ``y`` within bins of ``x`` defined by ``edges`` -> (centres, means)."""
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    centres, means = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(float(y[m].mean()))
    return np.asarray(centres), np.asarray(means)


def _analysis_figures(
    results_dir: Path, cfg: Config, model: FeedforwardMSI, dataset: Dataset, pred: FloatArray
) -> None:
    """Proportion-common vs disparity and the implicit-integration comparison."""
    t = dataset.targets
    _, var_vis_body = transform_visual_to_body(
        dataset.measurements.x_vis,
        dataset.measurements.x_eye,
        dataset.latents.sigma2_vis,
        dataset.latents.sigma2_eye,
    )
    _ = var_vis_body
    x_vis_body = dataset.measurements.x_vis + dataset.measurements.x_eye
    disparity = x_vis_body - dataset.measurements.x_prop
    edges = np.linspace(-30, 30, 21)
    c_an, m_an = _bin_means(disparity, t.p_common, edges)
    c_net, m_net = _bin_means(disparity, pred[:, 4], edges)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    viz.binned_curve(
        c_an,
        {"analytical": m_an},
        ax,
        xlabel="body-frame disparity (deg)",
        ylabel="p(C=1)",
        title="Proportion common-cause vs disparity",
    )
    ax.plot(c_net, m_net, marker="s", ms=4, lw=1.3, label="network")
    ax.legend(fontsize=8)
    _save(fig, results_dir, "09_proportion_common_vs_disparity.png")

    analytical = integ.analytical_model_averaged_estimate(t)
    reconstructed = integ.network_reconstructed_estimate(
        pred, cfg.generative.mu0, cfg.generative.sigma0_sq
    )
    msl = extract_activations(model, dataset.X)["msl"]
    decoded = integ.population_decoded_estimate(msl, analytical, cfg.analysis.decoder)
    cmp_recon = integ.compare_estimates(analytical, reconstructed)

    sub = np.random.default_rng(0).choice(
        len(analytical), size=min(2000, len(analytical)), replace=False
    )
    w_recon = integ.fusion_weight_curve(t.mu_vis_body, t.fused_mu, reconstructed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    viz.regression_scatter(analytical[sub], reconstructed[sub], axes[0], label="reconstructed")
    axes[0].set_title(
        f"Reconstructed vs analytical model-avg\nR^2={cmp_recon.r2:.3f}  "
        f"(population-decoded R^2={decoded.r2:.3f})"
    )
    axes[0].set_xlabel("analytical estimate (deg)")
    axes[0].set_ylabel("reconstructed estimate (deg)")
    finite = np.isfinite(w_recon)
    fsub = sub[finite[sub]]
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="optimal")
    axes[1].plot(t.p_common[fsub], np.clip(w_recon[fsub], -0.5, 1.5), ".", ms=3, alpha=0.4)
    axes[1].set_xlabel("analytical p(C=1)")
    axes[1].set_ylabel("empirical fusion weight")
    axes[1].set_title("Fusion weight vs p(C=1)")
    axes[1].legend(fontsize=8)
    _save(fig, results_dir, "10_integration_estimates.png")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config.")] = Path("configs/default.yaml"),
    results_dir: Annotated[Path, typer.Option(help="Output directory for figures.")] = Path(
        "results"
    ),
    checkpoint: Annotated[
        Path | None, typer.Option(help="Visualise this checkpoint instead of quick-training.")
    ] = None,
    n_trials: Annotated[int, typer.Option(help="Trials for the demo dataset.")] = 8000,
    epochs: Annotated[int, typer.Option(help="Epochs for the quick demo train.")] = 100,
) -> None:
    """Render all diagnostic figures into ``results_dir``."""
    cfg = load_config(config)
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={"n_trials": n_trials, "epochs": epochs, "n_seeds": 1}
            )
        }
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("[1/4] Encoding figures")
    _encoding_figures(results_dir, cfg)

    typer.echo("[2/4] Training / loading model")
    model, result = _get_model(cfg, checkpoint)

    # Held-out evaluation dataset (distinct seed) for honest output assessment.
    eval_rng = seed_everything(cfg.seed + 999)
    eval_ds = build_dataset(eval_rng, cfg, n_trials=min(n_trials, 6000))
    with torch.no_grad():
        pred = model(torch.as_tensor(eval_ds.X, dtype=torch.float32)).cpu().numpy()
    target = eval_ds.Y

    typer.echo("[3/4] Model-quality + output figures")
    _quality_figures(results_dir, result, pred, target)
    _output_figures(results_dir, pred, target)

    typer.echo("[4/4] Analysis figures")
    _analysis_figures(results_dir, cfg, model, eval_ds, pred)

    typer.echo(f"Done. Figures saved under {results_dir}/")


if __name__ == "__main__":
    app()
