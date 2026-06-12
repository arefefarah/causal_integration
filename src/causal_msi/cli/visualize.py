"""``visualize`` CLI: render diagnostic figures into ``results/``.

Everything downstream of training is driven by the saved checkpoint:

- the MODEL weights and CONFIG come from ``checkpoints/model_seed{N}.pth``;
- the DATASET is the exact one the model trained on (``checkpoints/dataset.npz``,
  written by ``train``);
- the TRAIN / VAL / TEST split indices are read from the checkpoint, so metrics are
  computed on the same partition the model was trained/validated/tested on.

All comparisons are network output vs the ANALYTICAL observer targets (``Y``, the
Kording 2007 optimal estimates). Per-output R^2 is reported for train, val and test.

Figure families:
1. ENCODING -- tuning curves, population vectors, code heatmaps, gain/Poisson checks.
2. MODEL QUALITY -- loss curves + per-output R^2 (train/val/test).
3. OUTPUT ASSESSMENT -- network-vs-analytical scatter + error histograms (test split).
4. ANALYSIS -- common-cause posterior vs disparity + integration comparison (test).
5. IMPLICIT COMMON CAUSE -- p(C=1) implied by the output vs analytical + true label.

If no checkpoint exists, a small demo model is quick-trained as a fallback.
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
from causal_msi.analysis import common_cause as cc  # noqa: E402
from causal_msi.analysis import integration as integ  # noqa: E402
from causal_msi.analysis.decoding import extract_activations  # noqa: E402
from causal_msi.analysis.performance import (  # noqa: E402
    OUTPUT_NAMES,
    error_distributions,
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
from causal_msi.generative import Dataset, build_dataset  # noqa: E402
from causal_msi.io import load_checkpoint, load_dataset, save_results  # noqa: E402
from causal_msi.models import FeedforwardMSI, build_model  # noqa: E402
from causal_msi.training import TrainResult, train_one  # noqa: E402
from causal_msi.utils import seed_everything  # noqa: E402

app = typer.Typer(add_completion=False, help="Render diagnostic figures into results/.")

FloatArray = np.ndarray
Splits = dict[str, np.ndarray]


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
    """Render encoding-verification figures from the deterministic encoder set."""
    rng = seed_everything(cfg.seed)
    enc = cfg.encoding
    encoders = Encoders.build(rng, enc)
    field = np.linspace(enc.visual_field[0], enc.visual_field[1], 200)
    unit_gain = np.ones_like(field)

    vis = gaussian_rf_code(field, encoders.rf_centers, enc.rf_width, unit_gain)
    ph = push_pull_code(field, encoders.prop_hand, unit_gain)
    pe = push_pull_code(field, encoders.prop_eye, unit_gain)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.tuning_curves(field, vis, axes[0], "visual_hand (Gaussian RF)", max_units=25)
    viz.tuning_curves(field, ph, axes[1], "prop_hand (push-pull)", max_units=25)
    viz.tuning_curves(field, pe, axes[2], "prop_eye (push-pull)", max_units=25)
    fig.suptitle("Encoding tuning curves (gain=1, no Poisson)")
    _save(fig, results_dir, "01_encoding_tuning_curves.png")

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.input_heatmap(vis, axes[0], "visual_hand", ylabel="stimulus (low->high)")
    viz.input_heatmap(ph, axes[1], "prop_hand", ylabel="stimulus (low->high)")
    viz.input_heatmap(pe, axes[2], "prop_eye", ylabel="stimulus (low->high)")
    fig.suptitle("Population code vs stimulus (mean rate)")
    _save(fig, results_dir, "03_encoding_heatmaps.png")

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
# Load the trained model + its exact dataset + splits (or quick-train a demo)
# --------------------------------------------------------------------------- #
def _load_for_viz(
    cfg: Config, checkpoint: Path | None, checkpoint_dir: Path, n_trials: int, epochs: int
) -> tuple[FeedforwardMSI, Config, Dataset, Splits, TrainResult | None]:
    """Return ``(model_cpu, eval_cfg, dataset, splits, result)`` for visualisation.

    Checkpoint mode (preferred): the model, config and split indices come from the
    checkpoint, and the dataset is the exact one ``train`` saved
    (``checkpoints/dataset.npz``); if that file is missing it is rebuilt
    deterministically from the checkpoint's config (same seed/n_trials -> same data).

    Demo mode (no checkpoint): a small model is quick-trained and its in-memory
    dataset + splits are used.
    """
    if checkpoint is not None:
        payload = load_checkpoint(checkpoint)
        eval_cfg = Config.model_validate(payload["config"])
        model = build_model(int(payload["input_dim"]), eval_cfg.model)
        model.load_state_dict(payload["state_dict"])
        splits = {k: np.asarray(v).astype(int) for k, v in payload["splits"].items()}
        ds_path = checkpoint_dir / "dataset.npz"
        if ds_path.exists():
            dataset = load_dataset(ds_path)
        else:
            typer.echo(f"  {ds_path} not found -> rebuilding dataset from checkpoint config")
            dataset = build_dataset(seed_everything(eval_cfg.seed), eval_cfg)
        return model.to("cpu").eval(), eval_cfg, dataset, splits, None

    demo_cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={"n_trials": n_trials, "epochs": epochs, "n_seeds": 1}
            )
        }
    )
    dataset = build_dataset(seed_everything(demo_cfg.seed), demo_cfg)
    result = train_one(dataset, demo_cfg, seed=demo_cfg.seed, progress=False)
    assert result.splits is not None
    return result.model.to("cpu").eval(), demo_cfg, dataset, result.splits, result


def _predict(model: FeedforwardMSI, x: FloatArray) -> FloatArray:
    """Run the model on the full input matrix (CPU) and return outputs as numpy."""
    with torch.no_grad():
        out: FloatArray = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    return out


def _per_split_metrics(
    pred: FloatArray, target: FloatArray, splits: Splits
) -> dict[str, dict[str, float]]:
    """Per-output R^2 of network vs analytical target, for each split."""
    return {
        split: {r.name: r.r2 for r in per_output_regression(pred[idx], target[idx])}
        for split, idx in splits.items()
    }


# --------------------------------------------------------------------------- #
# 2. Model-quality figures (loss curve + per-split R^2)
# --------------------------------------------------------------------------- #
def _quality_figures(
    results_dir: Path,
    result: TrainResult | None,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Loss curves (demo mode) and a grouped per-output R^2 bar chart across splits."""
    if result is not None and result.history:
        train_total = np.array([e.train_total for e in result.history])
        val_total = np.array([e.val_total for e in result.history])
        comps = {
            k: np.array([e.components.get(k, np.nan) for e in result.history])
            for k in ("est", "var")
        }
        fig, ax = plt.subplots(figsize=(7, 4.5))
        viz.loss_curves(train_total, val_total, ax, comps)
        ax.set_title(f"Training (best val={result.best_val:.3f} @ epoch {result.best_epoch})")
        _save(fig, results_dir, "05_training_loss.png")
    else:
        typer.echo("  (no training history in checkpoint mode -> no loss curve)")

    splits = list(metrics)
    xpos = np.arange(len(OUTPUT_NAMES))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ymin = 0.0
    for i, s in enumerate(splits):
        vals = [metrics[s][n] for n in OUTPUT_NAMES]
        ymin = min(ymin, *vals)
        bars = ax.bar(xpos + i * width, vals, width, label=s)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7
            )
    ax.set_xticks(xpos + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(list(OUTPUT_NAMES))
    ax.set_ylabel("R^2 (network vs analytical)")
    ax.set_ylim(min(ymin, 0.0) - 0.05, 1.05)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title("Per-output R^2 by split")
    ax.legend(fontsize=8)
    _save(fig, results_dir, "06_output_r2_bars.png")


# --------------------------------------------------------------------------- #
# 3. Output-assessment figures (on the TEST split)
# --------------------------------------------------------------------------- #
def _output_figures(
    results_dir: Path, pred: FloatArray, target: FloatArray, test_idx: np.ndarray
) -> None:
    """Network-vs-analytical scatter and per-output error histograms on the test split."""
    p, y = pred[test_idx], target[test_idx]
    regs = per_output_regression(p, y)
    sub = np.random.default_rng(0).choice(p.shape[0], size=min(2000, p.shape[0]), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    flat = axes.ravel()
    for i, (name, reg) in enumerate(zip(OUTPUT_NAMES, regs, strict=True)):
        viz.regression_scatter(y[sub, i], p[sub, i], flat[i], label=name)
        flat[i].set_title(f"{name}: R^2={reg.r2:.3f}, slope={reg.slope:.2f}")
    fig.suptitle("TEST split: network vs analytical optimal estimate (Kording Eqs. 9/10 + vars)")
    _save(fig, results_dir, "07_output_decoding_scatter.png")

    errs = error_distributions(p, y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, name in enumerate(OUTPUT_NAMES):
        viz.error_histogram(errs[name], axes.ravel()[i])
        axes.ravel()[i].set_title(f"{name} error  (mean={errs[name].mean():.2f})")
    fig.suptitle("TEST split: per-output error (network - analytical)")
    _save(fig, results_dir, "08_output_error_histograms.png")


# --------------------------------------------------------------------------- #
# 4. Analysis figures (on the TEST split)
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
    results_dir: Path,
    cfg: Config,
    model: FeedforwardMSI,
    dataset: Dataset,
    pred: FloatArray,
    test_idx: np.ndarray,
) -> None:
    """Common-cause-vs-disparity and integration comparison, on the test split."""
    t = dataset.targets
    x_vis_body = dataset.measurements.x_vis + dataset.measurements.x_eye
    disparity = (x_vis_body - dataset.measurements.x_prop)[test_idx]
    edges = np.linspace(-30, 30, 21)
    c_an, m_an = _bin_means(disparity, t.p_common[test_idx], edges)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    viz.binned_curve(
        c_an,
        {"analytical p(C=1)": m_an},
        ax,
        xlabel="body-frame disparity (deg)",
        ylabel="p(C=1)",
        title="TEST split: analytical common-cause posterior vs disparity",
    )
    _save(fig, results_dir, "09_proportion_common_vs_disparity.png")

    analytical = integ.analytical_model_averaged_estimate(t)[test_idx]
    network_est = integ.network_optimal_estimate(pred, "vis")[test_idx]
    msl = extract_activations(model, dataset.X[test_idx])["msl"]
    decoded = integ.population_decoded_estimate(msl, analytical, cfg.analysis.decoder)
    cmp_net = integ.compare_estimates(analytical, network_est)

    sub = np.random.default_rng(0).choice(
        len(analytical), size=min(2000, len(analytical)), replace=False
    )
    w_net = integ.fusion_weight_curve(t.seg_vis_mu[test_idx], t.fused_mu[test_idx], network_est)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    viz.regression_scatter(analytical[sub], network_est[sub], axes[0], label="network")
    axes[0].set_title(
        f"Network vs analytical optimal estimate (visual)\nR^2={cmp_net.r2:.3f}  "
        f"(population-decoded R^2={decoded.r2:.3f})"
    )
    axes[0].set_xlabel("analytical estimate (deg)")
    axes[0].set_ylabel("network estimate (deg)")
    finite = np.isfinite(w_net)
    fsub = sub[finite[sub]]
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="optimal")
    axes[1].plot(t.p_common[test_idx][fsub], np.clip(w_net[fsub], -0.5, 1.5), ".", ms=3, alpha=0.4)
    axes[1].set_xlabel("analytical p(C=1)")
    axes[1].set_ylabel("implied fusion weight")
    axes[1].set_title("Network's implied fusion weight vs p(C=1)")
    axes[1].legend(fontsize=8)
    _save(fig, results_dir, "10_integration_estimates.png")


# --------------------------------------------------------------------------- #
# 5. Implied common-cause posterior (on the TEST split)
# --------------------------------------------------------------------------- #
def _common_cause_figure(
    results_dir: Path, dataset: Dataset, pred: FloatArray, test_idx: np.ndarray
) -> None:
    """Recover the network-implied p(C=1) and test it against analytical + true C."""
    t = dataset.targets
    implied_full = cc.inferred_common_cause_posterior(pred, t)
    implied = implied_full[test_idx]
    analytical = t.p_common[test_idx]
    true_c = dataset.latents.C[test_idx]

    cmp = cc.compare_common_cause(implied, analytical)
    auc_net = cc.discrimination_auc(implied, true_c)
    auc_an = cc.discrimination_auc(analytical, true_c)
    split = cc.pc_by_true_cause(implied, true_c)

    m = np.isfinite(implied)
    sub = np.random.default_rng(0).choice(int(m.sum()), size=min(2000, int(m.sum())), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    a_im, a_an = implied[m][sub], analytical[m][sub]
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].scatter(a_an, a_im, s=6, alpha=0.3)
    axes[0].set_xlabel("analytical p(C=1)")
    axes[0].set_ylabel("network-implied p(C=1)")
    axes[0].set_title(f"Implied vs analytical\nR^2={cmp['r2']:.3f}  r={cmp['pearson_r']:.3f}")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].hist(
        implied[(true_c == 1) & m], bins=30, alpha=0.6, density=True, label="true C=1 (common)"
    )
    axes[1].hist(
        implied[(true_c == 2) & m], bins=30, alpha=0.6, density=True, label="true C=2 (separate)"
    )
    axes[1].set_xlabel("network-implied p(C=1)")
    axes[1].set_ylabel("density")
    axes[1].set_title(
        f"Implied p(C=1) by true cause\nAUC={auc_net:.3f} (separation={split['separation']:+.3f})"
    )
    axes[1].legend(fontsize=8)

    viz.bar_metrics(
        ["network\nimplied", "analytical\n(ceiling)"],
        [auc_net, auc_an],
        axes[2],
        ylabel="ROC-AUC (C=1 vs C=2)",
        title="Common-cause discrimination",
        ylim=(0.5, 1.0),
    )
    axes[2].axhline(0.5, color="grey", lw=1, ls=":")
    fig.suptitle("TEST split: implicit common-cause posterior recovered from the output")
    _save(fig, results_dir, "11_implied_common_cause.png")
    typer.echo(
        f"  implied vs analytical R^2={cmp['r2']:.3f} | "
        f"AUC implied={auc_net:.3f} analytical={auc_an:.3f}"
    )


def _print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Print the per-split per-output R^2 table to the console."""
    splits = list(metrics)
    typer.echo("  per-output R^2 (network vs analytical):")
    typer.echo("    " + f"{'output':<10}" + "".join(f"{s:>9}" for s in splits))
    for name in OUTPUT_NAMES:
        typer.echo("    " + f"{name:<10}" + "".join(f"{metrics[s][name]:9.3f}" for s in splits))


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config (used for the demo fallback).")] = Path(
        "configs/default.yaml"
    ),
    results_dir: Annotated[Path, typer.Option(help="Output directory for figures.")] = Path(
        "results"
    ),
    checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint to visualise. Default: auto-find latest.")
    ] = None,
    checkpoint_dir: Annotated[
        Path, typer.Option(help="Where to look for trained checkpoints + dataset.npz.")
    ] = Path("checkpoints"),
    n_trials: Annotated[
        int, typer.Option(help="Trials for the demo (no-checkpoint) train.")
    ] = 8000,
    epochs: Annotated[int, typer.Option(help="Epochs for the demo (no-checkpoint) train.")] = 100,
) -> None:
    """Render all diagnostic figures into ``results_dir`` from the saved model.

    By default visualises the latest checkpoint under ``checkpoint_dir`` and its
    saved dataset/splits; quick-trains a demo model only if none exists.
    """
    cfg = load_config(config)
    results_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        candidate = checkpoint_dir / f"model_seed{cfg.seed}.pth"
        if candidate.exists():
            checkpoint = candidate
        else:
            found = (
                sorted(checkpoint_dir.glob("model_seed*.pth")) if checkpoint_dir.exists() else []
            )
            checkpoint = found[0] if found else None

    typer.echo("[1/5] Encoding figures")
    _encoding_figures(results_dir, cfg)

    typer.echo("[2/5] Loading model + dataset + splits")
    if checkpoint is not None:
        typer.echo(f"  using checkpoint: {checkpoint}")
    else:
        typer.echo("  no checkpoint found -> quick-training a demo model")
    model, eval_cfg, dataset, splits, result = _load_for_viz(
        cfg, checkpoint, checkpoint_dir, n_trials, epochs
    )

    pred = _predict(model, dataset.X)
    metrics = _per_split_metrics(pred, dataset.Y, splits)
    _print_metrics(metrics)
    save_results(metrics, results_dir / "metrics.json")

    typer.echo("[3/5] Model-quality + output figures")
    _quality_figures(results_dir, result, metrics)
    _output_figures(results_dir, pred, dataset.Y, splits["test"])

    typer.echo("[4/5] Analysis figures")
    _analysis_figures(results_dir, eval_cfg, model, dataset, pred, splits["test"])

    typer.echo("[5/5] Implied common-cause posterior")
    _common_cause_figure(results_dir, dataset, pred, splits["test"])

    typer.echo(f"Done. Figures saved under {results_dir}/")


if __name__ == "__main__":
    app()
