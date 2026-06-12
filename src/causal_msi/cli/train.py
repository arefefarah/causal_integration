"""``train`` CLI: train a model (causal or integration-only), multi-seed."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import typer  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from causal_msi import viz  # noqa: E402
from causal_msi.analysis.performance import OUTPUT_NAMES, per_output_regression  # noqa: E402
from causal_msi.config import Config, load_config  # noqa: E402
from causal_msi.generative import Dataset, build_dataset  # noqa: E402
from causal_msi.io import load_dataset, save_dataset, save_results  # noqa: E402
from causal_msi.training import TrainResult, run_multiseed  # noqa: E402
from causal_msi.utils import seed_everything  # noqa: E402

app = typer.Typer(add_completion=False, help="Train the causal-inference network.")


def _save_loss_curves(results: list[TrainResult], results_dir: Path) -> None:
    """Save a training/validation loss-curve figure per seed into ``results_dir``.

    Parameters
    ----------
    results
        Per-seed training results (each with an epoch ``history``).
    results_dir
        Directory to write ``train_loss_seed{N}.png`` into (created if missing).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if not r.history:
            continue
        train_total = np.array([e.train_total for e in r.history])
        val_total = np.array([e.val_total for e in r.history])
        comps = {
            k: np.array([e.components.get(k, np.nan) for e in r.history]) for k in ("est", "var")
        }
        fig, ax = plt.subplots(figsize=(7, 4.5))
        viz.loss_curves(train_total, val_total, ax, comps)
        ax.set_title(f"seed {r.seed}: best val={r.best_val:.4f} @ epoch {r.best_epoch}")
        path = results_dir / f"train_loss_seed{r.seed}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        typer.echo(f"  wrote {path}")


def _report_metrics(
    results: list[TrainResult], dataset: Dataset, cfg: Config, results_dir: Path
) -> None:
    """Print, save (JSON), and plot per-output R^2 on train/val/test, per seed.

    Defined for the 4-output ``causal`` head (the ``OUTPUT_NAMES`` columns); skipped
    for the integration-only twin.

    Parameters
    ----------
    results
        Per-seed training results (each carrying its ``splits`` indices).
    dataset
        The dataset the models were trained on.
    cfg
        Full configuration.
    results_dir
        Directory for ``train_metrics_seed{N}.json`` / ``train_r2_seed{N}.png``.
    """
    if cfg.model.head_type != "causal":
        typer.echo("  (per-output R^2 readout is defined for the causal head; skipped)")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor(dataset.X, dtype=torch.float32)
    y = dataset.Y
    for r in results:
        if r.splits is None:
            continue
        model = r.model.to("cpu").eval()
        with torch.no_grad():
            pred = model(x).numpy()
        metrics = {
            split: {reg.name: reg.r2 for reg in per_output_regression(pred[idx], y[idx])}
            for split, idx in r.splits.items()
        }
        splits = list(metrics)
        typer.echo(f"  per-output R^2 (seed {r.seed}):")
        typer.echo("    " + f"{'output':<10}" + "".join(f"{s:>9}" for s in splits))
        for name in OUTPUT_NAMES:
            typer.echo("    " + f"{name:<10}" + "".join(f"{metrics[s][name]:9.3f}" for s in splits))
        save_results(metrics, results_dir / f"train_metrics_seed{r.seed}.json")
        _r2_bar_figure(metrics, results_dir / f"train_r2_seed{r.seed}.png", r.seed)
        typer.echo(f"  wrote {results_dir / f'train_metrics_seed{r.seed}.json'}")


def _r2_bar_figure(metrics: dict[str, dict[str, float]], path: Path, seed: int) -> None:
    """Plot a grouped bar chart of per-output R^2 across train/val/test splits."""
    splits = list(metrics)
    xpos = np.arange(len(OUTPUT_NAMES))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ymin = 0.0
    for i, s in enumerate(splits):
        vals = [metrics[s][n] for n in OUTPUT_NAMES]
        ymin = min(ymin, *vals)
        ax.bar(xpos + i * width, vals, width, label=s)
    ax.set_xticks(xpos + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(list(OUTPUT_NAMES))
    ax.set_ylabel("R^2")
    ax.set_ylim(min(ymin, 0.0) - 0.05, 1.05)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title(f"Per-output R^2 by split (seed {seed})")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {path}")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Path to a YAML config.")] = Path(
        "configs/default.yaml"
    ),
    head_type: Annotated[
        str | None, typer.Option(help="Override head_type: causal | integration_only.")
    ] = None,
    dataset_path: Annotated[
        Path | None, typer.Option(help="Pre-generated .npz dataset; if unset, generate one.")
    ] = None,
    checkpoint_dir: Annotated[Path, typer.Option(help="Where to save checkpoints.")] = Path(
        "checkpoints"
    ),
    results_dir: Annotated[
        Path, typer.Option(help="Where to save the per-seed training loss curves.")
    ] = Path("results"),
) -> None:
    """Train ``n_seeds`` models with early stopping and save per-seed checkpoints.

    A training/validation loss-curve figure is written for every seed under
    ``results_dir`` on each run (overwriting the previous one).
    """
    cfg = load_config(config)
    if head_type is not None:
        cfg = cfg.model_copy(
            update={"model": cfg.model.model_copy(update={"head_type": head_type})}
        )

    if dataset_path is not None:
        dataset = load_dataset(dataset_path)
    else:
        rng = seed_everything(cfg.seed)
        dataset = build_dataset(rng, cfg)

    # Persist the EXACT dataset the models train on so that visualize / analysis
    # evaluate on the same trials (and the same train/val/test splits stored in
    # each checkpoint), guaranteeing the metrics match.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, checkpoint_dir / "dataset.npz")

    results = run_multiseed(cfg, checkpoint_dir=checkpoint_dir, dataset=dataset)
    for r in results:
        typer.echo(
            f"seed {r.seed}: best_val={r.best_val:.5f} "
            f"@ epoch {r.best_epoch} -> {r.checkpoint_path}"
        )
    _save_loss_curves(results, results_dir)
    _report_metrics(results, dataset, cfg, results_dir)


if __name__ == "__main__":
    app()
