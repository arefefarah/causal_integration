"""Training loop, early stopping, and multi-seed runner.

Wraps a :class:`causal_msi.generative.Dataset` in torch ``DataLoader``s, trains a
:class:`causal_msi.models.FeedforwardMSI` with the configured optimiser
(rprop / adam), early-stops on validation loss, checkpoints the best model, and
logs per-output metrics. A multi-seed runner repeats training across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from causal_msi.config import Config
from causal_msi.generative import Dataset, build_dataset, train_val_test_split
from causal_msi.io import save_checkpoint
from causal_msi.losses import causal_loss, integration_loss
from causal_msi.models import FeedforwardMSI, build_model
from causal_msi.utils import get_device, seed_everything


@dataclass
class EpochLog:
    """Metrics for a single epoch."""

    epoch: int
    train_total: float
    val_total: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Outcome of one training run."""

    model: FeedforwardMSI
    best_val: float
    best_epoch: int
    history: list[EpochLog]
    seed: int
    checkpoint_path: Path | None = None


def _make_loaders(
    dataset: Dataset, config: Config, rng: np.random.Generator, device: torch.device
) -> tuple[
    DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]
]:
    """Split a dataset and wrap each partition in a DataLoader.

    Parameters
    ----------
    dataset
        Materialised dataset.
    config
        Full configuration (split fractions, batch size).
    rng
        Seeded numpy generator for the shuffle/split.
    device
        Target device for the tensors.

    Returns
    -------
    tuple of DataLoader
        ``(train_loader, val_loader, test_loader)``.
    """
    x = torch.as_tensor(dataset.X, dtype=torch.float32, device=device)
    y = torch.as_tensor(dataset.Y, dtype=torch.float32, device=device)
    train_idx, val_idx, test_idx = train_val_test_split(x.shape[0], config.training.split, rng)

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader[tuple[Tensor, ...]]:
        ds = TensorDataset(x[idx], y[idx])
        return DataLoader(ds, batch_size=config.training.batch_size, shuffle=shuffle)

    return loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)


def _make_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build the configured optimiser (rprop or adam)."""
    name = config.training.optimizer
    if name == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=config.training.lr)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.training.lr)
    raise ValueError(f"unknown optimizer {name!r}")


def _loss_fn(pred: Tensor, target: Tensor, config: Config) -> tuple[Tensor, dict[str, float]]:
    """Dispatch to the head-appropriate loss."""
    if config.model.head_type == "causal":
        return causal_loss(pred, target, config.training.loss_weights, config.training.pc_loss)
    return integration_loss(pred, target, config.training.loss_weights)


def _evaluate(
    model: FeedforwardMSI, loader: DataLoader[tuple[Tensor, ...]], config: Config
) -> dict[str, float]:
    """Compute mean loss components over a loader without gradient tracking."""
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            _, comp = _loss_fn(model(xb), yb, config)
            for k, v in comp.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def train_one(
    dataset: Dataset,
    config: Config,
    seed: int,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | None = None,
    progress: bool = True,
) -> TrainResult:
    """Train a single model with early stopping and best-checkpoint saving.

    Parameters
    ----------
    dataset
        Materialised dataset (inputs + targets).
    config
        Full configuration.
    seed
        Seed for this run (model init, shuffling).
    checkpoint_dir
        If given, the best model is saved to ``{dir}/model_seed{seed}.pth``.
    device
        Target device; defaults to :func:`causal_msi.utils.get_device`.
    progress
        Whether to show a tqdm progress bar.

    Returns
    -------
    TrainResult
        The trained (best) model, its validation loss, and the epoch history.
    """
    rng = seed_everything(seed)
    device = device or get_device()
    input_dim = dataset.X.shape[1]
    model = build_model(input_dim, config.model).to(device)
    optimizer = _make_optimizer(model, config)

    train_loader, val_loader, _ = _make_loaders(dataset, config, rng, device)

    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] = {}
    patience = config.training.early_stopping_patience
    since_improved = 0
    history: list[EpochLog] = []

    epoch_iter = range(config.training.epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc=f"seed {seed}", leave=False)

    for epoch in epoch_iter:
        model.train()
        train_totals = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss, _ = _loss_fn(model(xb), yb, config)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            train_totals += float(loss.detach())
            n_batches += 1
        train_total = train_totals / max(n_batches, 1)

        val_comp = _evaluate(model, val_loader, config)
        val_total = val_comp.get("total", float("inf"))
        history.append(EpochLog(epoch, train_total, val_total, val_comp))

        if val_total < best_val - 1e-9:
            best_val = val_total
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    ckpt_path: Path | None = None
    if checkpoint_dir is not None:
        metrics = {"best_val": best_val, "best_epoch": best_epoch}
        ckpt_path = save_checkpoint(
            model,
            config,
            metrics,
            Path(checkpoint_dir) / f"model_seed{seed}.pth",
            extra={"seed": seed, "input_dim": input_dim},
        )

    return TrainResult(
        model=model,
        best_val=best_val,
        best_epoch=best_epoch,
        history=history,
        seed=seed,
        checkpoint_path=ckpt_path,
    )


def run_multiseed(
    config: Config,
    checkpoint_dir: str | Path | None = None,
    dataset: Dataset | None = None,
    base_seed: int | None = None,
) -> list[TrainResult]:
    """Train ``config.training.n_seeds`` models with distinct seeds.

    Parameters
    ----------
    config
        Full configuration.
    checkpoint_dir
        Directory for per-seed checkpoints.
    dataset
        Optional pre-built dataset; if None, one is generated per the config using
        ``base_seed`` (so all seeds share the same data, varying only init/shuffle).
    base_seed
        Base seed; the ``i``-th run uses ``base_seed + i``. Defaults to
        ``config.seed``.

    Returns
    -------
    list of TrainResult
        One result per seed.
    """
    base_seed = config.seed if base_seed is None else base_seed
    if dataset is None:
        rng = seed_everything(base_seed)
        dataset = build_dataset(rng, config)

    results: list[TrainResult] = []
    for i in range(config.training.n_seeds):
        results.append(
            train_one(dataset, config, seed=base_seed + i, checkpoint_dir=checkpoint_dir)
        )
    return results
