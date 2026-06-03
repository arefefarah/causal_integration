"""``train`` CLI: train a model (causal or integration-only), multi-seed."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from causal_msi.config import load_config
from causal_msi.generative import build_dataset
from causal_msi.io import load_dataset
from causal_msi.training import run_multiseed
from causal_msi.utils import seed_everything

app = typer.Typer(add_completion=False, help="Train the causal-inference network.")


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
) -> None:
    """Train ``n_seeds`` models with early stopping and save per-seed checkpoints.

    Requires the ``TODO(science)`` observer functions to be implemented so a
    dataset can be built / loaded with valid targets.
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

    results = run_multiseed(cfg, checkpoint_dir=checkpoint_dir, dataset=dataset)
    for r in results:
        typer.echo(
            f"seed {r.seed}: best_val={r.best_val:.5f} "
            f"@ epoch {r.best_epoch} -> {r.checkpoint_path}"
        )


if __name__ == "__main__":
    app()
