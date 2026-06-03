"""``generate-data`` CLI: build a dataset from a config and write it to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from causal_msi.config import load_config
from causal_msi.generative import build_dataset
from causal_msi.io import save_dataset
from causal_msi.utils import seed_everything

app = typer.Typer(add_completion=False, help="Generate a population-coded dataset.")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Path to a YAML config.")] = Path(
        "configs/default.yaml"
    ),
    out: Annotated[Path, typer.Option(help="Output .npz path.")] = Path("data/dataset.npz"),
    n_trials: Annotated[int | None, typer.Option(help="Override number of trials.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override seed.")] = None,
) -> None:
    """Sample latents, render measurements, encode inputs, and save ``(X, Y)``.

    Note: this requires the ``TODO(science)`` analytical-observer functions in
    ``causal_msi.generative`` to be implemented; until then it raises
    ``NotImplementedError`` by design.
    """
    cfg = load_config(config)
    rng = seed_everything(cfg.seed if seed is None else seed)
    dataset = build_dataset(rng, cfg, n_trials=n_trials)
    path = save_dataset(dataset, out)
    typer.echo(f"Wrote dataset with {dataset.X.shape[0]} trials to {path}")


if __name__ == "__main__":
    app()
