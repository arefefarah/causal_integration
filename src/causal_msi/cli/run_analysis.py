"""``run-analysis`` CLI: run analysis subcommands on a trained checkpoint.

Each subcommand maps to an analysis module. The commands wire up data + model
loading and call into the (partly ``TODO(science)``) analysis functions; they are
the integration points the author fills in alongside the science.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from causal_msi.config import load_config
from causal_msi.io import load_checkpoint
from causal_msi.models import build_model

app = typer.Typer(add_completion=False, help="Run the analysis suite on a checkpoint.")


def _load_model_and_config(checkpoint: Path, config: Path) -> tuple[object, object]:
    """Load a config and rebuild the model from a checkpoint payload."""
    cfg = load_config(config)
    payload = load_checkpoint(checkpoint)
    input_dim = int(payload["input_dim"])
    model = build_model(input_dim, cfg.model)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


@app.command()
def decoding(
    checkpoint: Annotated[Path, typer.Option(help="Causal-model checkpoint.")],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option(help="Test .npz dataset.")] = Path("data/dataset.npz"),
    twin: Annotated[Path | None, typer.Option(help="integration_only twin checkpoint.")] = None,
) -> None:
    """Layer-wise p(C=1) decodability (SIL vs MSL) and emergent-vs-imposed control."""
    typer.echo("TODO: wire causal_msi.analysis.decoding once the science is implemented")


@app.command()
def integration(
    checkpoint: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option()] = Path("data/dataset.npz"),
) -> None:
    """Run the integration.py suite (reconstructed / decoded / strategy fits)."""
    typer.echo("TODO: wire causal_msi.analysis.integration once the science is implemented")


@app.command()
def mechanism(
    checkpoint: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option()] = Path("data/dataset.npz"),
) -> None:
    """Congruent/opposite + Bayes-factor + ablation mechanism analyses."""
    typer.echo("TODO: wire congruent_opposite / bayes_factor / ablation")


@app.command()
def indices(
    checkpoint: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option()] = Path("data/dataset.npz"),
) -> None:
    """AI / RE / RA + gain indices (divisive-normalization comparison)."""
    typer.echo("TODO: wire causal_msi.analysis.indices once the science is implemented")


@app.command()
def geometry(
    checkpoint: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option()] = Path("data/dataset.npz"),
) -> None:
    """RSA / dimensionality + reference-frame-of-causality."""
    typer.echo("TODO: wire causal_msi.analysis.geometry once the science is implemented")


@app.command()
def rf_shifts(
    checkpoint: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/default.yaml"),
    dataset: Annotated[Path, typer.Option()] = Path("data/dataset.npz"),
) -> None:
    """Receptive-field shift vs eye position per layer."""
    typer.echo("TODO: wire causal_msi.analysis.rf_shifts once the science is implemented")


if __name__ == "__main__":
    app()
