"""Where everything lives. Import these instead of hard-coding paths.

    research/
      configs/          yaml parameter files
      data/             generated datasets (.npz)  -- gitignored
      results/<run>/    one folder per run         -- gitignored
        config.yaml       the exact config that produced it
        model.pt          trained weights + splits
        metrics.json      every number the analysis produced
        figures/
          inputs/         encoding / stimulus checks
          training/       loss curves
          model/          network vs analytical observer
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

FIGURE_GROUPS = ("inputs", "training", "model")


def run_dir(name, create=True):
    """Path to results/<name>/, with its figure sub-folders."""
    path = RESULTS / name
    if create:
        for group in FIGURE_GROUPS:
            (path / "figures" / group).mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(name):
    """Path to data/<name>.npz."""
    name = str(name)
    return DATA / (name if name.endswith(".npz") else f"{name}.npz")
