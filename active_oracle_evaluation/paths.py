"""Locations of the shipped data, the checkpoints and the experiment outputs."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MODELS = REPO / "models"
OUTPUTS = Path(os.environ.get("OUTPUTS", REPO / "outputs"))


def output_dir(experiment):
    """Return outputs/<experiment>/, creating it if needed."""
    path = OUTPUTS / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_relative(path):
    """Return path relative to the repository root when it lies inside it, else unchanged."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)
