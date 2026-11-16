"""Settings of every experiment, read from config/experiments.toml."""

import tomllib

from active_oracle_evaluation.paths import REPO

CONFIG_FILE = REPO / "config" / "experiments.toml"


def settings(experiment):
    """Return the settings table of one experiment."""
    with open(CONFIG_FILE, "rb") as f:
        tables = tomllib.load(f)
    if experiment not in tables:
        raise KeyError(f"no [{experiment}] table in {CONFIG_FILE}")
    return tables[experiment]
