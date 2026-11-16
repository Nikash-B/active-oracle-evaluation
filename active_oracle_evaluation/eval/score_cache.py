"""Loader for the per-agent regret cache.

The cache is written by ``python -m active_oracle_evaluation.experiments.score_panel_cache``.

Agent losses on a fixed decision suite are deterministic, so the suite is scored once and every
downstream analysis (selection, estimator baselines, tail risk, phase breakdowns) reads the same
``.npz``. A source game is identified by the pair ``(agent_source, game_id)``, because the game
counter restarts at 0 in each per-agent pool; ``cluster_ids`` codes that pair densely and is the
resampling unit of the clustered bootstrap.

Loading needs no torch, no oracle and no agent code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Multiplier from per-decision equity regret onto the gnubg PR scale (paper, appendix
#: "Statistical protocol: derivations and definitions": 500).
PR_PER_EQUITY = 500.0


class ScoreCache:
    """A loaded score cache.

    Attributes
    ----------
    names
        Agent names sorted best (lowest PR) first.
    losses
        ``{agent: regret vector}``, index-aligned with ``phases`` / ``position_ids``.
    pr
        ``{agent: PR}`` from the JSON sidecar, else ``mean(regret) * 500``.
    cluster_ids
        Source game of each decision, densely coded from ``(agent_source, game_id)``. This, not
        ``game_ids``, is the input to
        :func:`~active_oracle_evaluation.eval.tail_risk.game_bootstrap_indices`.
    game_ids
        Raw ``metadata['game_id']`` of each decision, unique only within a per-agent pool.
    agent_sources
        The per-agent pool each decision was generated in (``""`` when unrecorded).
    """

    def __init__(self, path: str | Path):
        """Load the cache at ``path``."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        names = [str(x) for x in data["names"]]
        self.path = path
        self.phases = np.asarray([str(x) for x in data["phases"]])
        self.position_ids = np.asarray([str(x) for x in data["position_ids"]])
        losses = np.asarray(data["losses"], dtype=float)
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        pr = meta.get("pr", {})
        self.pr = {
            name: float(pr.get(name, losses[i].mean() * PR_PER_EQUITY))
            for i, name in enumerate(names)
        }
        self.names = sorted(names, key=lambda name: self.pr[name])
        self.losses = {name: losses[i] for i, name in enumerate(names)}
        self.n_pos = int(losses.shape[1])
        self._resolve_clusters(data)
        self.n_games = int(np.unique(self.cluster_ids).size)

    def _resolve_clusters(self, npz) -> None:
        """Set ``agent_sources``, ``game_ids`` and ``cluster_ids`` from the cache's provenance."""
        self.agent_sources = np.asarray([str(x) for x in npz["agent_sources"]])
        self.game_ids = np.asarray(npz["game_ids"], dtype=np.int64)
        self.cluster_ids = _dense_codes(self.agent_sources, self.game_ids)


def _dense_codes(agent_sources: np.ndarray, game_ids: np.ndarray) -> np.ndarray:
    """Return a dense integer code per distinct ``(agent_source, game_id)`` pair, in sorted order.

    ``np.unique`` sorts, so two loads of the same cache label the clusters identically.
    """
    keys = np.array([f"{a}\x00{int(g)}" for a, g in zip(agent_sources, game_ids)])
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int64)
