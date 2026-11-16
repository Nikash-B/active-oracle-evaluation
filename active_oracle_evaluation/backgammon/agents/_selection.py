"""Shared move-selection rule for agents that score afterstates.

Agents differ only in how they score an afterstate; all of them pick a move with
greedy argmax.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.backgammon.core.types import Move


def select_by_score(moves: list[Move], scores: np.ndarray) -> Move:
    """Pick the move with the highest ``scores`` entry (greedy, deterministic argmax)."""
    return moves[int(np.argmax(scores))]
