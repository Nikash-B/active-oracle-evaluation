"""1-ply static evaluation.

Scores each legal move by the value *to the mover* of the afterstate it reaches:

    value_to_mover(a) = +terminal_points(a)   if a ends the game (mover won)
                        -V(a)                  otherwise

where ``V(a)`` is the network's equity for the player to move at ``a`` (the
opponent). The 1-ply search agent and the 2-ply search (pre-filter and leaves)
call this function.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.search.evaluator import (
    ModelEvaluator,
    is_terminal,
    terminal_points,
)


def one_ply_scores(moves: list[Move], evaluator: ModelEvaluator) -> np.ndarray:
    """Value-to-mover for each move, terminal-aware, in one batched eval."""
    scores = np.empty(len(moves), dtype=np.float32)
    nonterminal_idx: list[int] = []
    nonterminal_boards = []
    for i, move in enumerate(moves):
        a = move.afterstate
        if is_terminal(a):
            scores[i] = terminal_points(a)  # mover just won
        else:
            nonterminal_idx.append(i)
            nonterminal_boards.append(a)
    if nonterminal_boards:
        values = evaluator.evaluate(nonterminal_boards)
        for j, i in enumerate(nonterminal_idx):
            scores[i] = -values[j]  # value to mover = -V(opponent's afterstate)
    return scores
