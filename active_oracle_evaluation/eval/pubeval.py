"""Pubeval baseline agent: Tesauro's linear move evaluator, scored by gnubg-nn.

Pubeval lives here rather than in ``agents/`` because it needs gnubg-nn's ``pub_eval_score``.
The agent picks the legal afterstate with the highest score; a higher ``pub_eval_score`` of an
afterstate is better for the player who just moved.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.eval._gnubg_board import to_gnubg_board


class PubevalAgent(Agent):
    """Greedy 1-ply agent over afterstates scored by gnubg-nn's ``pub_eval_score``."""

    name = "pubeval"

    def __init__(self) -> None:
        """Import gnubg-nn lazily so the engine import stays gnubg-free."""
        import gnubg_nn as g

        self._g = g

    def choose_move(self, game: Game) -> Move:
        """Return the legal move whose afterstate has the highest pubeval score."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        return max(moves, key=lambda m: self._g.pub_eval_score(to_gnubg_board(m.afterstate)))
