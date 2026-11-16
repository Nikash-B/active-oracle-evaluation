"""Search agent: play a frozen value network with 2-ply lookahead.

The search is a 2-ply expectimax over the 21 opponent rolls, with a top-k 1-ply pre-filter
(:func:`active_oracle_evaluation.backgammon.search.twoply.two_ply_scores`). This is the
"Network, 2-ply search" agent of the paper's table of PR at four search depths (appendix
"Validity of the oracle rating").

Move selection uses
:func:`active_oracle_evaluation.backgammon.agents._selection.select_by_score`.
"""

from __future__ import annotations

import torch

from active_oracle_evaluation.backgammon.agents._selection import select_by_score
from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.search.evaluator import ModelEvaluator
from active_oracle_evaluation.backgammon.search.twoply import two_ply_scores


class SearchAgent(Agent):
    """Selects moves by 2-ply search over a frozen value network."""

    def __init__(self, model: torch.nn.Module, encoder: Encoder) -> None:
        """Wrap ``model`` and ``encoder`` in a 2-ply search."""
        self.evaluator = ModelEvaluator(model, encoder)
        self.name = "search2ply"

    def choose_move(self, game: Game) -> Move:
        """Return the legal move with the best 2-ply search score."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        scores = two_ply_scores(moves, self.evaluator)
        return select_by_score(moves, scores)
