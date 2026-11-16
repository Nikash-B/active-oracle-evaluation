"""Agent that plays the move chosen by chance-aware MCTS.

Runs :class:`~active_oracle_evaluation.backgammon.search.mcts.MCTS` from the current position and plays the
root's most-visited move (temperature zero, as in the paper's appendix "Phase diagnostics and
search ablation", paragraph "Search ablation").
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.search.mcts import MCTS, AfterstateExpander, MCTSConfig


class MCTSAgent(Agent):
    """Plays the move MCTS visited most from the current (board, roll)."""

    name = "mcts"

    def __init__(
        self,
        model: object,
        encoder: Encoder,
        *,
        # 100 and 200 simulations, as in the paper's appendix "Phase diagnostics and search
        # ablation", paragraph "Search ablation".
        simulations: int = 100,
        # AlphaZero c_puct of the hyperparameter table in the paper's appendix "Agent
        # configurations".
        c_puct: float = 1.5,
    ) -> None:
        """Build the search config."""
        self.expander = AfterstateExpander(model, encoder)
        self.config = MCTSConfig(simulations=simulations, c_puct=c_puct)

    def choose_move(self, game: Game) -> Move:
        """Search from the current position and play the root's most-visited move."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        result = MCTS(self.expander, self.config).search(game.board, game.dice, game.rng)
        return result.moves[int(np.argmax(result.visit_counts))]
