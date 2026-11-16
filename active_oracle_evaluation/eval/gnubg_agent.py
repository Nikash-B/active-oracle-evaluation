"""GNU Backgammon reference agent: plays the oracle's best move at a fixed ply.

It serves as a strong external reference in two ways. In the anchored Elo round-robin it places a
near-optimal player on the scale. Scored by the same gnubg oracle, it always picks the oracle-best
move, so its PR is 0 by construction.

It lives in ``eval/`` because ``agents/`` does not depend on gnubg-nn.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move


class GnubgAgent(Agent):
    """Greedy agent that picks the afterstate the gnubg oracle rates best for the mover."""

    name = "gnubg"

    # The default plays as gnubg0, the 0-ply engine (the paper's table of PR at four search
    # depths, appendix "Validity of the oracle rating").

    def __init__(self, plies: int = 0) -> None:
        """Build the gnubg oracle that ranks afterstates at ``plies`` ply."""
        from active_oracle_evaluation.eval.oracle import (  # lazy: gnubg-nn only when used
            GnubgOracle,
            OracleConfig,
        )

        self._oracle = GnubgOracle(OracleConfig(plies=plies))

    def choose_move(self, game: Game) -> Move:
        """Return the legal move whose afterstate has the lowest equity for the opponent."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        # The afterstate is evaluated from the opponent's side, so the best move minimises it.
        return min(moves, key=lambda m: self._oracle.equity(m.afterstate))
