"""Monte Carlo playout oracle built on the gnubg-nn network.

The playout oracle reuses the network behind the 0-ply and 2-ply oracles
(``eval.oracle.GnubgOracle``) and plays each position to the end with the package's own game
engine, so the three oracles differ only in how deep they look. Both sides play the gnubg-nn
0-ply greedy policy, which picks the afterstate with the highest equity for the mover. Playout
``i`` of every position uses the dice stream ``random.Random(seed + i)``, so the candidates of
one position share their dice (common random numbers) and every estimate is reproducible.

``gnubg_nn`` is imported inside ``RolloutOracle.__init__`` so that
``import active_oracle_evaluation`` stays free of the optional dependency.
"""

from __future__ import annotations

import random

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.eval._gnubg_board import to_gnubg_board
from active_oracle_evaluation.eval.oracle import equity_from_probs


class RolloutPolicy:
    """The gnubg-nn 0-ply greedy move policy that both sides play in a playout."""

    def __init__(self, gnubg_nn) -> None:
        """Keep the gnubg-nn module that evaluates the afterstates."""
        self._g = gnubg_nn

    def _opponent_equity(self, after: Board) -> float:
        """Return the 0-ply equity of an afterstate to the opponent, who acts next."""
        return equity_from_probs(self._g.probabilities(to_gnubg_board(after), 0))

    def choose_move(self, game: Game) -> Move:
        """Return the legal move whose afterstate is worst for the opponent."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        return min(moves, key=lambda m: self._opponent_equity(m.afterstate))


class RolloutOracle:
    """Cubeless equity oracle that averages full playouts of the gnubg-nn 0-ply policy."""

    def __init__(self, *, playouts: int, seed: int) -> None:
        """Import gnubg-nn, seed its RNG and set the playouts per position."""
        import gnubg_nn as g  # lazy: keep the engine import gnubg-free

        if playouts <= 0:
            raise ValueError("playouts must be positive")
        g.set.seed(seed)
        self.playouts = int(playouts)
        self.seed = int(seed)
        self.policy = RolloutPolicy(g)

    def _play_once(self, board: Board, rng: random.Random) -> float:
        """Play ``board`` to the end once; return the signed points to its mover."""
        root_mover = board.turn
        game = Game(board=board, rng=rng)
        while not game.is_over:
            game.roll()
            game.apply(self.policy.choose_move(game))
        return float(game.points(root_mover))

    def equity(self, board: Board) -> float:
        """Return the mean signed points to ``board``'s mover over ``playouts`` playouts."""
        total = sum(
            self._play_once(board, random.Random(self.seed + i)) for i in range(self.playouts)
        )
        return total / self.playouts
