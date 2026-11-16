"""A hand-crafted heuristic agent: a fixed, non-learning reference baseline.

The score blends race (pip count), blot-hitting and safety:

    score(afterstate) = (opponent_pips - my_pips)
                        + HIT_BONUS * opponent_on_bar
                        - BLOT_PENALTY * my_blots

The agent picks the legal move whose afterstate maximizes this score and breaks
ties at random.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move, Player

# Score weights chosen for this study; the paper describes the heuristic only qualitatively.
HIT_BONUS = 12.0
BLOT_PENALTY = 4.0


def _count_blots(board: Board, player: Player) -> int:
    """Number of points where ``player`` has a lone, hittable checker."""
    return sum(1 for c in board.points if c == player.sign)


def _score(afterstate: Board, mover: Player) -> float:
    """Heuristic value of an afterstate from ``mover``'s perspective."""
    opponent = mover.opponent
    race = afterstate.pip_count(opponent) - afterstate.pip_count(mover)
    return (
        race + HIT_BONUS * afterstate.bar(opponent) - BLOT_PENALTY * _count_blots(afterstate, mover)
    )


class HeuristicAgent(Agent):
    """Greedy pip-count-plus-safety baseline."""

    name = "heuristic"

    def choose_move(self, game: Game) -> Move:
        """Return a highest-scoring legal move, breaking ties with the game's RNG."""
        mover = game.turn
        moves = game.legal_moves()
        best_score = float("-inf")
        best: list[Move] = []
        for move in moves:
            score = _score(move.afterstate, mover)
            if score > best_score:
                best_score = score
                best = [move]
            elif score == best_score:
                best.append(move)
        return game.rng.choice(best)
