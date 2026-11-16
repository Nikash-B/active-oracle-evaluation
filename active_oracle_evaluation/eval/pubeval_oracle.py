"""Pubeval as a weak, rank-only equity oracle for the oracle-depth sensitivity check.

The reference oracle is :class:`active_oracle_evaluation.eval.oracle.GnubgOracle`. This module
scores agents under Tesauro's linear ``pub_eval_score`` (the evaluator behind
:class:`active_oracle_evaluation.eval.pubeval.PubevalAgent`) as a weak comparison oracle.

``pub_eval_score`` is uncalibrated: it orders afterstates but is not on the cubeless-equity scale.
Use it only to rank agents by PR order (e.g. Kendall-tau against the gnubg oracles), never to
compare PR magnitudes with gnubg PR.

``gnubg_nn`` is imported inside ``__init__`` so that importing the package does not need it.
"""

from __future__ import annotations

from dataclasses import dataclass

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.eval._gnubg_board import to_gnubg_board


@dataclass(frozen=True)
class PubevalConfig:
    """Config with a ``version`` field, matching ``OracleConfig`` for the suite builder."""

    version: str = "pubeval"


class PubevalOracle:
    """Weak, rank-only equity oracle backed by gnubg-nn's ``pub_eval_score``.

    It follows the :class:`GnubgOracle` contract: ``equity(board)`` is the value to the player to
    act in ``board``. ``pub_eval_score(afterstate)`` is higher when better for the player who just
    moved, so ``equity`` returns its negation. The suite builder stores
    ``-oracle.equity(afterstate)`` as the mover's value, so pubeval's preferred move is the suite's
    best move.
    """

    def __init__(self, cfg: PubevalConfig = PubevalConfig()) -> None:
        """Import gnubg-nn and store the config."""
        import gnubg_nn as g

        self._g = g
        self.cfg = cfg

    def equity(self, board: Board) -> float:
        """Return the negated, uncalibrated pubeval score: the value to the player to act."""
        return -float(self._g.pub_eval_score(to_gnubg_board(board)))
