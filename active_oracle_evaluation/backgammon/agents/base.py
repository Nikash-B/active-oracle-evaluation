"""The agent contract.

An :class:`Agent` picks one of the legal moves for a game whose dice are already
rolled. Evaluation code only calls :meth:`Agent.choose_move`.
"""

from __future__ import annotations

import abc

from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move


class Agent(abc.ABC):
    """Base class for move-selecting policies."""

    name: str = "agent"

    @abc.abstractmethod
    def choose_move(self, game: Game) -> Move:
        """Return one legal move for ``game`` (dice already rolled).

        Implementations may assume ``game.legal_moves()`` is non-empty (the
        generator always returns at least a pass), and must return one of those
        moves.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return the class name and agent name."""
        return f"{type(self).__name__}(name={self.name!r})"
