"""The game state machine: roll -> move -> apply -> terminal -> score.

:class:`Game` is the stateful object most callers interact with. It wraps an
immutable :class:`~.board.Board`, owns the dice and RNG, drives the turn cycle,
and classifies the final result into single / gammon / backgammon. It also records the move history of the game.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.dice import Roll
from active_oracle_evaluation.backgammon.core.dice import roll as roll_dice
from active_oracle_evaluation.backgammon.core.moves import legal_moves
from active_oracle_evaluation.backgammon.core.types import Move, Outcome, Player, ResultKind


@dataclass(frozen=True)
class HistoryEntry:
    """One recorded ply: the move played."""

    move: Move


def classify_result(board: Board, winner: Player) -> ResultKind:
    """Single / gammon / backgammon for a finished position.

    * The loser bore off at least one checker -> single.
    * The loser bore off none, but has no checker on the bar or in the winner's
      home board -> gammon.
    * The loser bore off none and still has a checker on the bar or trapped in
      the winner's home board -> backgammon.
    """
    loser = winner.opponent
    if board.off(loser) >= 1:
        return ResultKind.SINGLE
    if board.bar(loser) > 0:
        return ResultKind.BACKGAMMON
    for point in board.home_range(winner):
        count = board.point(point)
        if (loser is Player.WHITE and count > 0) or (loser is Player.BLACK and count < 0):
            return ResultKind.BACKGAMMON
    return ResultKind.GAMMON


class Game:
    """A playable backgammon game (checker play only - no cube, no match score)."""

    def __init__(self, board: Board | None = None, rng: random.Random | None = None) -> None:
        """Start from ``board`` (default: the opening position) with dice drawn from ``rng``."""
        self.board: Board = board if board is not None else Board.start()
        self.rng: random.Random = rng or random.Random()
        self._dice: Roll | None = None
        self.history: list[HistoryEntry] = []

    @property
    def turn(self) -> Player:
        """The player on turn."""
        return self.board.turn

    @property
    def dice(self) -> Roll | None:
        """The current roll, or ``None`` if dice have not been rolled yet."""
        return self._dice

    def roll(self) -> Roll:
        """Roll the dice for the player on turn; raises if the game is over."""
        if self.is_over:
            raise RuntimeError("cannot roll: the game is already over")
        self._dice = roll_dice(self.rng)
        return self._dice

    def set_dice(self, roll: Roll) -> None:
        """Force the current roll instead of drawing a random one.

        Duplicate-dice evaluation uses this to replay one roll stream across two
        games with colors swapped, without touching the game's RNG.
        """
        if self.is_over:
            raise RuntimeError("cannot set dice: the game is already over")
        self._dice = roll

    def legal_moves(self) -> list[Move]:
        """Legal moves for the current roll. Roll the dice first."""
        if self._dice is None:
            raise RuntimeError("no dice rolled; call roll() before legal_moves()")
        return legal_moves(self.board, self._dice)

    def apply(self, move: Move) -> None:
        """Commit ``move`` (from :meth:`legal_moves`), advancing to the opponent.

        The move's precomputed afterstate becomes the board, and the dice are
        cleared so the next player must roll.
        """
        self.history.append(HistoryEntry(move=move))
        self.board = move.afterstate
        self._dice = None

    @property
    def is_over(self) -> bool:
        """Whether either player has borne off all checkers."""
        return self.board.has_won(Player.WHITE) or self.board.has_won(Player.BLACK)

    @property
    def winner(self) -> Player | None:
        """The winning player, or ``None`` while the game is live."""
        if self.board.has_won(Player.WHITE):
            return Player.WHITE
        if self.board.has_won(Player.BLACK):
            return Player.BLACK
        return None

    @property
    def result_kind(self) -> ResultKind | None:
        """Single / gammon / backgammon for a finished game, or ``None`` while live."""
        winner = self.winner
        return classify_result(self.board, winner) if winner is not None else None

    def outcome(self, perspective: Player = Player.WHITE) -> Outcome | None:
        """Six-class outcome from ``perspective``'s view, or ``None`` if ongoing."""
        winner = self.winner
        if winner is None:
            return None
        return Outcome.from_result(winner, classify_result(self.board, winner), perspective)

    def points(self, perspective: Player = Player.WHITE) -> int:
        """Signed points from ``perspective``'s view (0 while the game is live)."""
        outcome = self.outcome(perspective)
        return outcome.points if outcome is not None else 0
