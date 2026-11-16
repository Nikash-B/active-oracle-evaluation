"""Core value types: players, game results, outcomes, and moves.

These small, dependency-free types are shared across the engine. They are
immutable (frozen dataclasses and enums) so that moves and results can be hashed
and compared, which the afterstate de-duplication in :mod:`.moves`
relies on.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from active_oracle_evaluation.backgammon.core.constants import BAR, OFF


class Player(enum.IntEnum):
    """The two sides. Stored as +1 / -1 so a player doubles as a sign.

    The sign convention matches :class:`~.board.Board`, where a
    point holds a signed checker count (positive = White, negative = Black).
    """

    WHITE = +1
    BLACK = -1

    @property
    def opponent(self) -> "Player":
        """The other player."""
        return Player.BLACK if self is Player.WHITE else Player.WHITE

    @property
    def sign(self) -> int:
        """+1 for White, -1 for Black - i.e. the sign of this player's checkers."""
        return int(self.value)


class ResultKind(enum.IntEnum):
    """Magnitude of a finished game, worth this many points.

    A *gammon* (loser bore off nothing) is worth double; a *backgammon* (loser
    also has a checker in the winner's home board or on the bar) is worth triple.
    The integer values are the point multipliers.
    """

    SINGLE = 1
    GAMMON = 2
    BACKGAMMON = 3


class Outcome(enum.Enum):
    """A finished game from a *single player's* point of view.

    Six mutually exclusive categories, the same classes as the six-way
    outcome-probability head.
    """

    SINGLE_WIN = "W"
    GAMMON_WIN = "WG"
    BACKGAMMON_WIN = "WBG"
    SINGLE_LOSS = "L"
    GAMMON_LOSS = "LG"
    BACKGAMMON_LOSS = "LBG"

    @property
    def points(self) -> int:
        """Signed point value (+1..+3 for wins, -1..-3 for losses)."""
        mag = {
            Outcome.SINGLE_WIN: 1,
            Outcome.GAMMON_WIN: 2,
            Outcome.BACKGAMMON_WIN: 3,
            Outcome.SINGLE_LOSS: -1,
            Outcome.GAMMON_LOSS: -2,
            Outcome.BACKGAMMON_LOSS: -3,
        }
        return mag[self]

    @classmethod
    def from_result(cls, winner: Player, kind: ResultKind, perspective: Player) -> "Outcome":
        """Build the outcome a given ``perspective`` player experiences.

        ``winner``/``kind`` describe the game objectively; this folds in whose
        point of view we want, flipping win<->loss when the perspective player is
        not the winner.
        """
        won = perspective is winner
        win_map = {
            ResultKind.SINGLE: cls.SINGLE_WIN,
            ResultKind.GAMMON: cls.GAMMON_WIN,
            ResultKind.BACKGAMMON: cls.BACKGAMMON_WIN,
        }
        loss_map = {
            ResultKind.SINGLE: cls.SINGLE_LOSS,
            ResultKind.GAMMON: cls.GAMMON_LOSS,
            ResultKind.BACKGAMMON: cls.BACKGAMMON_LOSS,
        }
        return win_map[kind] if won else loss_map[kind]


@dataclass(frozen=True)
class SubMove:
    """A single checker hop that consumes one die.

    Coordinates are **mover-relative** (see :mod:`.constants`):
    ``src``/``dst`` are points 1..24, or the sentinels :data:`BAR` / :data:`OFF`.
    ``die`` is the pip value consumed; ``hit`` records whether this hop landed on
    and sent an opponent blot to the bar.
    """

    src: int
    dst: int
    die: int
    hit: bool = False

    def __str__(self) -> str:
        """Standard notation, e.g. ``bar/20*`` or ``6/off``."""
        src = "bar" if self.src == BAR else str(self.src)
        dst = "off" if self.dst == OFF else str(self.dst)
        return f"{src}/{dst}{'*' if self.hit else ''}"


@dataclass(frozen=True)
class Move:
    """A complete legal play for one dice roll: an ordered list of sub-moves.

    A move bundles its resulting :class:`~.board.Board`
    (the *afterstate*) so callers can evaluate ``V(afterstate)`` without
    recomputing it. The ``afterstate`` already has the turn passed to the
    opponent.

    ``submoves`` may be empty, representing a forced *pass* when the player has
    no legal play for the roll.
    """

    submoves: tuple[SubMove, ...]
    afterstate: "Board"  # noqa: F821 - forward ref; avoids an import cycle.
    dice: tuple[int, ...] = field(default=())

    @property
    def is_pass(self) -> bool:
        """True when the move has no sub-moves (a forced pass)."""
        return len(self.submoves) == 0
