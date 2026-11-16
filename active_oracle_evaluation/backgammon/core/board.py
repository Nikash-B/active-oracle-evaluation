"""The board: an immutable snapshot of all checker positions plus whose turn.

Storage model
-------------
A board stores checker positions in **absolute** coordinates: a fixed 1..24
numbering shared by both players, with a *signed* count per point
(positive = White, negative = Black). Bar and borne-off counts are tracked per
player. There is therefore exactly one physical representation of any position.

Move generation, however, is far simpler when written from the mover's point of
view (mover always moves high -> low, bears off past point 1, enters on the high
end). The board provides :meth:`to_mover_relative` / :meth:`from_mover_relative`
to translate between the two views, and :meth:`canonical` to expose the
mover-as-White view that encoders and value functions consume.

Immutability
------------
Boards are treated as immutable values: every "change" produces a new board.
This makes them hashable, so afterstates can be de-duplicated and cached.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.core.constants import (
    CHECKERS_PER_PLAYER,
    HOME_SIZE,
    NUM_POINTS,
    STARTING_POSITION,
)
from active_oracle_evaluation.backgammon.core.types import Player


class Board:
    """Immutable backgammon position.

    Parameters mirror the storage model above. Prefer the constructors
    :meth:`start` and :meth:`from_mover_relative` over calling ``__init__``
    directly.
    """

    __slots__ = ("_points", "_white_bar", "_black_bar", "_white_off", "_black_off", "turn", "_key")

    def __init__(
        self,
        points: tuple[int, ...],
        white_bar: int,
        black_bar: int,
        white_off: int,
        black_off: int,
        turn: Player,
    ) -> None:
        """Store the absolute point array, bar and borne-off counts, and the side to move."""
        if len(points) != NUM_POINTS:
            raise ValueError(f"points must have length {NUM_POINTS}, got {len(points)}")
        self._points = tuple(points)
        self._white_bar = white_bar
        self._black_bar = black_bar
        self._white_off = white_off
        self._black_off = black_off
        self.turn = turn
        # Precompute a hashable identity key once (boards are immutable).
        self._key = (self._points, white_bar, black_bar, white_off, black_off, turn)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def start(cls) -> "Board":
        """The standard opening position, White to move."""
        points = [0] * NUM_POINTS
        for point, count in STARTING_POSITION.items():
            points[point - 1] = count
        return cls(tuple(points), 0, 0, 0, 0, Player.WHITE)

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    def point(self, p: int) -> int:
        """Signed checker count at absolute point ``p`` (1..24)."""
        return self._points[p - 1]

    @property
    def points(self) -> tuple[int, ...]:
        """The full absolute point array (index 0 == point 1)."""
        return self._points

    def bar(self, player: Player) -> int:
        """Number of ``player``'s checkers on the bar."""
        return self._white_bar if player is Player.WHITE else self._black_bar

    def off(self, player: Player) -> int:
        """Number of ``player``'s checkers borne off."""
        return self._white_off if player is Player.WHITE else self._black_off

    # ------------------------------------------------------------------ #
    # Derived game facts
    # ------------------------------------------------------------------ #
    def pip_count(self, player: Player) -> int:
        """Total pips ``player`` must still travel to bear everyone off.

        White bears off past point 1, so a White checker on point ``p`` owes
        ``p`` pips; Black bears off past point 24, owing ``25 - p`` pips. A
        checker on the bar owes a full 25.
        """
        total = self.bar(player) * 25
        for idx, count in enumerate(self._points):
            point = idx + 1
            if player is Player.WHITE and count > 0:
                total += point * count
            elif player is Player.BLACK and count < 0:
                total += (25 - point) * (-count)
        return total

    def home_range(self, player: Player) -> range:
        """Absolute points forming ``player``'s home board.

        White's home is points 1..6; Black's is the mirror, 19..24.
        """
        if player is Player.WHITE:
            return range(1, HOME_SIZE + 1)
        return range(NUM_POINTS - HOME_SIZE + 1, NUM_POINTS + 1)

    def all_in_home(self, player: Player) -> bool:
        """True iff every one of ``player``'s checkers is home (bear-off ready)."""
        if self.bar(player) > 0:
            return False
        home = set(self.home_range(player))
        for idx, count in enumerate(self._points):
            point = idx + 1
            if point in home:
                continue
            if player is Player.WHITE and count > 0:
                return False
            if player is Player.BLACK and count < 0:
                return False
        return True

    def has_won(self, player: Player) -> bool:
        """True iff ``player`` has borne off all checkers."""
        return self.off(player) == CHECKERS_PER_PLAYER

    # ------------------------------------------------------------------ #
    # Perspective transforms
    # ------------------------------------------------------------------ #
    def to_mover_relative(self) -> tuple[list[int], int, int, int, int]:
        """Return ``(norm, mover_bar, opp_bar, mover_off, opp_off)``.

        ``norm`` is a length-25 list indexed by mover-relative point 1..24 (index
        0 is unused padding so arithmetic like ``25 - die`` reads naturally).
        Positive entries are the mover's checkers, negative the opponent's, and
        the mover always travels from high points toward point 1.
        """
        mover = self.turn
        norm = [0] * (NUM_POINTS + 1)
        if mover is Player.WHITE:
            for p in range(1, NUM_POINTS + 1):
                norm[p] = self.point(p)
            return norm, self._white_bar, self._black_bar, self._white_off, self._black_off
        # Black: mirror the board (point p <-> 25 - p) and flip signs so the
        # mover's checkers come out positive and also travel high -> low.
        for p in range(1, NUM_POINTS + 1):
            norm[p] = -self.point(NUM_POINTS + 1 - p)
        return norm, self._black_bar, self._white_bar, self._black_off, self._white_off

    @classmethod
    def from_mover_relative(
        cls,
        norm: list[int],
        mover_bar: int,
        opp_bar: int,
        mover_off: int,
        opp_off: int,
        mover: Player,
        new_turn: Player,
    ) -> "Board":
        """Inverse of :meth:`to_mover_relative`, rebuilding an absolute board.

        ``mover`` says whose perspective ``norm`` is in; ``new_turn`` sets whose
        turn the resulting board records (an afterstate uses ``mover.opponent``).
        """
        points = [0] * NUM_POINTS
        if mover is Player.WHITE:
            for p in range(1, NUM_POINTS + 1):
                points[p - 1] = norm[p]
            white_bar, black_bar = mover_bar, opp_bar
            white_off, black_off = mover_off, opp_off
        else:
            for p in range(1, NUM_POINTS + 1):
                points[NUM_POINTS - p] = -norm[p]
            white_bar, black_bar = opp_bar, mover_bar
            white_off, black_off = opp_off, mover_off
        return cls(tuple(points), white_bar, black_bar, white_off, black_off, new_turn)

    def canonical(self) -> "Board":
        """The position as the mover sees it, with the mover relabeled as White.

        Encoders and value functions consume the canonical view so they never
        have to special-case which color is on roll. For White to move this
        is a no-op; for Black it mirrors and flips the board.
        """
        norm, mover_bar, opp_bar, mover_off, opp_off = self.to_mover_relative()
        return Board.from_mover_relative(
            norm,
            mover_bar,
            opp_bar,
            mover_off,
            opp_off,
            mover=Player.WHITE,
            new_turn=Player.WHITE,
        )

    # ------------------------------------------------------------------ #
    # Equality and hashing
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        """Boards are equal when all positions, counts and the side to move match."""
        return isinstance(other, Board) and self._key == other._key

    def __hash__(self) -> int:
        """Hash of the precomputed identity key."""
        return hash(self._key)
