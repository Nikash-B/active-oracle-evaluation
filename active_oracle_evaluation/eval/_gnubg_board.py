"""Convert a repo ``Board`` to the gnubg-nn board frame.

Pure index remapping; the oracle passes the result to the native ``gnubg_nn`` calls.

gnubg frame conventions:

* A *side* is a 25-int list in that player's own numbering: index ``i`` is their point
  ``i + 1`` and index 24 is the bar. Borne-off checkers are omitted.
* A gnubg *board* is ``[opp, mover]``: board0 is the opponent and board1 is the player on
  roll. gnubg's move and evaluation calls act on the second list.
* The opponent's own point ``q`` is absolute point ``25 - q``.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.types import Player

_NUM_POINTS = 24  # backgammon rules


def to_gnubg_sides(board: Board) -> tuple[list[int], list[int]]:
    """Return ``(mover, opp)`` as gnubg 25-int sides for the player to act.

    Canonicalizes first so the mover is White (gnubg's mover frame equals the absolute one),
    then reads each point: mover checkers at their own index, opponent checkers mirrored to
    their own numbering (absolute ``p`` becomes opponent index ``24 - p``).
    """
    canon = board.canonical()
    mover = [0] * (_NUM_POINTS + 1)
    opp = [0] * (_NUM_POINTS + 1)
    for point in range(1, _NUM_POINTS + 1):
        count = canon.point(point)
        if count > 0:
            mover[point - 1] = count
        elif count < 0:
            opp[_NUM_POINTS - point] = -count
    mover[_NUM_POINTS] = canon.bar(Player.WHITE)
    opp[_NUM_POINTS] = canon.bar(Player.BLACK)
    return mover, opp


def to_gnubg_board(board: Board) -> list[list[int]]:
    """Return the gnubg ``[opp, mover]`` board used as the argument of gnubg-nn calls."""
    mover, opp = to_gnubg_sides(board)
    return [opp, mover]
