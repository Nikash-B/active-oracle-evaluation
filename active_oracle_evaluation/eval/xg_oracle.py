"""eXtreme Gammon (XG) position and evaluation codecs.

XG is a Windows GUI program with no scripting interface, so it is used in batch mode: the suite
decisions are exported as a match file, XG analyzes them in one session and saves a ``.xg`` file,
and ``python -m active_oracle_evaluation.experiments.xg_oracle`` parses that file with the
codecs below.

Formats follow XG's file-format specification (see References in README.md):

* Position: a 26-integer array in the frame of the player on roll. Index 0 is the opponent bar
  (negative), 1..24 are the points (1 = ace point, ``+`` player on roll, ``-`` opponent) and 25 is
  the bar of the player on roll (positive).
* Eval: a 7-float array ``[loseBG, loseG, loseS, winS, winG, winBG, equity]``.

XG stores each candidate's ``Eval`` in the frame of the player who moves, and each candidate's
``PosPlayed`` in that same frame, which is the mirror of the afterstate's own frame
(:func:`mirror_xg_pos`).
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.types import Player

_NUM_POINTS = 24  # backgammon rules
_CHECKERS_PER_SIDE = 15  # backgammon rules


def board_to_xg_pos(board: Board) -> tuple[int, ...]:
    """Encode a ``Board`` as XG's 26-integer position, in the frame of the player to act."""
    canon = board.canonical()  # the player to act becomes White
    xg_pos = [0] * 26
    for point in range(1, _NUM_POINTS + 1):
        xg_pos[point] = canon.point(point)
    xg_pos[25] = canon.bar(Player.WHITE)
    xg_pos[0] = -canon.bar(Player.BLACK)
    return tuple(xg_pos)


def mirror_xg_pos(pos) -> tuple[int, ...]:
    """Mirror a 26-integer position into the other player's frame (reverse and negate)."""
    return tuple(-int(pos[25 - i]) for i in range(26))


def xg_pos_to_board(pos) -> Board:
    """Decode an XG 26-integer position into a White-to-move ``Board`` (inverse of the encoder)."""
    points = tuple(int(pos[p]) for p in range(1, _NUM_POINTS + 1))
    white_bar = int(pos[25])
    black_bar = -int(pos[0])
    white_on = sum(v for v in points if v > 0) + white_bar
    black_on = -sum(v for v in points if v < 0) + black_bar
    return Board(
        points,
        white_bar,
        black_bar,
        _CHECKERS_PER_SIDE - white_on,
        _CHECKERS_PER_SIDE - black_on,
        Player.WHITE,
    )


def cubeless_equity_from_xg_eval(ev) -> float:
    """Return cubeless equity to the mover from XG's 7-float eval, using the six outcome terms."""
    lose_bg, lose_g, lose_s, win_s, win_g, win_bg = (float(x) for x in ev[:6])
    return win_s + 2.0 * win_g + 3.0 * win_bg - lose_s - 2.0 * lose_g - 3.0 * lose_bg
