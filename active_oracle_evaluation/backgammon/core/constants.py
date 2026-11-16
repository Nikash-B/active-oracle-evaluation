"""Board geometry and fixed game constants.

This module defines the shape of a backgammon board. The rest of the engine
imports these constants instead of hard-coding numbers.

Coordinate conventions
-----------------------
We use **mover-relative point numbers** for moves (the conventional backgammon
notation): each player numbers the points 1..24 from their *own* perspective,
where 24 is the farthest point and 1..6 is the home board they bear off from.
A move therefore always reads "from a higher point to a lower point" for the
player on roll - e.g. ``24/18`` or ``6/off`` - which is exactly how players
write moves.

Two sentinel "points" extend that line:

* :data:`BAR` (= 25): the bar, where hit checkers wait to re-enter. A checker
  re-enters from the bar onto point ``25 - die``.
* :data:`OFF` (= 0): the tray of borne-off checkers, reached by moving a home
  checker "off the low end" of the board.

The :class:`~.board.Board`, by contrast, stores positions in
**absolute** coordinates (a fixed 1..24 numbering shared by both players) so
there is one unambiguous physical board. The translation between the two is
``Board.to_mover_relative`` / ``Board.from_mover_relative``.
"""

from __future__ import annotations

#: Number of playable points on the board.
NUM_POINTS: int = 24

#: Checkers each player starts with (and must bear off to win).
CHECKERS_PER_PLAYER: int = 15

#: Size of the home board (points 1..6 in mover-relative coordinates).
HOME_SIZE: int = 6

#: Sentinel "point" representing the bar (mover-relative). Chosen as 25 so that
#: ``25 - die`` gives the re-entry point, matching real board geometry.
BAR: int = 25

#: Sentinel "point" representing the borne-off tray (mover-relative). Chosen as
#: 0 so that bearing off is simply "move to a point <= 0".
OFF: int = 0

#: Standard opening position, expressed in *absolute* coordinates as a mapping
#: ``{point: signed_count}`` where a positive count is White and negative is
#: Black. Black's layout is the mirror of White's (``black_point = 25 - white``).
#: Total pip count for each side is the classic 167.
STARTING_POSITION: dict[int, int] = {
    24: +2,  # White back checkers
    13: +5,  # White midpoint
    8: +3,
    6: +5,  # White home board
    1: -2,  # Black back checkers (mirror of White's 24-point)
    12: -5,  # Black midpoint
    17: -3,
    19: -5,  # Black home board
}
