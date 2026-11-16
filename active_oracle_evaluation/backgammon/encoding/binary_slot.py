"""Binary-slot board encoding used as the input features of every network.

This is the concrete :class:`~.base.Encoder` the value and policy
networks consume. It turns a board into a fixed **247-dimensional** float vector:

    240  board occupancy  (24 points x 2 players x 5 slots)
      4  bar / off counts (mover bar, opp bar, mover off, opp off)
      3  scalar summaries (pip-count difference, mover blots, mover made points)
    ---
    247

Perspective
-----------
``encode`` always works on :meth:`Board.canonical` - the position seen from the
mover's point of view, with the mover relabelled as White. The network therefore
never has to learn that "Black is just White mirrored": two positions that are the
same up to whose turn it is encode identically. Because ``canonical()`` is
idempotent, passing an already-canonical board is harmless.

The 5-slot occupancy scheme
---------------------------
For a point holding ``n`` checkers of one player, the five slots are

    [ n>=1, n>=2, n>=3, n>=4, max(0, n-4)/2 ]

The first four are the classic TD-Gammon "at least k" indicators; the fifth
encodes any checkers beyond four on the point as a small continuous magnitude. A
point is occupied by at most one player, so the other player's five slots are all
zero.

All feature offsets are fixed and documented so the layout is stable across
checkpoints (a saved network is only meaningful paired with this exact order).
"""

from __future__ import annotations

import functools

import numpy as np

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.constants import CHECKERS_PER_PLAYER, NUM_POINTS
from active_oracle_evaluation.backgammon.core.types import Player
from active_oracle_evaluation.backgammon.encoding.base import Encoder

#: Occupancy slots emitted per (point, player): four TD-Gammon "at least k"
#: indicators (Tesauro, 1995) plus one overflow slot chosen for this study.
SLOTS_PER_POINT_PLAYER = 5
#: 24 points x 2 players x 5 slots.
BOARD_FEATURES = NUM_POINTS * 2 * SLOTS_PER_POINT_PLAYER
#: bar(mover), bar(opp), off(mover), off(opp).
BAROFF_FEATURES = 4
#: pip-count difference, mover blots, mover made points.
SCALAR_FEATURES = 3
#: Total length of the feature vector.
ENCODING_SIZE = BOARD_FEATURES + BAROFF_FEATURES + SCALAR_FEATURES  # 247

#: Normalising constants, chosen for this study and fixed by the saved checkpoints.
#: ``167`` is the standard opening pip count, so a pip difference is expressed
#: roughly in [-1, 1]. The bar uses the TD-Gammon ``/2`` scaling (Tesauro, 1995);
#: everything else is a fraction of the 15 checkers / 24 points.
_PIP_NORM = 167.0
_BAR_NORM = 2.0
_OFF_NORM = float(CHECKERS_PER_PLAYER)
_BLOT_NORM = float(CHECKERS_PER_PLAYER)
_MADE_NORM = float(NUM_POINTS)

#: Precomputed five occupancy slots for a point holding ``n`` checkers, for every
#: ``n`` in ``0..CHECKERS_PER_PLAYER`` (a point can hold at most a player's 15
#: checkers). ``encode`` indexes this table by the point counts instead of
#: recomputing the slots per point.
#: Slots: ``[n>=1, n>=2, n>=3, n>=4, max(0, n-4)/2]`` (all exact in float32).
_SLOT_LUT = np.array(
    [[n >= 1, n >= 2, n >= 3, n >= 4, max(0, n - 4) / 2.0] for n in range(CHECKERS_PER_PLAYER + 1)],
    dtype=np.float32,
)


#: How many distinct boards the encoder memoizes (chosen for this study; affects
#: speed only). ``encode`` is a pure function of the board, so the cache never
#: needs invalidating; repeated encodes during play are temporally local, so a
#: small LRU captures most of them.
_ENCODE_CACHE_SIZE = 8192


@functools.lru_cache(maxsize=_ENCODE_CACHE_SIZE)
def _encode_features(board: Board) -> np.ndarray:
    """Compute (and memoize) the 247-float binary-slot vector for ``board``.

    Vectorized over the 24 points with numpy. The slot values are 0/1 indicators
    and exact halves, so they are exact in float32.

    The result is shared across callers and marked **read-only** so a cached entry
    can never be corrupted by an in-place write. Consumers here only ever read it
    (e.g. ``np.stack`` copies before handing it to torch).
    """
    canon = board.canonical()  # mover relabelled as White; idempotent.
    feats = np.zeros(ENCODING_SIZE, dtype=np.float32)

    # --- 240 board occupancy features -------------------------------- #
    # The mover is White (positive counts), the opponent Black (negative).
    # Per point we emit the mover's five slots then the opponent's five.
    pts = np.asarray(canon.points, dtype=np.int64)  # signed counts, index 0 == point 1
    mover = np.maximum(pts, 0)  # White (positive); a point holds <= 15 checkers
    opp = np.maximum(-pts, 0)  # Black (negative -> positive)
    block = np.empty((NUM_POINTS, 2 * SLOTS_PER_POINT_PLAYER), dtype=np.float32)
    block[:, :SLOTS_PER_POINT_PLAYER] = _SLOT_LUT[mover]  # mover's five slots
    block[:, SLOTS_PER_POINT_PLAYER:] = _SLOT_LUT[opp]  # opponent's five slots
    feats[:BOARD_FEATURES] = block.reshape(-1)

    blots = int(np.count_nonzero(mover == 1))
    made = int(np.count_nonzero(mover >= 2))

    # --- 4 bar / off features ---------------------------------------- #
    i = BOARD_FEATURES
    feats[i + 0] = canon.bar(Player.WHITE) / _BAR_NORM
    feats[i + 1] = canon.bar(Player.BLACK) / _BAR_NORM
    feats[i + 2] = canon.off(Player.WHITE) / _OFF_NORM
    feats[i + 3] = canon.off(Player.BLACK) / _OFF_NORM
    i += BAROFF_FEATURES

    # --- 3 scalar summary features ----------------------------------- #
    # Positive pip difference means the mover is ahead in the race.
    pip_diff = canon.pip_count(Player.BLACK) - canon.pip_count(Player.WHITE)
    feats[i + 0] = pip_diff / _PIP_NORM
    feats[i + 1] = blots / _BLOT_NORM
    feats[i + 2] = made / _MADE_NORM

    feats.flags.writeable = False  # shared cached entry; never mutate in place
    return feats


class BinarySlotEncoder(Encoder):
    """Maps a board to the 247-feature binary-slot vector described above."""

    @property
    def size(self) -> int:
        """Length of the feature vector (247)."""
        return ENCODING_SIZE

    def encode(self, board: Board) -> np.ndarray:
        """Return the ``float32`` feature vector for ``board`` (mover's view).

        Delegates to the memoized module-level :func:`_encode_features`. The
        returned array is shared and **read-only**; copy it if you need to mutate.
        """
        return _encode_features(board)
