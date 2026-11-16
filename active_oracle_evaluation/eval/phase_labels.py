"""Deterministic game-phase labels for a backgammon board.

Each position gets exactly one coarse *phase*, chosen by a fixed priority of
pure-Python heuristics over the :class:`Board` representation. The labels slice
the decision suite by stage of the game; they are not used for play. The same
board always yields the same label, with no oracle call and no dependency on
``gnubg-nn``.

The six labels::

    opening   bearoff   race   holding   blitz_backgame   contact

Coordinate convention
----------------------
All geometric reasoning is done in **mover-relative** coordinates (the side
being labeled is "the mover"; see :meth:`Board.to_mover_relative`). The mover's
checkers are positive and travel from high points toward point 1; the opponent's
are negative and travel the other way. In this frame:

* the **mover's home board** is points 1..6;
* the **opponent's home board** is points 19..24 (the mover's back region).

Priority order
--------------
The first rule that fires wins. From highest to lowest priority:

1. ``opening`` - very early game, decided purely by ply count.
2. ``bearoff`` - the labeled side has every checker home (bearing off).
3. ``race`` - the two armies have passed each other; no hit/block is possible.
4. ``holding`` - contact and the labeled side holds an advanced anchor in the
   opponent's outfield/home (mover-relative points 18..22).
5. ``blitz_backgame`` - contact and a deep/heavy back structure (deep anchors,
   stacked back checkers, or a strong opposing home board).
6. ``contact`` - the fallback whenever contact exists but nothing above fires.

The numeric thresholds for rules 4 and 5 are the module constants below.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.constants import HOME_SIZE, NUM_POINTS
from active_oracle_evaluation.backgammon.core.types import Player

# Heuristic constants used by the rules below. The paper (appendix "Phase diagnostics and search
# ablation", paragraph "Phases") states the opening length and the holding points 18..22; the
# other thresholds are chosen for this study.

#: A "made point" / "anchor" needs at least this many checkers on it (backgammon rules).
ANCHOR_MIN = 2

#: Plies counted as the opening (paper, appendix "Phase diagnostics and search ablation",
#: paragraph "Phases": the first two plies).
OPENING_PLIES = 2

#: Mover-relative "point" of a checker on the bar: just above the last point.
BAR_POINT = NUM_POINTS + 1

#: Mover-relative points that count as an *advanced holding anchor* (rule 4 of the paper's
#: appendix "Phase diagnostics and search ablation", paragraph "Phases").
#: Covers the golden 20-point, the 21/22 high anchors, and the 18-point ("bar")
#: anchor. Excludes the deep 23/24 anchors (handled by rule 5), so the opening
#: 24-point anchor is not read as holding.
HOLDING_ANCHOR_POINTS = (18, 19, 20, 21, 22)

#: Mover-relative points that count as *deep* back points (rule 5, chosen for this study): the
#: opponent's 1/2 points, i.e. the ace-point and 2-point anchors.
DEEP_POINTS = (23, 24)

#: A two-anchor backgame: at least this many of DEEP_POINTS are anchored (chosen for this study).
BLITZ_DEEP_ANCHORS_MIN = 2

#: A heavy back position: at least this many mover checkers sit on DEEP_POINTS
#: plus the 22-point (mover-relative 22..24), regardless of how they're split
#: (chosen for this study).
BLITZ_DEEP_CHECKERS_MIN = 3

#: A dangerous opposing home board: the opponent has made at least this many of
#: their six home points (mover-relative 19..24) (chosen for this study).
BLITZ_OPP_HOME_POINTS_MIN = 4


def _mover_relative(board: Board, player: Player) -> tuple[list[int], int, int]:
    """Return ``(norm, mover_bar, opp_bar)`` for ``player``.

    ``norm`` is indexed by mover-relative point 1..24 (index 0 unused). Positive
    entries are ``player``'s checkers, negative the opponent's, with ``player``
    travelling high -> low. This mirrors :meth:`Board.to_mover_relative` for an
    arbitrary side, not only ``board.turn``.
    """
    norm = [0] * (NUM_POINTS + 1)
    if player is Player.WHITE:
        for p in range(1, NUM_POINTS + 1):
            norm[p] = board.point(p)
        return (
            norm,
            board.bar(Player.WHITE),
            board.bar(Player.BLACK),
        )
    # Black: mirror (p <-> 25 - p) and flip signs so the mover comes out positive.
    for p in range(1, NUM_POINTS + 1):
        norm[p] = -board.point(NUM_POINTS + 1 - p)
    return (
        norm,
        board.bar(Player.BLACK),
        board.bar(Player.WHITE),
    )


def _has_contact(norm: list[int], mover_bar: int, opp_bar: int) -> bool:
    """True iff a hit or block is still possible (the armies have not passed).

    The mover travels high -> low and the opponent low -> high (in this frame),
    so once every mover checker sits strictly below every opponent checker the
    two can only move apart - a pure race. A checker on the bar always implies
    contact: the mover re-enters at the high end (point 24-ish) and the opponent
    re-enters at the low end, both back into the danger zone.
    """
    mover_points = [p for p in range(1, NUM_POINTS + 1) if norm[p] > 0]
    opp_points = [p for p in range(1, NUM_POINTS + 1) if norm[p] < 0]
    # mover_rear: the mover's farthest-from-home checker (highest point / bar).
    mover_rear = BAR_POINT if mover_bar > 0 else (max(mover_points) if mover_points else 0)
    # opp_front: the opponent's farthest-from-their-home checker (lowest point /
    # bar). On the bar the opponent re-enters at the low end -> treat as 0.
    opp_front = 0 if opp_bar > 0 else (min(opp_points) if opp_points else NUM_POINTS + 1)
    return mover_rear > opp_front


def _is_holding(norm: list[int]) -> bool:
    """The mover anchors an advanced point in the opponent's outfield/home."""
    return any(norm[p] >= ANCHOR_MIN for p in HOLDING_ANCHOR_POINTS)


def _is_blitz_backgame(norm: list[int]) -> bool:
    """True iff the mover has two deep anchors, stacked back checkers, or faces a strong board."""
    deep_anchors = sum(1 for p in DEEP_POINTS if norm[p] >= ANCHOR_MIN)
    if deep_anchors >= BLITZ_DEEP_ANCHORS_MIN:
        return True
    deep_checkers = sum(norm[p] for p in (22, 23, 24) if norm[p] > 0)
    if deep_checkers >= BLITZ_DEEP_CHECKERS_MIN:
        return True
    opp_home = range(NUM_POINTS - HOME_SIZE + 1, NUM_POINTS + 1)  # 19..24
    opp_made = sum(1 for p in opp_home if norm[p] <= -ANCHOR_MIN)
    return opp_made >= BLITZ_OPP_HOME_POINTS_MIN


def label_phase(board: Board, *, ply_index: int | None = None) -> str:
    """Return the game phase of ``board`` from the point of view of ``board.turn``.

    Parameters
    ----------
    board:
        The position to label.
    ply_index:
        Optional 0-based ply count of the game so far. Only used to detect the
        ``opening`` (the first :data:`OPENING_PLIES` plies); pass ``None`` to disable the
        opening rule.

    Returns
    -------
    str
        One of the six labels. The function is total: it always returns a label.
    """
    side = board.turn

    # 1. opening - earliest plies, decided purely by the move counter.
    if ply_index is not None and ply_index < OPENING_PLIES:
        return "opening"

    # 2. bearoff - every one of this side's checkers is home (bearing off).
    if board.all_in_home(side):
        return "bearoff"

    norm, mover_bar, opp_bar = _mover_relative(board, side)

    # 3. race - the armies have passed each other; no contact remains.
    if not _has_contact(norm, mover_bar, opp_bar):
        return "race"

    # 4. holding - contact plus an advanced anchor in the opponent's territory.
    if _is_holding(norm):
        return "holding"

    # 5. blitz_backgame - contact plus a deep/heavy back structure.
    if _is_blitz_backgame(norm):
        return "blitz_backgame"

    # 6. contact - the catch-all for any other position with contact.
    return "contact"
