"""Legal-move generation -> afterstates.

All generation happens in **mover-relative** coordinates (see :mod:`.constants`):
the player on roll always moves from high points toward point 1, bears off past
point 1, and re-enters from the bar on the high end. Writing the rules once from this single perspective keeps one code
path for both colors.

The pipeline for a roll is:

1. Expand the roll into individual dice (doubles -> four dice).
2. Recursively try every legal ordering of dice, producing *complete* sub-move
   sequences (a sequence ends only when no further die can be played).
3. Apply the rules that constrain which sequences are actually legal:
   * **Maximal usage** - you must play as many dice as possible, so keep only the
     longest sequences.
   * **Higher-die rule** - if exactly one die can be played and either die would
     work alone, you must play the larger one.
4. Build the resulting :class:`~.board.Board` (the *afterstate*,
   turn passed to the opponent) for each sequence and de-duplicate by board.

Each returned :class:`Move` carries its afterstate so callers never recompute it.
"""

from __future__ import annotations

from dataclasses import dataclass

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.constants import BAR, HOME_SIZE, NUM_POINTS, OFF
from active_oracle_evaluation.backgammon.core.types import Move, SubMove


@dataclass
class _NormState:
    """Mutable working state during generation, in mover-relative coordinates.

    ``norm`` is indexed by point 1..24 (index 0 unused); positive = mover's
    checkers, negative = opponent's. Bars and borne-off counts are split into
    mover/opponent. This is intentionally a plain mutable struct for speed; it
    never escapes this module - results are converted back to immutable boards.
    """

    norm: list[int]
    mover_bar: int
    opp_bar: int
    mover_off: int
    opp_off: int

    def copy(self) -> "_NormState":
        """Return an independent copy of this state."""
        return _NormState(self.norm[:], self.mover_bar, self.opp_bar, self.mover_off, self.opp_off)


def _can_bear_off(state: _NormState) -> bool:
    """Bearing off is legal only with every checker home and none on the bar."""
    if state.mover_bar > 0:
        return False
    for p in range(HOME_SIZE + 1, NUM_POINTS + 1):
        if state.norm[p] > 0:
            return False
    return True


def _highest_home_point(state: _NormState) -> int:
    """Highest occupied home point (1..6), or 0 if the mover has none home.

    Used to apply the oversized bear-off rule: a die larger than the position of
    a checker may only bear that checker off if it is the highest one remaining.
    """
    for p in range(HOME_SIZE, 0, -1):
        if state.norm[p] > 0:
            return p
    return 0


def _single_die_plays(state: _NormState, die: int) -> list[tuple[SubMove, _NormState]]:
    """All legal plays of a single ``die`` from ``state``.

    Returns ``(submove, resulting_state)`` pairs. Returns an empty list when the
    die cannot be played at all (which is how the recursion detects a dead end).
    """
    plays: list[tuple[SubMove, _NormState]] = []

    # Rule: checkers on the bar must re-enter before anything else can move.
    if state.mover_bar > 0:
        entry = BAR - die  # BAR == 25, so die 1 -> 24 ... die 6 -> 19
        if state.norm[entry] >= -1:  # open point or a lone opponent blot
            hit = state.norm[entry] == -1
            new = state.copy()
            new.mover_bar -= 1
            if hit:
                new.norm[entry] = 0
                new.opp_bar += 1
            new.norm[entry] += 1
            plays.append((SubMove(BAR, entry, die, hit), new))
        return plays  # while on the bar, no other move type is possible

    can_bear = _can_bear_off(state)
    highest = _highest_home_point(state) if can_bear else 0

    for p in range(NUM_POINTS, 0, -1):
        if state.norm[p] <= 0:
            continue  # no mover checker on this point
        dest = p - die

        if dest >= 1:
            # Ordinary move within the board. Legal unless the destination is
            # blocked by two or more opponent checkers (norm[dest] <= -2).
            if state.norm[dest] >= -1:
                hit = state.norm[dest] == -1
                new = state.copy()
                new.norm[p] -= 1
                if hit:
                    new.norm[dest] = 0
                    new.opp_bar += 1
                new.norm[dest] += 1
                plays.append((SubMove(p, dest, die, hit), new))
            continue

        # dest <= 0: this die would carry the checker off the board.
        if not can_bear:
            continue
        if dest == 0:
            # Exact bear-off: a checker on point ``die`` borne off with ``die``.
            new = state.copy()
            new.norm[p] -= 1
            new.mover_off += 1
            plays.append((SubMove(p, OFF, die, False), new))
        elif p == highest:
            # Oversized bear-off: die exceeds the point, allowed only from the
            # highest occupied point (no checker waits on a higher point).
            new = state.copy()
            new.norm[p] -= 1
            new.mover_off += 1
            plays.append((SubMove(p, OFF, die, False), new))
        # else: oversized die but a higher checker exists -> not playable here.

    return plays


def _expand(
    state: _NormState, dice_remaining: tuple[int, ...]
) -> list[tuple[tuple[SubMove, ...], _NormState]]:
    """Recursively enumerate complete sub-move sequences.

    A sequence is "complete" when no remaining die can be played (either dice are
    exhausted or every one is blocked). This guarantees every returned sequence
    is maximal *for its ordering*; the global maximal-usage filter is applied by
    the caller. Iterating over ``set(dice_remaining)`` collapses the identical
    dice of a double so we don't expand the same branch four times.
    """
    results: list[tuple[tuple[SubMove, ...], _NormState]] = []
    progressed = False
    for die in set(dice_remaining):
        plays = _single_die_plays(state, die)
        if not plays:
            continue
        progressed = True
        rest = list(dice_remaining)
        rest.remove(die)
        rest_t = tuple(rest)
        for submove, newstate in plays:
            for subseq, finalstate in _expand(newstate, rest_t):
                results.append(((submove, *subseq), finalstate))
    if not progressed:
        results.append(((), state))  # dead end: this sequence is complete
    return results


def legal_moves(board: Board, dice: tuple[int, int]) -> list[Move]:
    """Return every legal :class:`Move` for ``board`` given a two-die roll.

    The result always contains at least one move; when the player is completely
    stuck it is a single *pass* (empty sub-moves) that simply hands the turn over.
    Each move's :attr:`~.types.Move.afterstate` is the resulting
    board with the turn passed to the opponent.
    """
    d1, d2 = dice
    is_double = d1 == d2
    dice_pool = (d1,) * 4 if is_double else (d1, d2)

    norm, mover_bar, opp_bar, mover_off, opp_off = board.to_mover_relative()
    start = _NormState(norm, mover_bar, opp_bar, mover_off, opp_off)

    sequences = _expand(start, dice_pool)
    max_len = max(len(seq) for seq, _ in sequences)

    # --- Maximal-usage rule: you must play as many dice as possible. ---------
    kept = [(seq, st) for seq, st in sequences if len(seq) == max_len]

    # --- Higher-die rule: if only one die is playable and either could be -----
    # played alone, the larger die must be the one used.
    if not is_double and max_len == 1:
        used_dice = {seq[0].die for seq, _ in kept}
        if d1 in used_dice and d2 in used_dice:
            larger = max(d1, d2)
            kept = [(seq, st) for seq, st in kept if seq[0].die == larger]

    mover = board.turn
    new_turn = mover.opponent

    # --- Build afterstate boards and de-duplicate by resulting position. ------
    moves: dict[Board, Move] = {}
    for seq, st in kept:
        afterstate = Board.from_mover_relative(
            st.norm,
            st.mover_bar,
            st.opp_bar,
            st.mover_off,
            st.opp_off,
            mover=mover,
            new_turn=new_turn,
        )
        if afterstate not in moves:
            moves[afterstate] = Move(submoves=seq, afterstate=afterstate, dice=dice)
    return list(moves.values())
