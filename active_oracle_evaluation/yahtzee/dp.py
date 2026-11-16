"""Exact backward DP for solitaire Yahtzee: optimal values and the shared option builder.

The solver and the policy evaluator both build their options with :func:`option_arrays`, so
they apply the same rules.

The state
---------
``(mask, yz, upper)``:

* ``mask`` — 13 bits, which categories are filled;
* ``upper`` — the upper-section running total **capped at 63**. Capping is exact, not an
  approximation: at 63 the 35-point bonus is already earned and further upper points change nothing
  about the future, so all totals ≥ 63 are the same state;
* ``yz`` — whether the Yahtzee box was filled *with 50* rather than crossed off at 0. It only
  matters once bit 12 is set, and it is what makes a later Yahtzee worth +100 and trigger the Joker
  rules. Without it the state would be Markov-incomplete and the answer wrong.

Flattened to ``(mask * 2 + yz) * 64 + upper`` so a whole DP pass is numpy indexing.

Ordering
--------
Masks are processed by descending popcount: filling a category only ever *adds* a bit, so every
successor of a state has already been computed when that state is reached. No iteration to a fixed
point is needed — the game is a finite DAG with exactly 13 levels.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.yahtzee import FULL_MASK, N_CAT, UPPER_BONUS, UPPER_BONUS_THRESHOLD
from active_oracle_evaluation.yahtzee.dice import (
    CHILD_INDEX,
    LAYER_OFFSET,
    LAYER_SIZE,
    N_DICE,
    N_ROLL,
    P_ROLL,
    ROLL_LO,
    expect,
)
from active_oracle_evaluation.yahtzee.scoring import SCORE, YAHTZEE_ROLL, joker_options

#: Upper totals 0..63; 63 absorbs everything above it
#: (paper, appendix "Solitaire Yahtzee: exact ground truth", paragraph "The solved game").
N_UPPER = UPPER_BONUS_THRESHOLD + 1
UPPER = np.arange(N_UPPER)
#: Flat state count: 8192 masks x 2 yahtzee-bonus flags x 64 upper totals.
N_STATE = (1 << N_CAT) * 2 * N_UPPER

#: Stands in for "illegal option" inside a max. Large enough that no real value approaches it;
#: chosen for this study.
NEG = -1e18


def state_index(mask: int, yz: int, upper: int) -> int:
    """Flat index of one state, matching the layout the DP tables use."""
    return ((mask * 2 + yz) * N_UPPER) + min(upper, UPPER_BONUS_THRESHOLD)


#: The start of a game: nothing filled, no Yahtzee bonus live, zero upper total.
START = state_index(0, 0, 0)


def masks_by_descending_popcount() -> list[int]:
    """Every non-terminal mask, ordered so successors precede predecessors."""
    return sorted(range(FULL_MASK), key=lambda m: -bin(m).count("1"))


def build_options(mask: int):
    """Legal ``(category, raw score)`` choices for each of the 252 rolls in one state.

    Returns ``(CAT, RAW, VALID)``, each ``(252, K)`` where ``K`` is the number of free categories.
    ``RAW`` is the face score only — the upper bonus and the +100 Extra Yahtzee Bonus depend on
    ``upper`` and ``yz`` and are applied in :func:`option_arrays`.

    When the Yahtzee box already holds 50, the five Yahtzee rolls have their option rows *replaced*
    by the Joker choices rather than extended: under the Joker rules the player is forced into a
    particular box or set of boxes, so the ordinary options are not available on those rolls.
    """
    free = [c for c in range(N_CAT) if not (mask >> c) & 1]
    n_opt = max(len(free), 1)

    cat = np.full((N_ROLL, n_opt), -1, dtype=np.int64)
    raw = np.zeros((N_ROLL, n_opt), dtype=np.int64)
    valid = np.zeros((N_ROLL, n_opt), dtype=bool)
    for j, c in enumerate(free):
        cat[:, j] = c
        raw[:, j] = SCORE[:, c]
        valid[:, j] = True

    if (mask >> 12) & 1:  # Yahtzee box filled -> Joker rules are in force
        for face, roll in YAHTZEE_ROLL.items():
            cat[roll, :] = -1
            raw[roll, :] = 0
            valid[roll, :] = False
            for j, (c, s) in enumerate(joker_options(face, free)):
                cat[roll, j] = c
                raw[roll, j] = s
                valid[roll, j] = True
    return cat, raw, valid


def option_arrays(mask: int, yz: int):
    """Per-option immediate reward, successor state and validity, for all 64 upper totals.

    Returns ``(reward, successor, valid, category)``. The first three are ``(64, 252, K)``; the
    reward includes both bonuses and the successor is a flat state index. ``category`` is
    ``(252, K)`` and says which box each option fills (``-1`` where the option is invalid).
    Rule-based agents rank by the box itself rather than by value.
    """
    cat, raw, valid = build_options(mask)
    n_opt = cat.shape[1]
    yz_set = (mask >> 12) & 1

    is_upper = (cat >= 0) & (cat < 6)
    next_mask = np.where(cat >= 0, mask | (1 << np.maximum(cat, 0)), mask)

    # yz flips only when the Yahtzee box is being filled now, and only if it is filled with 50.
    yz_next = np.full((N_ROLL, n_opt), yz, dtype=np.int64)
    filling_yahtzee = cat == 12
    yz_next[filling_yahtzee] = (raw[filling_yahtzee] == 50).astype(np.int64)

    u = UPPER[:, None, None]
    raw_b = raw[None, :, :]
    next_upper = np.where(
        is_upper[None],
        np.minimum(UPPER_BONUS_THRESHOLD, u + raw_b),
        np.broadcast_to(u, (N_UPPER, N_ROLL, n_opt)),
    )
    # The bonus is paid exactly once, on the transition that first reaches 63.
    crosses = is_upper[None] & (u < UPPER_BONUS_THRESHOLD) & (next_upper == UPPER_BONUS_THRESHOLD)
    reward = raw_b + np.where(crosses, UPPER_BONUS, 0)

    if yz_set and yz == 1:  # every further Yahtzee is worth +100 (official rules)
        extra = np.zeros((N_ROLL, n_opt), dtype=np.int64)
        extra[list(YAHTZEE_ROLL.values()), :] = 100
        reward = reward + extra[None]

    successor = (next_mask[None] * 2 + yz_next[None]) * N_UPPER + next_upper
    return (
        reward.astype(np.float64),
        successor,
        np.broadcast_to(valid[None], reward.shape),
        cat,
    )


def submax_arg(values_on_keeps: np.ndarray):
    """Subset maxima over the keep lattice, and *which* sub-multiset attained each max.

    Policy evaluation needs the argmax: it replays the keep the agent chose, which is the argmax
    against the agent's own value table, not against the optimal one.
    """
    best = values_on_keeps.copy()
    arg = np.broadcast_to(np.arange(values_on_keeps.shape[1]), values_on_keeps.shape).copy()
    for k in range(1, N_DICE + 1):
        lo, n = LAYER_OFFSET[k], LAYER_SIZE[k]
        kids = CHILD_INDEX[k - 1]
        child_vals = best[:, kids]
        j = child_vals.argmax(-1)[..., None]
        child_best = np.take_along_axis(child_vals, j, -1)[..., 0]
        child_arg = np.take_along_axis(arg[:, kids], j, -1)[..., 0]
        improves = child_best > best[:, lo : lo + n]
        best[:, lo : lo + n] = np.where(improves, child_best, best[:, lo : lo + n])
        arg[:, lo : lo + n] = np.where(improves, child_arg, arg[:, lo : lo + n])
    return best, arg


def follow(values_on_keeps: np.ndarray, arg: np.ndarray) -> np.ndarray:
    """Value of the keep an agent actually chose, for each full roll."""
    return np.take_along_axis(values_on_keeps, arg[:, ROLL_LO:], axis=1)


def solve_optimal() -> np.ndarray:
    """Exact optimal value of every state: expected further score under optimal play.

    One backward pass over the 8,192 masks. Returns a flat ``(N_STATE,)`` table; the expected
    score of a whole game is ``V[START]``.
    """
    values = np.zeros(N_STATE)
    for mask in masks_by_descending_popcount():
        for yz in (0, 1) if (mask >> 12) & 1 else (0,):
            reward, successor, valid, _cat = option_arrays(mask, yz)
            # Best category for each (upper, roll): the option maximising reward + successor value.
            end_of_turn = np.where(valid, reward + values[successor], NEG).max(-1)
            after_second, _ = submax_arg(expect(end_of_turn))
            after_first, _ = submax_arg(expect(after_second[:, ROLL_LO:]))
            values[((mask * 2 + yz) * N_UPPER) + UPPER] = after_first[:, ROLL_LO:] @ P_ROLL
    return values
