"""The dice lattice: every multiset of at most five d6, and the completion operator over it.

``KEEPS`` holds all 462 multisets of size 0..5 over faces 1..6, ordered by size, so a *keep* (the
dice held between rolls) and a *roll* (five dice) are the same kind of object, and "t is
obtainable from s by dropping dice" is multiset containment. The 252 size-5 entries at the end
are the rolls; ``ROLL_LO`` is where they start.

All tables are fixed by the rules, so they are built once at import and reused for every state
of the DP. :func:`expect` gives the value of holding each keep: complete it with fair dice and
take the expectation. ``CHILD_INDEX`` supports the bottom-up best-sub-multiset sweep in
:mod:`active_oracle_evaluation.yahtzee.dp`.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial

import numpy as np

#: Faces of a d6 and dice per roll, fixed by the rules.
FACES = range(1, 7)
N_DICE = 5

#: All multisets of size 0..5 over the six faces, ordered by size then lexicographically.
KEEPS: list[tuple[int, ...]] = []
for _k in range(N_DICE + 1):
    KEEPS.extend(combinations_with_replacement(FACES, _k))
KEEP_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(KEEPS)}
N_KEEP = len(KEEPS)

#: Number of multisets of each size, and the offset where each size layer begins.
LAYER_SIZE = [len(list(combinations_with_replacement(FACES, k))) for k in range(N_DICE + 1)]
LAYER_OFFSET = np.cumsum([0] + LAYER_SIZE[:-1])

#: The full five-dice rolls: the last layer of the lattice.
ROLL_LO = int(LAYER_OFFSET[N_DICE])
ROLLS: list[tuple[int, ...]] = KEEPS[ROLL_LO:]
N_ROLL = len(ROLLS)


def multiset_probability(ms: tuple[int, ...]) -> float:
    """Probability of rolling exactly this multiset with ``len(ms)`` fair dice."""
    n = len(ms)
    if n == 0:
        return 1.0
    counts: dict[int, int] = {}
    for d in ms:
        counts[d] = counts.get(d, 0) + 1
    ways = factorial(n)
    for c in counts.values():
        ways //= factorial(c)
    return ways / (6.0**n)


#: Probability of each of the 252 full rolls.
P_ROLL = np.array([multiset_probability(r) for r in ROLLS])

# --- completion operator -------------------------------------------------------------------
# For keep-size k, COMPLETE_INDEX[k][i] lists the rolls reachable by adding 5-k fair dice to keep i,
# and COMPLETE_WEIGHT[k][i] their probabilities.
COMPLETE_INDEX: list[np.ndarray] = []
COMPLETE_WEIGHT: list[np.ndarray] = []
for _k in range(N_DICE + 1):
    _comps = list(combinations_with_replacement(FACES, N_DICE - _k))
    _w = np.array([multiset_probability(c) for c in _comps])
    _lo = LAYER_OFFSET[_k]
    _idx = np.empty((LAYER_SIZE[_k], len(_comps)), dtype=np.int32)
    for _i in range(LAYER_SIZE[_k]):
        _keep = KEEPS[_lo + _i]
        for _j, _c in enumerate(_comps):
            _idx[_i, _j] = KEEP_INDEX[tuple(sorted(_keep + _c))] - ROLL_LO
    COMPLETE_INDEX.append(_idx)
    COMPLETE_WEIGHT.append(np.tile(_w, (LAYER_SIZE[_k], 1)))

# --- sub-multiset operator -----------------------------------------------------------------
# CHILD_INDEX[k-1][i] are the size-(k-1) multisets obtained by dropping one die from multiset i.
# Fewer than 6 distinct children are padded with a repeat, which is harmless under max.
CHILD_INDEX: list[np.ndarray] = []
for _k in range(1, N_DICE + 1):
    _lo = LAYER_OFFSET[_k]
    _idx = np.empty((LAYER_SIZE[_k], 6), dtype=np.int32)
    for _i in range(LAYER_SIZE[_k]):
        _keep = KEEPS[_lo + _i]
        _kids = sorted({KEEP_INDEX[_keep[:_j] + _keep[_j + 1 :]] for _j in range(len(_keep))})
        while len(_kids) < 6:
            _kids.append(_kids[0])
        _idx[_i] = _kids
    CHILD_INDEX.append(_idx)


def expect(values_on_rolls: np.ndarray) -> np.ndarray:
    """``(B, 252)`` values on full rolls → ``(B, 462)`` value of holding each keep."""
    batch = values_on_rolls.shape[0]
    out = np.empty((batch, N_KEEP))
    for k in range(N_DICE + 1):
        lo = LAYER_OFFSET[k]
        out[:, lo : lo + LAYER_SIZE[k]] = (
            values_on_rolls[:, COMPLETE_INDEX[k]] * COMPLETE_WEIGHT[k]
        ).sum(-1)
    return out
