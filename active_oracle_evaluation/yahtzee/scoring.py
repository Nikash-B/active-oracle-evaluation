"""Category scoring for the 252 full rolls, and the Joker rules.

``SCORE[r, c]`` is what roll ``r`` scores in category ``c`` under official Hasbro rules, ignoring
bonuses. The upper bonus and the Extra Yahtzee Bonus depend on scorecard state, not on the roll,
so the DP applies them.

The Joker rules apply only when a second or later Yahtzee is rolled and the Yahtzee box already
holds 50. The player must then use the roll in its own upper box if that box is free; otherwise in
any free lower box, where the straights and full house pay their face value rather than zero; and
only if neither is available does it go as a zero into a free upper box. The +100 Extra Yahtzee
Bonus is added on top, and is paid only when the Yahtzee box holds 50.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.yahtzee import N_CAT
from active_oracle_evaluation.yahtzee.dice import N_ROLL, ROLLS

# --- per-roll dice statistics ---------------------------------------------------------------
_counts = np.zeros((N_ROLL, 7), dtype=np.int64)
for _i, _r in enumerate(ROLLS):
    for _d in _r:
        _counts[_i, _d] += 1
_total = np.array([sum(r) for r in ROLLS], dtype=np.int64)
_max_of_a_kind = _counts[:, 1:].max(1)
_present = _counts[:, 1:] > 0

#: ``SCORE[roll, category]`` under official rules, bonuses excluded. The fixed payouts (25, 30,
#: 40, 50) are the official rules.
SCORE = np.zeros((N_ROLL, N_CAT), dtype=np.int64)
for _f in range(1, 7):  # ones..sixes: sum of that face
    SCORE[:, _f - 1] = _counts[:, _f] * _f
SCORE[:, 6] = np.where(_max_of_a_kind >= 3, _total, 0)  # three of a kind: sum of all dice
SCORE[:, 7] = np.where(_max_of_a_kind >= 4, _total, 0)  # four of a kind: sum of all dice
_has3 = (_counts[:, 1:] == 3).any(1)
_has2 = (_counts[:, 1:] == 2).any(1)
SCORE[:, 8] = np.where(_has3 & _has2, 25, 0)  # full house (5-of-a-kind is NOT one here)
_small = np.zeros(N_ROLL, dtype=bool)
for _s in range(3):  # 1234, 2345, 3456
    _small |= _present[:, _s : _s + 4].all(1)
SCORE[:, 9] = np.where(_small, 30, 0)
_large = _present[:, 0:5].all(1) | _present[:, 1:6].all(1)
SCORE[:, 10] = np.where(_large, 40, 0)
SCORE[:, 11] = _total  # chance
SCORE[:, 12] = np.where(_max_of_a_kind == 5, 50, 0)  # yahtzee

#: Face -> index of the five-of-that-face roll.
YAHTZEE_ROLL: dict[int, int] = {r[0]: i for i, r in enumerate(ROLLS) if len(set(r)) == 1}

#: Lower-box payout under the Joker rules; ``None`` means "sum of the dice", i.e. 5 x face.
#: The straights and the full house pay full value here (official forced-Joker rules; paper,
#: appendix "Solitaire Yahtzee: exact ground truth", paragraph "The solved game").
JOKER_LOWER_SCORE: dict[int, int | None] = {
    6: None,  # three of a kind -> 5 x face
    7: None,  # four of a kind  -> 5 x face
    8: 25,  # full house
    9: 30,  # small straight
    10: 40,  # large straight
    11: None,  # chance          -> 5 x face
}


def joker_options(face: int, free: list[int]) -> list[tuple[int, int]]:
    """``(category, score)`` choices for a Joker Yahtzee of ``face``, given the free categories.

    Encodes the forced ordering: the matching upper box if free, else any free lower box, else a
    zero into a free upper box. Returns every legal choice rather than the best one — the caller
    maximises, because which is best depends on the value of the resulting state.
    """
    upper = face - 1
    if upper in free:
        return [(upper, 5 * face)]
    lower_free = [c for c in free if 6 <= c <= 11]
    if lower_free:
        return [
            (c, 5 * face if JOKER_LOWER_SCORE[c] is None else JOKER_LOWER_SCORE[c])
            for c in lower_free
        ]
    return [(c, 0) for c in free]
