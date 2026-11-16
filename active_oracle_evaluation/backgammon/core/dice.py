"""Dice: single random rolls and the 21 distinct roll types with their probabilities."""

from __future__ import annotations

import random
from functools import lru_cache

#: A roll is a tuple of die faces. Doubles are expanded to four dice at
#: move-generation time, not here.
Roll = tuple[int, int]


def roll(rng: random.Random | None = None) -> Roll:
    """Roll two dice. Pass an :class:`random.Random` for reproducible games."""
    r = rng or random
    return (r.randint(1, 6), r.randint(1, 6))


@lru_cache(maxsize=1)
def distinct_rolls() -> tuple[tuple[Roll, float], ...]:
    """The 21 distinct unordered rolls paired with their probabilities.

    There are 36 equally-likely ordered outcomes. Grouping unordered, the 6
    doubles have probability 1/36 each and the 15 non-doubles 2/36 each. The
    probabilities sum to 1; search averages over opponent rolls with these weights.
    """
    out: list[tuple[Roll, float]] = []
    for high in range(1, 7):
        for low in range(1, high + 1):
            # Standard dice probabilities: 1/36 per double, 2/36 per non-double.
            prob = (1 / 36) if high == low else (2 / 36)
            out.append(((high, low), prob))
    return tuple(out)
