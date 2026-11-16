"""Reproducible dice streams for duplicate-dice (common-random-number) evaluation.

A :class:`DiceBank` turns one integer ``seed`` into a deterministic roll stream per
trial index. The same ``(seed, trial)`` always yields the same rolls, so the duplicate
harness can play a trial twice, with seats swapped, on identical dice.

Each stream is seeded with the string ``f"{seed}:{trial}"`` (a format chosen for this study):
``random.Random`` hashes a string seed with SHA-512, so the stream does not depend on
``PYTHONHASHSEED`` and is the same in every process.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from active_oracle_evaluation.backgammon.core.dice import Roll
from active_oracle_evaluation.backgammon.core.dice import roll as roll_dice


@dataclass(frozen=True)
class DiceBank:
    """A deterministic source of per-trial roll streams."""

    seed: int

    def stream(self, trial: int) -> Iterator[Roll]:
        """Yield the unbounded deterministic roll sequence for ``trial``."""
        rng = random.Random(f"{self.seed}:{trial}")
        while True:
            yield roll_dice(rng)
