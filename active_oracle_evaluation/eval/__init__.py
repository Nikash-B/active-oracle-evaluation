"""Evaluation harness: duplicate-dice play and per-decision scoring against an oracle.

The package-level names are the ones the evaluation tools import directly; every other
component is imported from its own submodule.
"""

from __future__ import annotations

from active_oracle_evaluation.eval.dice_bank import DiceBank
from active_oracle_evaluation.eval.duplicate import play_duplicate_pair
from active_oracle_evaluation.eval.scoring import score_decisions

__all__ = [
    "DiceBank",
    "play_duplicate_pair",
    "score_decisions",
]
