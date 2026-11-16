"""Variance reduction of duplicate-dice pairing relative to independent single games.

Backs the paper's variance-reduction paragraph (appendix "Background and conventions",
paragraph "Variance reduction").
"""

from __future__ import annotations

from dataclasses import dataclass

#: Games per duplicate-dice trial: one dice stream played twice (paper, section "Evaluation
#: protocol", paragraph "Rating").
GAMES_PER_PAIR = 2


@dataclass(frozen=True)
class VarianceReduction:
    """Games-saving factor of the paired design; ``factor`` > 1 means fewer games are needed."""

    factor: float


def variance_reduction(paired_sd: float, unpaired_sd: float) -> VarianceReduction:
    """Return ``unpaired_sd^2 / (GAMES_PER_PAIR * paired_sd^2)`` as a games-saving factor.

    For a fixed number of games ``G``, the variance of the unpaired mean is
    ``unpaired_sd^2 / G``; the paired design spends ``GAMES_PER_PAIR`` games per trial,
    giving ``GAMES_PER_PAIR * paired_sd^2 / G``.
    """
    if unpaired_sd <= 0:
        raise ValueError("unpaired_sd must be positive")
    if paired_sd < 0:
        raise ValueError("paired_sd must be non-negative")
    paired_cost_var = GAMES_PER_PAIR * paired_sd**2
    factor = float("inf") if paired_cost_var == 0 else (unpaired_sd**2) / paired_cost_var
    return VarianceReduction(factor=factor)
