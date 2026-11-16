"""Exact solitaire Yahtzee: engine, DP, policy evaluation, and the oracle-regret metric.

Solitaire Yahtzee is small enough to solve exactly, so every quantity that the backgammon
experiments estimate against an imperfect oracle can be measured here against ground truth.
The ``active_oracle_evaluation.experiments.yahtzee_*`` experiments take their domain-neutral
statistics from ``active_oracle_evaluation.eval``; this package itself depends on numpy only.
"""

from __future__ import annotations

#: Number of scorecard categories. Category ``c`` is bit ``c`` of the 13-bit mask, in scorecard
#: order: ones..sixes, three and four of a kind, full house, small and large straight, chance,
#: Yahtzee. Fixed by the official rules.
N_CAT = 13
FULL_MASK = (1 << N_CAT) - 1

#: Upper-section bonus of 35 points once the upper total reaches 63. The DP caps the upper score
#: at 63 because further upper points cannot change the bonus. Both values are the official
#: rules (paper, appendix "Solitaire Yahtzee: exact ground truth", paragraph "The solved game").
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS = 35

#: Expected score of the optimal policy under the official rules, and the standard deviation of
#: its final score, as computed by ``active_oracle_evaluation.yahtzee.dp`` (paper, appendix
#: "Solitaire Yahtzee: exact ground truth", paragraph "The solved game": 254.5877 and 59.6079,
#: the SD rounded here to two decimals). The experiments check
#: the DP against these values.
OPTIMAL_EXPECTED_SCORE = 254.5877
OPTIMAL_SCORE_SD = 59.61

__all__ = [
    "N_CAT",
    "FULL_MASK",
    "UPPER_BONUS_THRESHOLD",
    "UPPER_BONUS",
    "OPTIMAL_EXPECTED_SCORE",
    "OPTIMAL_SCORE_SD",
]
