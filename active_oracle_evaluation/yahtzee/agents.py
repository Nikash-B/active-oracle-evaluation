"""The Yahtzee evaluation panel: one value-table family plus two rule-based agents.

Every agent here is exactly evaluable by
:func:`~active_oracle_evaluation.yahtzee.policy.evaluate` in one backward pass.

**Value-table agents** rank options by ``reward + V̂[successor]`` and differ only in ``V̂``:
``optimal`` uses the exact table, ``noisy`` perturbs it, and ``greedy`` sets it to zero. Varying
the error of ``V̂`` puts near-tied pairs in the panel by construction.

**Rule-based agents** ignore ``V̂`` and rank by their own rule, so they do not lie on the
one-parameter curve of the value-table family. ``fixed_order`` fills free boxes in a fixed
order whatever is rolled; ``heuristic`` takes the immediate reward plus a per-category bias.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.yahtzee import N_CAT

#: σ grid for the value-table error family (the paper's η, appendix "Solitaire Yahtzee: exact
#: ground truth", paragraph "Agent panel and suites"). The paper names η = 0.5 and η = 1; the
#: full six-value grid is chosen for this study. The smallest values give near-tied agents; the
#: larger ones spread the panel.
NOISE_SIGMAS = (0.5, 1.0, 2.0, 5.0, 15.0, 50.0)

#: An arbitrary fixed filling order for the dice-blind agent, chosen for this study.
FIXED_FILL_ORDER = (12, 0, 8, 3, 10, 5, 1, 9, 6, 2, 11, 4, 7)


def noise_seed(sigma: float, base: int) -> int:
    """Seed for the σ agent, derived from σ itself so it does not depend on panel order."""
    # The offset round(σ * 1000) is chosen for this study.
    return base + int(round(sigma * 1000))


def noisy_value_table(vstar: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """``V*`` plus i.i.d. Gaussian error — the one-parameter family in value-table error."""
    return vstar + np.random.default_rng(seed).normal(0.0, sigma, vstar.shape)


def greedy_value_table(vstar: np.ndarray) -> np.ndarray:
    """``V̂ ≡ 0``: maximise this turn's score, ignore everything after it."""
    return np.zeros_like(vstar)


def _rank_by_category(rank: np.ndarray):
    """Build a preference that ranks options by a fixed per-category score.

    ``rank[c]`` is the score for filling category ``c``, higher first. The returned callable has
    the signature :func:`~active_oracle_evaluation.yahtzee.policy.evaluate` expects. The agent
    still chooses its keeps by exact lookahead against this preference, so only its choice of box
    ignores the dice.
    """

    def preference(reward, successor, valid, category):
        """Score each option by the rank of the category it fills."""
        scores = np.where(category >= 0, rank[np.maximum(category, 0)], -np.inf)
        return np.broadcast_to(scores[None, :, :], reward.shape)

    return preference


def fixed_order_preference():
    """Dice-blind: fill free boxes in :data:`FIXED_FILL_ORDER`, whatever is rolled."""
    rank = np.empty(N_CAT)
    for position, category in enumerate(FIXED_FILL_ORDER):
        rank[category] = N_CAT - position
    return _rank_by_category(rank)


#: Per-category bias for the rule-based player, in points; small next to real scores. The values
#: are chosen for this study.
HEURISTIC_BIAS = np.zeros(N_CAT)
HEURISTIC_BIAS[11] = -6.0  # chance: hold it back as the dumping ground for a bad roll
HEURISTIC_BIAS[12] = +3.0  # yahtzee: worth taking when it actually shows up
for upper_category in range(6):
    HEURISTIC_BIAS[upper_category] = +1.5  # upper boxes: mild pull, to chase the 35-point bonus


def heuristic_preference():
    """A rule-based player that ranks by immediate reward plus :data:`HEURISTIC_BIAS`.

    It is dice-aware but has no lookahead and no value table. The biases keep ``chance`` free as
    a dumping ground, take a Yahtzee when it appears, and lean on the upper boxes for the bonus.
    """

    def preference(reward, successor, valid, category):
        """Score each option by its immediate reward plus the category bias."""
        bias = np.where(category >= 0, HEURISTIC_BIAS[np.maximum(category, 0)], 0.0)
        return reward + bias[None, :, :]

    return preference


#: Wide expected-score band each agent must land in; it flags a broken agent, not a weak one.
#: The bands are chosen for this study.
PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "optimal": (254.0, 255.0),
    "noisy_s0.5": (250.0, 255.0),
    "noisy_s1": (248.0, 255.0),
    "noisy_s2": (240.0, 254.0),
    "noisy_s5": (215.0, 250.0),
    "noisy_s15": (140.0, 210.0),
    "noisy_s50": (90.0, 170.0),
    "greedy": (190.0, 240.0),
    "heuristic": (195.0, 245.0),
    "fixed_order": (35.0, 90.0),
}


def check_plausibility(scores: dict[str, float]) -> list[str]:
    """Names whose expected score falls outside :data:`PLAUSIBILITY`. Empty means all plausible."""
    return [
        f"{name}: {value:.2f} outside {PLAUSIBILITY[name]}"
        for name, value in scores.items()
        if name in PLAUSIBILITY and not (PLAUSIBILITY[name][0] <= value <= PLAUSIBILITY[name][1])
    ]


def build_panel(vstar: np.ndarray, seed: int = 20260831) -> dict[str, dict]:
    """The panel as ``{name: {"vhat": ..., "preference": ...}}``, ready for ``evaluate(**spec)``.

    Order: ``optimal``, the noisy family in increasing σ, ``greedy``, then the rule-based agents.
    The default ``seed`` is chosen for this study.
    """
    panel: dict[str, dict] = {"optimal": {"vhat": vstar, "preference": None}}
    for sigma in NOISE_SIGMAS:
        panel[f"noisy_s{sigma:g}"] = {
            "vhat": noisy_value_table(vstar, sigma, noise_seed(sigma, seed)),
            "preference": None,
        }
    panel["greedy"] = {"vhat": greedy_value_table(vstar), "preference": None}
    zeros = np.zeros_like(vstar)
    panel["heuristic"] = {"vhat": zeros, "preference": heuristic_preference()}
    panel["fixed_order"] = {"vhat": zeros, "preference": fixed_order_preference()}
    return panel


__all__ = [
    "NOISE_SIGMAS",
    "FIXED_FILL_ORDER",
    "PLAUSIBILITY",
    "noise_seed",
    "noisy_value_table",
    "greedy_value_table",
    "fixed_order_preference",
    "heuristic_preference",
    "build_panel",
    "check_plausibility",
]
