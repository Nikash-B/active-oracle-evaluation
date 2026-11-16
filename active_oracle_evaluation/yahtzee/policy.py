"""Exact policy evaluation, and exact oracle regret along an agent's own trajectory.

An agent is an approximate end-of-turn value table ``Vhat`` plus exact within-turn lookahead
against its own ``Vhat``. Every agent in the panel (optimal, noisy, greedy, heuristic,
fixed-order) differs only in ``Vhat`` or in the preference that ranks its scoring options. Two
consequences:

* evaluating any agent exactly costs one backward pass, whatever produced ``Vhat``;
* the panel is a one-parameter family in the error of ``Vhat``, which the reference-quality
  experiment (``python -m active_oracle_evaluation.experiments.yahtzee_reference_quality``)
  varies.

The pass computes, exactly, ``V_pi`` (expected final score), ``M_pi`` (its second moment, so the
score SD is exact and can be checked against
:data:`~active_oracle_evaluation.yahtzee.OPTIMAL_SCORE_SD`) and ``R_pi`` (expected cumulative
oracle regret along the agent's own state distribution).

Correctness check: ``R_pi(start) == V*(start) - V_pi(start)``, the performance-difference lemma.
The two sides are accumulated over different quantities and agree only if both the policy's own
dynamics and the per-decision regret are right.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.yahtzee.dice import P_ROLL, ROLL_LO, expect
from active_oracle_evaluation.yahtzee.dp import (
    N_STATE,
    N_UPPER,
    NEG,
    START,
    UPPER,
    follow,
    masks_by_descending_popcount,
    option_arrays,
    submax_arg,
)


def evaluate(vhat: np.ndarray, vstar: np.ndarray, *, preference=None):
    """Exactly evaluate the policy "exact turn lookahead against ``vhat``".

    Returns ``(v_pi, m_pi, r_pi)`` as flat state tables. ``vstar`` is the optimal table from
    :func:`~active_oracle_evaluation.yahtzee.dp.solve_optimal`; it is used *only* to price
    regret, never to choose an action.

    ``preference`` overrides *how the agent ranks its options* while leaving the machinery — exact
    keep lookahead, moment propagation, regret accounting — untouched. It is called as
    ``preference(reward, successor, valid, category)`` and must return a ``(64, 252, K)`` score
    whose argmax is the category the agent fills. The default is the value-table rule
    ``reward + vhat[successor]``, which is what makes the panel a one-parameter family in the error
    of ``vhat``. A rule-based agent supplies its own ranking and ignores ``vhat`` entirely; that is
    the only hook it needs, because "exact lookahead against your own preferences" is still an exact
    policy and still evaluates in one backward pass.
    """
    v_pi = np.zeros(N_STATE)
    m_pi = np.zeros(N_STATE)
    r_pi = np.zeros(N_STATE)

    for mask in masks_by_descending_popcount():
        for yz in (0, 1) if (mask >> 12) & 1 else (0,):
            reward, successor, valid, category = option_arrays(mask, yz)
            here = ((mask * 2 + yz) * N_UPPER) + UPPER

            # --- what the agent believes, and therefore what it does ---------------------
            score = (
                reward + vhat[successor]
                if preference is None
                else preference(reward, successor, valid, category)
            )
            believed = np.where(valid, score, NEG)
            chosen = believed.argmax(-1)[..., None]
            believed_end = np.take_along_axis(believed, chosen, -1)[..., 0]
            keep2_hat, arg2 = submax_arg(expect(believed_end))
            keep1_hat, arg1 = submax_arg(expect(follow(keep2_hat, arg2)))

            reward_taken = np.take_along_axis(reward, chosen, -1)[..., 0]
            state_next = np.take_along_axis(successor, chosen, -1)[..., 0]

            # --- the oracle's view of the same turn, which prices the regret -------------
            optimal = np.where(valid, reward + vstar[successor], NEG)
            optimal_end = optimal.max(-1)
            regret_category = optimal_end - np.take_along_axis(optimal, chosen, -1)[..., 0]
            best2, _ = submax_arg(expect(optimal_end))
            regret_keep2 = best2[:, ROLL_LO:] - follow(expect(optimal_end), arg2)
            best1, _ = submax_arg(expect(best2[:, ROLL_LO:]))
            regret_keep1 = best1[:, ROLL_LO:] - follow(expect(best2[:, ROLL_LO:]), arg1)

            # --- true moments and regret, propagated along the agent's OWN choices -------
            # E[(w + X)^2] = w^2 + 2w E[X] + E[X^2]: the second moment needs the first, which is
            # why v_pi and m_pi are accumulated together in one pass.
            value_end = reward_taken + v_pi[state_next]
            second_end = reward_taken**2 + 2.0 * reward_taken * v_pi[state_next] + m_pi[state_next]
            regret_end = regret_category + r_pi[state_next]

            value2, second2, regret2 = (
                follow(expect(x), arg2) for x in (value_end, second_end, regret_end)
            )
            regret2 = regret2 + regret_keep2
            value1, second1, regret1 = (follow(expect(x), arg1) for x in (value2, second2, regret2))
            regret1 = regret1 + regret_keep1

            v_pi[here] = value1 @ P_ROLL
            m_pi[here] = second1 @ P_ROLL
            r_pi[here] = regret1 @ P_ROLL
    return v_pi, m_pi, r_pi


def score_sd(v_pi: np.ndarray, m_pi: np.ndarray, state: int = START) -> float:
    """Exact SD of the final score from the first and second moments at ``state``."""
    mean = v_pi[state]
    return float(np.sqrt(max(m_pi[state] - mean * mean, 0.0)))


def identity_error(
    v_pi: np.ndarray, r_pi: np.ndarray, vstar: np.ndarray, state: int = START
) -> float:
    """``|R_pi - (V* - V_pi)|`` at ``state`` — the performance-difference lemma, as a residual."""
    return float(abs(r_pi[state] - (vstar[state] - v_pi[state])))
