"""Yahtzee decision suite: nodes drawn from one agent's play, labelled for every agent.

A decision node is ``(state, roll, rerolls_left)``: the two keep decisions and the category
decision of each turn. Every game contributes exactly 39 nodes (13 turns x 3), so a whole-game
regret divided by 39 is directly comparable to a per-decision suite mean.

For each node and each agent we record the action the agent takes at that node and the exact
regret of that action against ``V*``. By default the trajectory follows the optimal agent;
``mixture=True`` instead picks the driving agent uniformly at random per game.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.yahtzee.dice import (
    COMPLETE_INDEX,
    COMPLETE_WEIGHT,
    LAYER_OFFSET,
    N_ROLL,
    P_ROLL,
    ROLL_LO,
    expect,
)
from active_oracle_evaluation.yahtzee.dp import N_UPPER, NEG, option_arrays, submax_arg

#: Decisions in one game: 13 turns x (keep, keep, category) (paper, section "In a solved game,
#: suite regret recovers the ranking but not the magnitudes").
DECISIONS_PER_GAME = 39


def _sample_roll(rng: np.random.Generator) -> int:
    """Index of a fresh five-dice roll, drawn from the exact multiset distribution."""
    return int(rng.choice(N_ROLL, p=P_ROLL))


def _sample_completion(rng: np.random.Generator, keep: int) -> int:
    """Roll index after completing ``keep`` (a lattice index) back up to five dice."""
    size = int(np.searchsorted(LAYER_OFFSET, keep, side="right")) - 1
    row = keep - LAYER_OFFSET[size]
    choices, weights = COMPLETE_INDEX[size][row], COMPLETE_WEIGHT[size][row]
    return int(rng.choice(choices, p=weights))


def _turn_tables(end_values: np.ndarray):
    """From end-of-turn values ``(B, 252)`` to the two keep-stage tables and their subset maxima.

    ``keep1`` is the value of holding a keep with **one** reroll to come, ``keep2`` with **two**.
    ``best1`` / ``best2`` are their subset maxima, i.e. the value of the best keep available from a
    given roll. With optimal values, these give the best-keep value in the keep-node regret.
    """
    keep1 = expect(end_values)
    best1, arg1 = submax_arg(keep1)
    keep2 = expect(best1[:, ROLL_LO:])
    best2, arg2 = submax_arg(keep2)
    return keep1, best1, arg1, keep2, best2, arg2


def _panel_beliefs(reward, successor, valid, category, row, vhats, prefs) -> np.ndarray:
    """Every agent's ranking of the category options for one upper total, ``(agents, 252, K)``."""
    believed = np.where(valid[row], reward[row] + vhats[:, successor[row][0]], NEG)
    for a, pref in enumerate(prefs):
        if pref is not None:
            believed[a] = np.where(
                valid[row][0], pref(reward[row], successor[row], valid[row], category)[0], NEG
            )
    return believed


def _unflatten(state: int) -> tuple[int, int, int]:
    """``(mask, yz, upper)`` of a flat state index."""
    return (state // N_UPPER) // 2, (state // N_UPPER) % 2, state % N_UPPER


def generate(
    vstar: np.ndarray,
    panel: dict[str, dict],
    n_games: int,
    seed: int,
    *,
    mixture: bool = False,
) -> dict:
    """Play ``n_games`` and return the labelled suite.

    Returns ``action`` and ``regret`` arrays of shape ``(n_agents, n_games * 39)``. Regret is in
    points of expected final score, and is zero exactly when the agent's action is optimal at that
    node.
    """
    names = list(panel)
    vhats = np.stack([panel[n]["vhat"] for n in names])
    prefs = [panel[n]["preference"] for n in names]
    n_agents = len(names)
    rng = np.random.default_rng(seed)

    total = n_games * DECISIONS_PER_GAME
    out = {
        "action": np.empty((n_agents, total), dtype=np.int32),
        "regret": np.empty((n_agents, total), dtype=np.float64),
    }
    at = 0

    for game in range(n_games):
        # Which agent's play defines this game's trajectory.
        driver = int(rng.integers(n_agents)) if mixture else names.index("optimal")
        mask, yz, upper = 0, 0, 0

        for _turn in range(13):
            reward, successor, valid, _cat = option_arrays(mask, yz)
            row = slice(upper, upper + 1)  # this state's single upper total

            # End-of-turn values: the oracle's, and every agent's own belief.
            star_end = np.where(valid[row], reward[row] + vstar[successor[row]], NEG).max(-1)
            believed = _panel_beliefs(reward, successor, valid, _cat, row, vhats, prefs)
            chosen_cat = believed.argmax(-1)  # (agents, 252)
            agent_end = np.take_along_axis(believed, chosen_cat[..., None], -1)[..., 0]

            _k1s, best1s, _a1s, k2s, best2s, _a2s = _turn_tables(star_end)
            _k1a, _b1a, arg1a, _k2a, _b2a, arg2a = _turn_tables(agent_end)

            # ---- node 1: first keep, two rerolls to come --------------------------------
            roll1 = _sample_roll(rng)
            keep_a2 = arg2a[:, ROLL_LO + roll1]  # each agent's chosen keep
            reg2 = best2s[0, ROLL_LO + roll1] - k2s[0, keep_a2]
            _record(out, at, keep_a2, reg2)
            at += 1

            # ---- node 2: second keep, one reroll to come -------------------------------
            roll2 = _sample_completion(rng, int(keep_a2[driver]))
            keep_a1 = arg1a[:, ROLL_LO + roll2]
            reg1 = best1s[0, ROLL_LO + roll2] - _k1s[0, keep_a1]
            _record(out, at, keep_a1, reg1)
            at += 1

            # ---- node 3: the category ---------------------------------------------------
            roll3 = _sample_completion(rng, int(keep_a1[driver]))
            cat_choice = chosen_cat[:, roll3]
            star_row = np.where(
                valid[row][0, roll3], reward[row][0, roll3] + vstar[successor[row][0, roll3]], NEG
            )
            reg0 = star_row.max() - star_row[cat_choice]
            _record(out, at, cat_choice, reg0)
            at += 1

            # advance the trajectory under the driving agent's own choice
            take = int(cat_choice[driver])
            nxt = int(successor[row][0, roll3, take])
            mask, yz, upper = _unflatten(nxt)

    return out


def _record(out, at, action, regret) -> None:
    """Write one node's per-agent action and regret."""
    out["action"][:, at] = action
    out["regret"][:, at] = regret


def simulate_on_policy(
    vstar: np.ndarray, vhat: np.ndarray, preference, n_games: int, seed: int
) -> dict:
    """Play ``n_games`` under the agent's own policy; return per-game score and total regret.

    Returns ``{"score": [n_games], "regret": [n_games]}`` in points of final score.
    ``mean(score)`` estimates ``V^pi(start)`` and, by the performance-difference lemma,
    ``mean(regret)`` estimates ``V*(start) - V^pi(start)``; the ``yahtzee_trials`` experiment
    checks both.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n_games)
    regrets = np.empty(n_games)

    for game in range(n_games):
        mask, yz, upper = 0, 0, 0
        score = 0.0
        regret = 0.0

        for _turn in range(13):
            reward, successor, valid, cat = option_arrays(mask, yz)
            row = slice(upper, upper + 1)
            star_end = np.where(valid[row], reward[row] + vstar[successor[row]], NEG).max(-1)
            # Same agent block as generate(): `believed` is (1, 252, 13), and a preference agent
            # replaces the value-table ranking entirely.
            if preference is None:
                believed = np.where(valid[row], reward[row] + vhat[successor[row][0]], NEG)
            else:
                believed = np.where(
                    valid[row][0], preference(reward[row], successor[row], valid[row], cat)[0], NEG
                )[None]
            chosen_cat = believed.argmax(-1)  # (1, 252)
            agent_end = np.take_along_axis(believed, chosen_cat[..., None], -1)[..., 0]

            k1s, best1s, _a1s, k2s, best2s, _a2s = _turn_tables(star_end)
            _k1a, _b1a, arg1a, _k2a, _b2a, arg2a = _turn_tables(agent_end)

            roll1 = _sample_roll(rng)
            keep2 = int(arg2a[0, ROLL_LO + roll1])
            regret += float(best2s[0, ROLL_LO + roll1] - k2s[0, keep2])

            roll2 = _sample_completion(rng, keep2)
            keep1 = int(arg1a[0, ROLL_LO + roll2])
            regret += float(best1s[0, ROLL_LO + roll2] - k1s[0, keep1])

            roll3 = _sample_completion(rng, keep1)
            take = int(chosen_cat[0, roll3])
            star_row = np.where(
                valid[row][0, roll3], reward[row][0, roll3] + vstar[successor[row][0, roll3]], NEG
            )
            regret += float(star_row.max() - star_row[take])

            score += float(reward[row][0, roll3, take])
            nxt = int(successor[row][0, roll3, take])
            mask, yz, upper = _unflatten(nxt)

        scores[game] = score
        regrets[game] = regret

    return {"score": scores, "regret": regrets}
