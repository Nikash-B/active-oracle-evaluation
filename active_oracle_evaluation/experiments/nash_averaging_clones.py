"""Clone cells of Nash averaging: duplicate each agent and measure how the other ratings move.

Each agent of the round-robin is duplicated in turn. The clone inherits its original's wins and
games against every other agent and plays an evenly split match against the original. On a
field with a dominant class (agents with identical payoff columns that beat every other agent)
the maximum-entropy Nash equilibrium is uniform on that class, so it is computed exactly.
Bradley-Terry is fitted by the MM algorithm without a prior, anchored at one agent.
"""

import math

import numpy as np


def precision_matrix(sidecar):
    """Return the agents, wins and games matrices of a round-robin sidecar."""
    agents = list(sidecar["agents"])
    idx = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    wins = np.full((n, n), np.nan)  # wins[i, j] = games i won of games[i, j]
    games = np.zeros((n, n))
    for pairing in sidecar["pairings"]:
        i, j = idx[pairing["a"]], idx[pairing["b"]]
        wins[i, j] = pairing["a_wins"]
        wins[j, i] = pairing["games"] - pairing["a_wins"]
        games[i, j] = games[j, i] = pairing["games"]
    assert not np.isnan(wins[~np.eye(n, dtype=bool)]).any(), "missing pairing"
    return agents, wins, games


def winrate_matrix(wins, games):
    """Return the win-probability matrix, 0.5 on the diagonal and for unplayed pairs."""
    n = wins.shape[0]
    p = np.full((n, n), 0.5)
    mask = games > 0
    p[mask] = wins[mask] / games[mask]
    np.fill_diagonal(p, 0.5)
    return p


def payoff(p):
    """Return the antisymmetric payoff (P - P^T) / 2 in win-rate units."""
    return (p - p.T) / 2.0


def exact_maxent_if_dominant(a):
    """Return the exact maximum-entropy equilibrium and ratings when a dominant class exists."""
    n = a.shape[0]
    for i in range(n):
        cls = [j for j in range(n) if np.array_equal(a[:, j], a[:, i])]
        outside = [j for j in range(n) if j not in cls]
        if all(a[i, j] > 0.0 for j in outside):
            p = np.zeros(n)
            p[cls] = 1.0 / len(cls)
            skill = a @ p
            assert float(np.max(a @ p)) <= 1e-15
            return p, skill
    return None


def bt_elo_fit(wins, games, anchor, iterations, convergence):
    """Fit Bradley-Terry by MM on log-strength and return Elo with ``anchor`` at 0."""
    n = wins.shape[0]
    theta = np.zeros(n)
    total_wins = np.nansum(np.where(np.eye(n, dtype=bool), 0.0, wins), axis=1)
    for _ in range(iterations):
        s = np.exp(theta)
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and games[i, j] > 0:
                    denom[i] += games[i, j] / (s[i] + s[j])
        theta_new = np.log(total_wins / denom)
        theta_new -= theta_new[anchor]
        if np.max(np.abs(theta_new - theta)) < convergence:
            theta = theta_new
            break
        theta = theta_new
    return theta * 400.0 / math.log(10.0)


def inject_clone(agents, wins, games, dup_idx, clone_games):
    """Append an exact duplicate of agent ``dup_idx`` that splits ``clone_games`` with it."""
    n = len(agents)
    m = n + 1
    w2 = np.zeros((m, m))
    g2 = np.zeros((m, m))
    w2[:n, :n] = np.where(np.eye(n, dtype=bool), 0.0, np.nan_to_num(wins))
    g2[:n, :n] = games
    for j in range(n):
        if j == dup_idx:
            continue
        w2[n, j] = np.nan_to_num(wins)[dup_idx, j]
        w2[j, n] = np.nan_to_num(wins)[j, dup_idx]
        g2[n, j] = g2[j, n] = games[dup_idx, j]
    w2[n, dup_idx] = w2[dup_idx, n] = clone_games / 2.0
    g2[n, dup_idx] = g2[dup_idx, n] = clone_games
    return agents + [agents[dup_idx] + "_clone"], w2, g2


def max_shift(after, before, others):
    """Return the largest absolute change over the indices ``others``."""
    return float(np.max(np.abs(np.array([after[i] for i in others]) - before)))


def clone_cells(sidecar, cfg):
    """Return one record per duplicated agent: Nash and Elo shifts of the other agents."""
    agents, wins, games = precision_matrix(sidecar)
    n = len(agents)
    exact = exact_maxent_if_dominant(payoff(winrate_matrix(wins, games)))
    assert exact is not None, "no dominant class on the base panel"
    nash_margin = exact[1] * 2.0
    bt_args = (cfg["bt_iterations"], cfg["bt_convergence"])
    elo = bt_elo_fit(wins, games, agents.index(cfg["anchor"]), *bt_args)
    cells = []
    for dup in range(n):
        agents2, w2, g2 = inject_clone(agents, wins, games, dup, cfg["clone_games"])
        exact2 = exact_maxent_if_dominant(payoff(winrate_matrix(w2, g2)))
        assert exact2 is not None, f"clone cell {agents[dup]}: dominant class lost"
        p2, skill2 = exact2
        elo2 = bt_elo_fit(w2, g2, agents2.index(cfg["anchor"]), *bt_args)
        others = [i for i in range(n) if i != dup]
        if agents[dup] == cfg["anchor"]:
            # Re-anchor both fits at the fallback agent when the anchor is the duplicate.
            elo_base = elo - elo[agents.index(cfg["anchor_fallback"])]
            elo2 = elo2 - elo2[agents2.index(cfg["anchor_fallback"])]
        else:
            elo_base = elo
        cells.append(
            {
                "duplicated": agents[dup],
                "nash_max_shift_others_winrate_units": max_shift(
                    skill2 * 2.0, nash_margin[others], others
                ),
                "elo_max_shift_others": max_shift(elo2, elo_base[others], others),
                "nash_mass_split_on_pair": [float(p2[dup]), float(p2[n])],
            }
        )
    return cells
