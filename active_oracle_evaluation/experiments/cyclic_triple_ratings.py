"""Ratings fitted to a win-rate matrix: Bradley-Terry Elo, maxent Nash and alpha-rank."""

import math

import numpy as np


def bt_theta(P, games, iterations, tolerance):
    """Fit Bradley-Terry strengths (log scale, agent 0 fixed at 0) to a 3x3 win-rate matrix."""
    theta = np.zeros(3)
    for _ in range(iterations):
        s = np.exp(theta)
        denom = np.array([sum(games / (s[i] + s[j]) for j in range(3) if j != i) for i in range(3)])
        wins_tot = np.array([sum(P[i, j] * games for j in range(3) if j != i) for i in range(3)])
        theta_new = np.log(wins_tot / denom)
        theta_new -= theta_new[0]
        if np.max(np.abs(theta_new - theta)) < tolerance:
            theta = theta_new
            break
        theta = theta_new
    return theta


def maxent_nash_3(a, tolerance):
    """Return the exact Nash equilibrium of a 3-cycle with A[0,1], A[1,2], A[2,0] > 0.

    The unique (hence maxent) equilibrium is p* = (a12, a20, a01) / sum: each agent is
    weighted by the edge it does not play in. The result is checked via A p* = 0.
    """
    a01, a12, a20 = a[0, 1], a[1, 2], a[2, 0]
    assert a01 > 0 and a12 > 0 and a20 > 0, "not the expected 3-cycle orientation"
    p = np.array([a12, a20, a01])
    p = p / p.sum()
    assert float(np.max(np.abs(a @ p))) < tolerance
    return p


def alpha_rank(P, alphas):
    """Return single-population alpha-rank stationary distributions, keyed by str(alpha).

    Follows Omidshafiei et al. (2019): a Markov chain over pure strategies, with transition
    i->j proportional to a logistic in the payoff gap, swept over the selection intensity.
    """
    n = P.shape[0]
    eta = 1.0 / (n - 1)
    out = {}
    for alpha in alphas:
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    gap = (P[j, i] - 0.5) - (P[i, j] - 0.5)
                    T[i, j] = eta / (1.0 + math.exp(-alpha * gap))
            T[i, i] = 1.0 - T[i].sum()
        vals, vecs = np.linalg.eig(T.T)
        k = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, k])
        pi = np.abs(pi) / np.abs(pi).sum()
        out[str(alpha)] = [float(x) for x in pi]
    return out
