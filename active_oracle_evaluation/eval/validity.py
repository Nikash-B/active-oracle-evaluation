"""Statistics comparing the oracle Performance Rating (PR) with head-to-head results.

* Rank agreement: Kendall tau and Spearman rho between the PR order and the Elo order.
* Calibration: the slope and R^2 of head-to-head log-odds regressed on the PR gap, with
  pair-bootstrap, agent-bootstrap and two-way cluster-robust uncertainty.
* Robustness: worst-case exploitability over a round-robin panel.

Pure functions over plain data (no torch or gnubg).
"""

from __future__ import annotations

import math
from itertools import combinations, permutations

import numpy as np

# 8!: enumerate all permutations up to eight names (chosen for this study).
EXACT_PERMUTATION_LIMIT = 40320
# Win rates are clipped to [clip, 1 - clip] before the logit (chosen for this study).
WIN_RATE_CLIP = 0.02
# Sampled permutations when exact enumeration is too large (chosen for this study).
PERMUTATION_SAMPLES = 20000
# Tolerance for counting a permuted |tau| as equal to the observed one (chosen for this study).
TAU_TIE_EPS = 1e-12
# Bootstrap replicates for the slope intervals (chosen for this study).
BOOTSTRAP_REPLICATES = 10000
# Percentiles of a 95% interval, the interval level used throughout the paper.
INTERVAL_PERCENTILES = [2.5, 97.5]


def tied_values(scores: dict[str, float]) -> list[tuple[str, str]]:
    """Return the pairs of names whose scores tie exactly, to check before sorting for kendall_tau.

    :func:`kendall_tau` takes orderings, so a tie is already broken by ``sorted()`` when it runs.
    """
    items = list(scores.items())
    return [(a, b) for (a, x), (b, y) in combinations(items, 2) if x == y]


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Return Kendall tau-a between two strict orderings (+1 identical, -1 reversed)."""
    rank_a = {name: i for i, name in enumerate(order_a)}
    rank_b = {name: i for i, name in enumerate(order_b)}
    names = [name for name in order_a if name in rank_b]
    concordant = discordant = 0
    for x, y in combinations(names, 2):
        sign = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
        if sign > 0:
            concordant += 1
        elif sign < 0:
            discordant += 1
    n_pairs = len(names) * (len(names) - 1) // 2
    return (concordant - discordant) / n_pairs if n_pairs else 0.0


def _pair_points(
    pr: dict[str, float], wins, games, names: list[str], *, clip: float = WIN_RATE_CLIP
):
    """Return the ``(x, y, i, j)`` calibration points, one per ordered pair with games played.

    ``x = PR_j - PR_i`` (positive when i is stronger, since lower PR is better) and
    ``y = logit(P(i beats j))`` with the win rate clipped to ``[clip, 1 - clip]``.
    """
    n = len(names)
    xs, ys, ii, jj = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i == j or games[i][j] == 0:
                continue
            p = min(max(wins[i][j] / games[i][j], clip), 1 - clip)
            xs.append(pr[names[j]] - pr[names[i]])
            ys.append(math.log(p / (1 - p)))
            ii.append(i)
            jj.append(j)
    return np.asarray(xs), np.asarray(ys), np.asarray(ii), np.asarray(jj)


def pr_gap_winprob_fit(
    pr: dict[str, float], wins, games, names: list[str], *, clip: float = WIN_RATE_CLIP
):
    """Regress head-to-head log-odds on the PR gap; return ``(slope, r2, n_pairs)``."""
    x, y, _, _ = _pair_points(pr, wins, games, names, clip=clip)
    if x.size < 2:
        return 0.0, 0.0, int(x.size)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), r2, int(x.size)


def spearman_rho(order_a: list[str], order_b: list[str]) -> float:
    """Return the Spearman rank correlation between two orderings of the same names."""
    rank_b = {name: i for i, name in enumerate(order_b)}
    names = [name for name in order_a if name in rank_b]
    n = len(names)
    if n < 2:
        return 0.0
    rank_a = {name: i for i, name in enumerate(names)}
    d2 = sum((rank_a[name] - rank_b[name]) ** 2 for name in names)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kendall_tau_pvalue(order_a: list[str], order_b: list[str], *, seed: int = 0) -> float:
    """Return the two-sided permutation p-value for Kendall tau between two orderings.

    Permutes ``order_b`` and returns the fraction of permutations with ``|tau|`` at least the
    observed one: exact enumeration up to eight names, otherwise ``PERMUTATION_SAMPLES`` seeded
    samples (with the observed assignment counted once).
    """
    rank_b = {name: i for i, name in enumerate(order_b)}
    names = [name for name in order_a if name in rank_b]
    n = len(names)
    if n < 2:
        return 1.0
    observed = abs(kendall_tau(order_a, order_b))
    eps = TAU_TIE_EPS
    if math.factorial(n) <= EXACT_PERMUTATION_LIMIT:
        count = total = 0
        for perm in permutations(names):
            total += 1
            if abs(kendall_tau(order_a, list(perm))) >= observed - eps:
                count += 1
        return count / total
    rng = np.random.default_rng(seed)
    arr = np.asarray(names, dtype=object)
    count = 1
    for _ in range(PERMUTATION_SAMPLES):
        perm = list(rng.permutation(arr))
        if abs(kendall_tau(order_a, perm)) >= observed - eps:
            count += 1
    return count / (PERMUTATION_SAMPLES + 1)


def bootstrap_slope_ci(
    pr: dict[str, float],
    wins,
    games,
    names: list[str],
    *,
    seed: int = 0,
):
    """Return a 95% bootstrap CI ``(lo, hi)`` for the calibration slope, resampling pairs."""
    x, y, _, _ = _pair_points(pr, wins, games, names)
    if x.size < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = x.size
    slopes = np.empty(BOOTSTRAP_REPLICATES)
    for b in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, m, size=m)
        xb, yb = x[idx], y[idx]
        if np.ptp(xb) == 0:  # all resampled x equal: slope undefined
            slopes[b] = np.nan
            continue
        slopes[b] = np.polyfit(xb, yb, 1)[0]
    slopes = slopes[~np.isnan(slopes)]
    lo, hi = np.percentile(slopes, INTERVAL_PERCENTILES)
    return float(lo), float(hi)


def agent_bootstrap_slope_ci(
    pr: dict[str, float],
    wins,
    games,
    names: list[str],
    *,
    seed: int = 0,
):
    """Return a 95% CI ``(lo, hi, n_used)`` for the calibration slope, resampling agents.

    Each replicate draws ``n`` agents with replacement and refits on the pairs among the drawn
    multiset (a cluster bootstrap for dyadic data). ``n_used`` counts the fittable replicates.
    """
    n = len(names)
    if n < 2:
        return 0.0, 0.0, 0
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, n, size=n)
        xs, ys = [], []
        for a in draw:
            for b in draw:
                if a == b or games[a][b] == 0:
                    continue
                p = min(max(wins[a][b] / games[a][b], WIN_RATE_CLIP), 1 - WIN_RATE_CLIP)
                xs.append(pr[names[b]] - pr[names[a]])
                ys.append(math.log(p / (1 - p)))
        if len(xs) < 2:
            continue
        x, y = np.asarray(xs), np.asarray(ys)
        if np.ptp(x) == 0:
            continue
        slopes.append(np.polyfit(x, y, 1)[0])
    if len(slopes) < 2:
        return 0.0, 0.0, len(slopes)
    lo, hi = np.percentile(slopes, INTERVAL_PERCENTILES)
    return float(lo), float(hi), len(slopes)


def dyadic_robust_slope_se(pr: dict[str, float], wins, games, names: list[str]) -> float:
    """Return the two-way cluster-robust standard error of the calibration slope.

    Each point belongs to the clusters of both agents in its pair, so the variance is
    ``V_i + V_j - V_ij`` (Cameron, Gelbach and Miller, 2011). If that is not positive, the
    larger one-way variance is used.
    """
    x, y, ii, jj = _pair_points(pr, wins, games, names)
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    xc = x - x.mean()
    sxx = float((xc**2).sum())

    def meat(labels) -> float:
        """Sum over clusters of the squared within-cluster score ``sum(x_c * resid)``."""
        return float(
            sum(float((xc[labels == c] * resid[labels == c]).sum()) ** 2 for c in np.unique(labels))
        )

    pair_id = np.array([f"{min(a, b)}-{max(a, b)}" for a, b in zip(ii, jj)])
    variance = (meat(ii) + meat(jj) - meat(pair_id)) / sxx**2
    if variance <= 0:
        variance = max(meat(ii), meat(jj)) / sxx**2
    return float(np.sqrt(variance))
