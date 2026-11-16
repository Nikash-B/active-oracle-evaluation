"""Tail-risk and per-phase statistics over per-agent oracle regret.

Two agents with nearly the same mean regret can differ in their worst-case
decisions and in which game phases they fail. This module computes those
statistics from a per-agent regret vector.

Definitions
-----------
* **regret** of a decision = ``best_equity − equity(chosen action)`` ≥ 0, in
  equity (the per-decision oracle loss; same quantity behind PR/EPM). Percentiles
  here are reported in those equity units. ``PR = mean(regret) * 500``.
* **catastrophic error** = a decision whose regret exceeds a threshold (default
  0.05 / 0.10 / 0.20 equity). The *catastrophic-error rate* is the fraction of
  decisions above the threshold.

The functions take a 1-D regret vector (numpy array or any sequence) and are
free of torch, oracle and matplotlib. Inputs come from the cached suite scores;
no agent is re-scored here.
"""

from __future__ import annotations

import numpy as np

#: Default bootstrap replicates (paper, appendix "Statistical protocol: derivations and
#: definitions", paragraph "Multiple comparisons and intervals": 2,000; the tail comparisons of
#: appendix "Phase diagnostics and search ablation" pass 10,000).
_N_BOOT = 2000

#: Default two-sided level of the percentile intervals (paper, appendix "Statistical
#: protocol: derivations and definitions", paragraph "Clustered bootstrap": 95% intervals).
_ALPHA = 0.05


def tail_stats(regret_vec) -> dict[str, float]:
    """Return the mean, median and 90th, 95th and 99th percentile regret (equity) of one agent.

    Percentiles use numpy's default linear interpolation. An empty vector yields
    all-zeros (a degenerate but well-defined summary).
    """
    r = np.asarray(regret_vec, dtype=float)
    if r.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    # The paper's tail-statistics table reports p95 and p99; the median and p90 are chosen for
    # this study.
    p50, p90, p95, p99 = np.percentile(r, [50, 90, 95, 99])
    return {
        "mean": float(r.mean()),
        "median": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def catastrophic_rate(regret_vec, thresholds: tuple[float, ...]) -> dict[float, float]:
    """Return the fraction of decisions whose regret strictly exceeds each threshold.

    Returns ``{threshold: rate}``; each rate is in ``[0, 1]`` and is monotonically
    non-increasing in the threshold (a higher bar can only exclude more decisions).
    """
    r = np.asarray(regret_vec, dtype=float)
    n = r.size
    if n == 0:
        return {t: 0.0 for t in thresholds}
    return {t: float((r > t).sum()) / n for t in thresholds}


def phase_breakdown(regret_vec, phases) -> dict[str, dict[str, float]]:
    """Return per-phase ``{mean, p95, n}`` regret, keyed by phase label.

    ``regret_vec`` and ``phases`` are index-aligned (one entry per decision). The
    counts partition the input exactly (``Σ n == len(regret_vec)``); a phase with
    no decisions is simply absent. ``phases`` are the :mod:`phase_labels`
    labels stored with the suite.
    """
    r = np.asarray(regret_vec, dtype=float)
    ph = np.asarray(phases)
    out: dict[str, dict[str, float]] = {}
    for label in np.unique(ph):
        sel = r[ph == label]
        out[str(label)] = {
            "mean": float(sel.mean()) if sel.size else 0.0,
            "p95": float(np.percentile(sel, 95)) if sel.size else 0.0,
            "n": int(sel.size),
        }
    return out


def game_bootstrap_indices(clusters, *, n_boot: int = _N_BOOT, seed: int = 0) -> list[np.ndarray]:
    """Return replicate index vectors for the game-clustered bootstrap.

    Decisions drawn from the same game share one dice sequence and trajectory, so
    they are positively correlated and resampling decisions would understate the
    variance. The resampling unit is therefore the game: draw ``n_games`` games
    with replacement and keep *every* decision inside each drawn game.

    ``clusters`` is the game label of each decision (index-aligned with the regret
    vector). Returns ``n_boot`` index arrays into the decision axis; replicate
    lengths vary when cluster sizes do. The same list should be **reused across
    agents, phases and statistics** so that differences between them are evaluated
    on the same resampled games and carry the right correlation.
    """
    cl = np.asarray(clusters)
    if cl.size == 0:
        return [np.empty(0, dtype=np.intp) for _ in range(n_boot)]
    _, inv = np.unique(cl, return_inverse=True)
    n_games = int(inv.max()) + 1
    members = [np.flatnonzero(inv == g) for g in range(n_games)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_games, size=(n_boot, n_games))
    return [np.concatenate([members[g] for g in row]) for row in draws]


def phase_mean_replicates(regret_vec, phases, indices, phase_order) -> np.ndarray:
    """Return bootstrap replicates of the per-phase mean regret, shape ``(n_boot, n_phases)``.

    Column ``j`` is the mean regret over the decisions of ``phase_order[j]`` inside
    the resampled games. A replicate that happens to contain no decision of some
    phase yields ``nan`` in that column (possible for a rare phase; percentiles
    downstream should use nan-aware reducers). Computed with ``bincount`` so one
    pass over a replicate produces every phase at once.
    """
    r = np.asarray(regret_vec, dtype=float)
    ph = np.asarray(phases)
    code_of = {p: j for j, p in enumerate(phase_order)}
    # -1 marks a phase we were not asked about; those decisions are dropped.
    codes = np.array([code_of.get(str(p), -1) for p in ph], dtype=np.intp)
    n_ph = len(phase_order)
    out = np.full((len(indices), n_ph), np.nan)
    for b, idx in enumerate(indices):
        c = codes[idx]
        keep = c >= 0
        c, vals = c[keep], r[idx][keep]
        counts = np.bincount(c, minlength=n_ph).astype(float)
        sums = np.bincount(c, weights=vals, minlength=n_ph)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def percentile_ci(samples, *, alpha: float = _ALPHA) -> tuple[float, float]:
    """Two-sided ``1 - alpha`` percentile interval of a bootstrap sample (nan-aware)."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap_p_two_sided(samples) -> float:
    """Two-sided bootstrap p-value for ``statistic != 0``.

    Twice the smaller tail mass either side of zero, floored at ``1/n_boot`` — the
    resolution limit of a finite bootstrap, below which "p = 0" would overstate what
    the replicates can show.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    p = 2.0 * min(float((s > 0).mean()), float((s < 0).mean()))
    return min(1.0, max(p, 1.0 / s.size))


def holm_adjust(p_values) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    Sort ascending, scale the k-th by ``m - k`` and enforce monotonicity.
    """
    p = [float(x) for x in p_values]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def bootstrap_p95_ci(
    regret_vec, *, n_boot: int = _N_BOOT, alpha: float = _ALPHA, seed: int = 0
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the 95th-percentile regret (resampling decisions).

    Returns ``(low, high)`` of a ``1 − alpha`` two-sided interval;
    an empty vector returns ``(0.0, 0.0)``.
    """
    r = np.asarray(regret_vec, dtype=float)
    n = r.size
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = r[rng.integers(0, n, size=n)]
        stats[b] = np.percentile(sample, 95)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
