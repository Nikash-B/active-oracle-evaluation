"""Selection designs for a budget-``K`` evaluation subset of the oracle decision suite.

Each strategy maps the pool to first-order inclusion probabilities ``pi_r`` with
``sum_r pi_r = min(K, N)`` and ``0 < pi_r <= 1``. The subset is drawn by Madow systematic PPS
(:func:`_pps_sample`), so position ``r`` is included with probability exactly ``pi_r`` and the draw
has exactly ``K`` positions. The Horvitz-Thompson estimate
:func:`~active_oracle_evaluation.eval.rank_recovery.ht_pr` is then unbiased for the
full-pool PR under any design with ``pi_r > 0``; steering changes only its variance. A uniform floor
(``eps``) keeps ``pi_r > 0`` for the steered designs.

Strategies (:data:`STRATEGIES`):

* ``random``: uniform inclusion ``pi_r = K / N``.
* ``phase_balanced``: equal budget across game phases (water-filling), uniform within a phase.
* ``disagreement``: inclusion proportional to panel disagreement (Simpson diversity of the agents'
  chosen actions), with a uniform floor.
* ``active_rank``: phase-balanced, and within each phase inclusion proportional to the summed
  pairwise regret gap ``sum_(i, j) |loss_i - loss_j|`` over agent pairs.

The inputs are plain numpy arrays (``losses``, ``choices``, ``phases`` from
``data/decision_suites/full_suite_scores_0ply.npz``). The designs follow the paper's
active-selection protocol (appendix "Experimental protocol", paragraph "Frozen decision
suite and active selection").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

#: The four selection rules of the paper (appendix "Experimental protocol", paragraph
#: "Frozen decision suite and active selection").
STRATEGIES: tuple[str, ...] = ("random", "phase_balanced", "disagreement", "active_rank")

#: Uniform-probability floor mixed into the steered designs so every position keeps ``π_r > 0``,
#: which keeps the Horvitz–Thompson estimate unbiased for the whole pool (paper, appendix
#: "Experimental protocol", paragraph "Frozen decision suite and active selection").
DEFAULT_EPS = 0.1

#: Numerical tolerances of the capped-PPS and water-filling loops; chosen for this study.
_PIN_TOL = 1e-12
_FILL_TOL = 1e-9


@dataclass
class Selection:
    """A drawn evaluation subset and the inclusion probabilities needed for unbiased PR."""

    indices: list[int]  # selected positions, into the pool order (cache / eligible-records order)
    inclusion_probs: list[float]  # π_r for r in ``indices``, aligned
    pi_full: list[float] | None = None  # π over the whole pool (len n_pos), optional


# --------------------------------------------------------------------------- #
# Inclusion probabilities and PPS draw
# --------------------------------------------------------------------------- #


def inclusion_probs(weights, k: int) -> np.ndarray:
    """First-order inclusion probabilities ``π ∝ weights``, capped at 1, with ``Σ π = min(k, N)``.

    Iterative "probabilities proportional to size, capped at 1": scale the free weights to fill the
    remaining budget; any that reach 1 are pinned and the budget recomputed over the rest.
    """
    return _capped_pps(np.asarray(weights, dtype=float), float(min(max(k, 0), np.size(weights))))


def _capped_pps(w: np.ndarray, budget: float) -> np.ndarray:
    """Inclusion probabilities ``∝ w`` summing to ``budget`` (possibly fractional), each ≤ 1."""
    n = w.size
    pi = np.zeros(n)
    if n == 0 or budget <= 0:
        return pi
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if budget >= n:
        return np.ones(n)
    free = w > 0
    remaining = float(budget)
    while True:
        s = w[free].sum()
        if s <= 0 or not free.any():
            break
        tentative = remaining * w[free] / s
        over = tentative >= 1.0
        if not over.any():
            pi[free] = tentative
            break
        idx_free = np.flatnonzero(free)
        pin = idx_free[over]
        pi[pin] = 1.0
        remaining -= len(pin)
        free[pin] = False
        if remaining <= _PIN_TOL:
            break
    return pi


def floor_mix(importance, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Mix a uniform floor into an importance vector: ``(1-eps)·norm(importance) + eps·uniform``.

    Every position gets a strictly positive weight (so its inclusion probability is > 0 and HT
    stays unbiased), while ``1-eps`` of the mass stays on the high-importance positions.
    """
    s = np.clip(np.asarray(importance, dtype=float), 0.0, None)
    n = s.size
    if n == 0:
        return np.zeros(0)
    total = s.sum()
    norm = s / total if total > 0 else np.full(n, 1.0 / n)
    return (1.0 - eps) * norm + eps * (1.0 / n)


def _pps_sample(pi, *, seed: int) -> list[int]:
    """Draw a fixed-size subset by Madow systematic PPS; returns indices in selection order.

    Lays the inclusion probabilities end to end (seed-randomized order), then takes
    ``k = round(Σ π)`` points spaced one apart from a random start in ``[0, 1)``. Each unit spans an
    interval ``≤ 1``, so it captures at most one point: exactly ``k`` distinct units, each with
    first-order probability ``π_r`` (Madow, 1949).
    """
    pi = np.asarray(pi, dtype=float)
    n = pi.size
    k = int(round(float(pi.sum())))
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    if k >= n:
        return rng.permutation(n).tolist()
    order = rng.permutation(n)
    cum = np.cumsum(pi[order])
    start = float(rng.uniform(0.0, 1.0))
    points = start + np.arange(k)
    pos = np.clip(np.searchsorted(cum, points, side="left"), 0, n - 1)
    seen: set[int] = set()
    out: list[int] = []
    for s in order[pos].tolist():
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Per-position scores
# --------------------------------------------------------------------------- #


def disagreement_importance(choices) -> np.ndarray:
    """Panel disagreement per position: Simpson diversity ``1 − Σ_a p_a²`` of the chosen actions.

    ``p_a`` = fraction of agents choosing action ``a`` at the position; 0 when the panel agrees,
    approaching ``1 − 1/|panel|`` when every agent picks a different action. Action indices are
    nominal, so this uses diversity (not range), after Simpson (1949). ``choices`` is
    ``int[n_agents, n_pos]``.
    """
    choices = np.asarray(choices)
    n_agents, n_pos = choices.shape
    out = np.zeros(n_pos)
    for r in range(n_pos):
        _, counts = np.unique(choices[:, r], return_counts=True)
        p = counts / n_agents
        out[r] = 1.0 - float(np.sum(p * p))
    return out


def pairwise_regret_importance(losses) -> np.ndarray:
    """Per-position separation signal ``Σ_{(i,j)}|regret_i − regret_j|`` over all agent pairs.

    Large where a position drives agents' PR estimates apart. ``losses`` is
    ``float[n_agents, n_pos]``.
    """
    losses = np.asarray(losses, dtype=float)
    n_agents, n_pos = losses.shape
    out = np.zeros(n_pos)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            out += np.abs(losses[i] - losses[j])
    return out


# --------------------------------------------------------------------------- #
# Phase-balanced (water-filling) inclusion probabilities
# --------------------------------------------------------------------------- #


def _cell_budgets(sizes: dict, k: float) -> dict:
    """Water-fill a budget ``k`` across phases with ``sizes`` (phase → count), capped at size."""
    budgets = {c: 0.0 for c in sizes}
    remaining = float(min(k, sum(sizes.values())))
    active = {c for c, m in sizes.items() if m > 0}
    while remaining > _FILL_TOL and active:
        share = remaining / len(active)
        newly_full = []
        for c in list(active):
            room = sizes[c] - budgets[c]
            if share >= room - _PIN_TOL:
                budgets[c] = float(sizes[c])
                remaining -= room
                newly_full.append(c)
        if newly_full:
            for c in newly_full:
                active.discard(c)
        else:
            for c in active:
                budgets[c] += share
            remaining = 0.0
    return budgets


def phase_balanced_pi(phases, k: int, *, importance=None, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Inclusion probabilities with an equal budget per phase, uniform or importance-weighted."""
    phases = list(phases)
    n = len(phases)
    pi = np.zeros(n)
    if n == 0:
        return pi
    idx_by_phase: dict = defaultdict(list)
    for r, ph in enumerate(phases):
        idx_by_phase[ph].append(r)
    sizes = {ph: len(ix) for ph, ix in idx_by_phase.items()}
    budgets = _cell_budgets(sizes, min(k, n))
    imp = None if importance is None else np.asarray(importance, dtype=float)
    for ph, ix in idx_by_phase.items():
        ix_arr = np.asarray(ix, dtype=int)
        b = budgets[ph]
        if b <= 0:
            continue
        if imp is None:
            pi[ix_arr] = b / sizes[ph]
        else:
            pi[ix_arr] = _capped_pps(floor_mix(imp[ix_arr], eps), b)
    return pi


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def build_pi(
    strategy: str,
    k: int,
    *,
    losses,
    choices,
    phases,
    eps: float = DEFAULT_EPS,
    importance_losses=None,
    importance: str | None = None,
) -> np.ndarray:
    """Inclusion probabilities ``π`` over the whole pool (Σ=min(k,N), 0<π≤1) for one ``strategy``.

    ``active_rank`` needs a steering source, and raises if neither of these is given
    (and also raises if both are given):

    * ``importance_losses=<array>``: a signal available pool-wide (for example a cheaper oracle or a
      noisy annotator). This is an achievable design.
    * ``importance="skyline"``: steer on ``losses`` itself. This is an upper bound on what steering
      can achieve, not an operating point.

    ``random``, ``phase_balanced`` and ``disagreement`` need no oracle labels and ignore both
    arguments.
    """
    n = len(phases)
    if strategy == "random":
        return inclusion_probs(np.ones(n), k)
    if strategy == "phase_balanced":
        return phase_balanced_pi(phases, k, importance=None, eps=eps)
    if strategy == "disagreement":
        return inclusion_probs(floor_mix(disagreement_importance(choices), eps), k)
    if strategy == "active_rank":
        if importance is not None and importance != "skyline":
            raise ValueError(f"importance must be None or 'skyline', not {importance!r}")
        if importance_losses is not None and importance == "skyline":
            raise ValueError("pass importance_losses=<array> or importance='skyline', not both")
        if importance_losses is None and importance is None:
            raise ValueError(
                "active_rank needs a steering source: importance_losses=<array> for an achievable "
                "design, or importance='skyline' to steer on the estimand itself (an upper bound)."
            )
        imp = pairwise_regret_importance(losses if importance == "skyline" else importance_losses)
        return phase_balanced_pi(phases, k, importance=imp, eps=eps)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def select(
    strategy: str,
    k: int,
    *,
    losses,
    choices,
    phases,
    seed: int,
    importance_losses=None,
    importance: str | None = None,
) -> Selection:
    """Build ``strategy``'s inclusion probabilities and draw a budget-``K`` subset.

    ``losses`` / ``choices`` are ``[n_agents, n_pos]`` arrays (per-agent regret and chosen action
    index); ``phases`` is a length-``n_pos`` sequence of phase labels. ``importance_losses`` /
    ``importance`` name ``active_rank``'s steering source (see :func:`build_pi`).
    Returns a :class:`Selection` with ``indices``, aligned ``inclusion_probs``, and the full-pool
    ``pi_full``.
    """
    pi = build_pi(
        strategy,
        k,
        losses=losses,
        choices=choices,
        phases=phases,
        importance_losses=importance_losses,
        importance=importance,
    )
    idx = _pps_sample(pi, seed=seed)
    return Selection(
        indices=idx,
        inclusion_probs=[float(pi[i]) for i in idx],
        pi_full=pi.tolist(),
    )
