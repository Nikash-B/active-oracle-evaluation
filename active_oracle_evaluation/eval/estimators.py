"""Label-efficient estimators of a pool mean from oracle labels on a subset.

The pool holds ``N`` decisions and the estimand is an agent's full-pool mean regret; only
``K << N`` decisions receive an oracle label.

* :func:`classical_mean`: the sample mean of a uniform draw.
* :func:`horvitz_thompson` / :func:`hajek`: design-based, for any design with ``pi_r > 0``.
* :func:`lure`: the Active Testing estimator for sequential without-replacement sampling.
* :func:`ppi` / :func:`ppi_plus`: prediction-powered inference, which combines a cheap predictor
  on every pool item (here, regret at a shallower oracle depth) with labels on a subset.
* :func:`sequential_halving_bai`: fixed-budget best-agent identification.

All functions operate on 1-D (or agent-by-position) numpy arrays. The estimators back the
estimator-baseline comparison of the paper (appendix "Statistical protocol: derivations and
definitions", paragraph "Estimator baselines at matched label budget", and its estimator
table).
"""

from __future__ import annotations

import numpy as np

#: ``lam`` of vanilla PPI (Angelopoulos et al., 2023a).
PPI_LAMBDA = 1.0

#: Fewest labels from which :func:`ppi_plus_lambda` fits ``lam``; chosen for this study.
MIN_LABELS_FOR_LAMBDA = 3

# --------------------------------------------------------------------------- #
# Design-free
# --------------------------------------------------------------------------- #


def classical_mean(y_labeled) -> float:
    """Plain sample mean of the labeled values — unbiased only under uniform sampling."""
    y = np.asarray(y_labeled, dtype=float)
    return float(y.mean()) if y.size else 0.0


# --------------------------------------------------------------------------- #
# Design-based (require inclusion probabilities)
# --------------------------------------------------------------------------- #


def horvitz_thompson(y_labeled, inclusion_probs, n_pool: int) -> float:
    """``(1/N)·Σ_{r∈S} y_r/π_r`` — unbiased for the pool mean under any design with ``π_r > 0``.

    Steering the budget changes the variance, never the expectation. The cost is that a small
    ``π_r`` on a large ``y_r`` produces a huge summand, so HT is high-variance exactly where the
    design is most aggressive — which is what :func:`hajek` trades away.
    """
    y = np.asarray(y_labeled, dtype=float)
    pi = np.asarray(inclusion_probs, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.sum(y / pi) / n_pool)


def hajek(y_labeled, inclusion_probs) -> float:
    """``Σ y_r/π_r ÷ Σ 1/π_r`` — the ratio form: slightly biased, usually far lower variance.

    Self-normalizing, so it stays bounded when the realized sample under-represents the pool.
    """
    y = np.asarray(y_labeled, dtype=float)
    pi = np.asarray(inclusion_probs, dtype=float)
    if y.size == 0:
        return 0.0
    w = 1.0 / pi
    return float(np.sum(y * w) / np.sum(w))


def lure(y_ordered, q_at_draw, n_pool: int) -> float:
    """Active-Testing LURE estimator for sequential sampling **without replacement**.

    ``q_at_draw[m]`` is the proposal probability that draw ``m`` (0-based, of ``M`` total)
    assigned to the item it actually picked, renormalized over the items still remaining at that
    step; ``y_ordered`` is that item's label. With ``N`` the pool size and ``m`` 1-based, the
    weight is

        ``v_m = 1 + (N − M)/(N − m) · (1/((N − m + 1)·q_m) − 1)``

    and the estimate is ``mean(v_m · y_m)``. Unbiased for the pool mean for any proposal with full
    support. An exhaustive sample (``M >= N``) returns the exact pool mean.
    """
    y = np.asarray(y_ordered, dtype=float)
    q = np.asarray(q_at_draw, dtype=float)
    m_total = y.size
    if m_total == 0:
        return 0.0
    if m_total >= n_pool:  # exhaustive: nothing left to extrapolate
        return float(y.mean())
    m = np.arange(1, m_total + 1, dtype=float)
    v = 1.0 + (n_pool - m_total) / (n_pool - m) * (1.0 / ((n_pool - m + 1.0) * q) - 1.0)
    return float(np.mean(v * y))


# --------------------------------------------------------------------------- #
# Prediction-powered (require a predictor over the whole pool)
# --------------------------------------------------------------------------- #


def ppi(y_labeled, f_labeled, f_pool, *, lam: float = PPI_LAMBDA) -> float:
    """Prediction-powered mean: ``λ·mean(f over pool) + mean(y − λ·f over the labeled subset)``.

    The first term uses the cheap predictor on **every** pool item; the second is the *rectifier*
    that removes the predictor's bias using the expensive labels. ``λ = 1`` is vanilla PPI,
    ``λ = 0`` collapses to :func:`classical_mean`. Unbiased for the pool mean whenever the labeled
    subset is a uniform draw, for any ``λ`` and any predictor; a poor predictor only adds variance.
    This is the survey-sampling difference estimator.
    """
    y = np.asarray(y_labeled, dtype=float)
    f = np.asarray(f_labeled, dtype=float)
    fp = np.asarray(f_pool, dtype=float)
    if y.size == 0:
        return float(fp.mean()) if fp.size else 0.0
    return float(lam * fp.mean() + np.mean(y - lam * f))


def ppi_plus_lambda(y_labeled, f_labeled) -> float:
    """The variance-optimal ``λ`` for :func:`ppi`, estimated from the labeled subset.

    Minimizing ``Var(y − λf)`` gives ``λ* = Cov(y, f)/Var(f)``, the OLS slope of label on
    prediction. Returns 0 (i.e. fall back to the plain sample mean) when the predictor has no
    variance or there are too few labels to estimate a covariance. This is the power tuning of
    PPI++ (Angelopoulos et al., 2023b).
    """
    y = np.asarray(y_labeled, dtype=float)
    f = np.asarray(f_labeled, dtype=float)
    if y.size < MIN_LABELS_FOR_LAMBDA:
        return 0.0
    var_f = float(f.var(ddof=1))
    if var_f <= 0.0 or not np.isfinite(var_f):
        return 0.0
    lam = float(np.cov(y, f, ddof=1)[0, 1] / var_f)
    if not np.isfinite(lam):
        return 0.0
    return lam


def ppi_plus(y_labeled, f_labeled, f_pool) -> float:
    """PPI with the tuned ``λ`` of :func:`ppi_plus_lambda`.

    Because ``λ`` is fitted on the same labels it weights, this carries an ``O(1/K)`` bias.
    """
    lam = ppi_plus_lambda(y_labeled, f_labeled)
    return ppi(y_labeled, f_labeled, f_pool, lam=lam)


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


def sequential_without_replacement(weights, k: int, rng) -> tuple[list[int], list[float]]:
    """Draw ``k`` distinct items with probability ∝ ``weights``, renormalized after each draw.

    Returns ``(indices, q_at_draw)`` in draw order, where ``q_at_draw[m]`` is the probability the
    proposal gave the item it picked at that step, the input :func:`lure` needs.
    """
    w = np.asarray(weights, dtype=float).copy()
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive for full proposal support")
    n = w.size
    k = min(int(k), n)
    alive = np.ones(n, dtype=bool)
    idx: list[int] = []
    qs: list[float] = []
    for _ in range(k):
        wa = np.where(alive, w, 0.0)
        total = wa.sum()
        p = wa / total
        pick = int(rng.choice(n, p=p))
        idx.append(pick)
        qs.append(float(p[pick]))
        alive[pick] = False
    return idx, qs


# --------------------------------------------------------------------------- #
# Best-arm identification
# --------------------------------------------------------------------------- #


def sequential_halving_bai(
    y_all,
    importance_fn,
    k_total: int,
    rng,
) -> tuple[int, int]:
    """Fixed-budget best-arm identification by sequential halving over agents.

    Returns ``(winner_index, labels_used)``, the agent estimated to have the lowest pool mean.

    One label scores a position for every agent at once, so eliminating an agent saves no budget;
    elimination only redirects where later labels go. Each of ``ceil(log2(n_agents))`` rounds spends
    an equal share of ``k_total`` on positions drawn in proportion to ``importance_fn(survivors)``,
    never relabeling a position, then keeps the better half of the survivors by mean over all labels
    bought so far. ``importance_fn`` must use only free predictors or already-purchased labels.
    The schedule follows sequential halving (Karnin et al., 2013).
    """
    y = np.asarray(y_all, dtype=float)
    n_agents, n_pos = y.shape
    if n_agents < 2:
        return 0, 0
    rounds = max(1, int(np.ceil(np.log2(n_agents))))
    survivors = list(range(n_agents))
    labeled: list[int] = []
    labeled_set: set[int] = set()

    for r in range(rounds):
        # Split the budget evenly across rounds; the last round absorbs any remainder.
        take = k_total // rounds if r < rounds - 1 else k_total - len(labeled)
        take = max(0, min(take, n_pos - len(labeled)))
        if take:
            w = np.asarray(importance_fn(survivors), dtype=float).copy()
            w[list(labeled_set)] = 0.0  # never pay for the same position twice
            total = w.sum()
            if total <= 0:  # importance exhausted -> fall back to uniform over the unlabeled
                free = np.setdiff1d(
                    np.arange(n_pos), np.fromiter(labeled_set, int, len(labeled_set))
                )
                pick = rng.choice(free, size=min(take, free.size), replace=False)
            else:
                pick = rng.choice(n_pos, size=take, replace=False, p=w / total)
            labeled.extend(int(i) for i in np.atleast_1d(pick))
            labeled_set.update(labeled)
        if len(survivors) == 1 or not labeled:
            continue
        idx = np.fromiter(labeled_set, int, len(labeled_set))
        means = {a: float(y[a, idx].mean()) for a in survivors}
        survivors = sorted(survivors, key=lambda a: means[a])[: max(1, len(survivors) // 2)]

    if len(survivors) > 1 and labeled:
        idx = np.fromiter(labeled_set, int, len(labeled_set))
        survivors = [min(survivors, key=lambda a: float(y[a, idx].mean()))]
    return survivors[0], len(labeled_set)
