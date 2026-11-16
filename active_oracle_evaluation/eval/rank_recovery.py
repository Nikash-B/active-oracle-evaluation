"""Rank-recovery metrics for the label-efficiency experiment (pure numpy).

Given a ``losses`` matrix of per-agent oracle regret ``[n_agents, n_pos]`` and a selected subset
``S`` of position indices, these functions measure how well the ranking induced from ``S`` matches
the full-suite ranking. Selection itself lives in
:mod:`active_oracle_evaluation.eval.eval_select`.

* Ranking uses the plain subset mean PR (:func:`pr_subset`). Rank is scale-free, so the plain
  mean is valid under non-uniform selection.
* Absolute PR (for PR-MAE) uses the Horvitz-Thompson estimator (:func:`ht_pr`), which is unbiased
  for the full-suite mean under unequal inclusion probabilities.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from active_oracle_evaluation.eval.validity import kendall_tau

#: Performance rating = 500 x mean equity loss per decision (paper, section "Evaluation
#: protocol", paragraph "Rating").
PR_SCALE = 500.0


def pr_subset(losses: np.ndarray, S) -> np.ndarray:
    """Per-agent plain-mean PR over the column subset ``S`` of ``losses`` (used for ranking)."""
    return losses[:, np.asarray(S, dtype=int)].mean(axis=1) * PR_SCALE


def ht_pr(loss_row: np.ndarray, indices, inclusion_probs, n_pool: int) -> float:
    """Horvitz-Thompson PR estimate for one agent over a non-uniformly selected subset.

    Computes ``(500 / n_pool) * sum_{r in S} loss_r / pi_r``. With uniform ``pi_r = |S| / n_pool``
    it equals the plain subset mean. ``indices`` and ``inclusion_probs`` are aligned.
    """
    idx = np.asarray(indices, dtype=int)
    pi = np.asarray(inclusion_probs, dtype=float)
    return float((loss_row[idx] / pi).sum() / n_pool * PR_SCALE)


def ranking(pr_vec) -> list[int]:
    """Agent indices ordered best to worst (lowest PR first); ties break by agent index."""
    return [int(i) for i in np.argsort(np.asarray(pr_vec), kind="stable")]


def kendall_vs_full(order_sub: list[int], order_full: list[int]) -> float:
    """Kendall-τ between the subset-induced order and the full-suite order (+1 identical)."""
    return kendall_tau([str(i) for i in order_sub], [str(i) for i in order_full])


def pairwise_accuracy(order_sub: list[int], order_full: list[int]) -> float:
    """Fraction of agent pairs ordered the same way in the subset order as in the full order."""
    rank_sub = {a: i for i, a in enumerate(order_sub)}
    rank_full = {a: i for i, a in enumerate(order_full)}
    pairs = list(combinations(order_full, 2))
    if not pairs:
        return 1.0
    same = sum(1 for x, y in pairs if (rank_sub[x] < rank_sub[y]) == (rank_full[x] < rank_full[y]))
    return same / len(pairs)


def top1_recovered(order_sub: list[int], order_full: list[int]) -> bool:
    """True iff the subset's best agent is the full suite's best agent."""
    return bool(order_sub[0] == order_full[0])


def pr_mae(pr_sub, pr_full) -> float:
    """Mean absolute error (PR units) between the subset PR estimates and the full-suite PR."""
    return float(np.mean(np.abs(np.asarray(pr_sub) - np.asarray(pr_full))))


def ci_width(losses: np.ndarray, S, *, n_boot: int, seed: int) -> float:
    """Mean over agents of the 95% bootstrap PR interval width, resampling positions in ``S``."""
    S = np.asarray(S, dtype=int)
    sub = losses[:, S]  # [n_agents, m]
    m = sub.shape[1]
    if m == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, m, size=(n_boot, m))
    boot = sub[:, boot_idx].mean(axis=2) * PR_SCALE  # [n_agents, n_boot]
    lo = np.percentile(boot, 2.5, axis=1)
    hi = np.percentile(boot, 97.5, axis=1)
    return float(np.mean(hi - lo))
