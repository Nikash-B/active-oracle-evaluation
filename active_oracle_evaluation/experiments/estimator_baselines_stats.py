"""Standard errors, rank agreement and matched-budget helpers of the estimator baselines."""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.eval.score_cache import ScoreCache

ESTIMATORS = (
    "classical",
    "ht_active",
    "hajek_active",
    "lure_active",
    "ppi",
    "ppi_plus",
    "ht_oracle_design",
)


def _se_srs(resid: np.ndarray, n_pool: int) -> float:
    """Standard error of a simple-random-sample mean, with the finite-population correction.

    ``resid`` is the labels for the sample mean, or ``y - lambda * f`` for the PPI forms.
    """
    k = resid.size
    if k < 2:
        return float("nan")
    return float(np.sqrt(resid.var(ddof=1) / k * max(0.0, 1.0 - k / n_pool)))


def _se_ht(y_sel: np.ndarray, pi_sel: np.ndarray, n_pool: int) -> float:
    """HT standard error, independent-draw form ``(1/N^2) * sum (1 - pi) / pi^2 * y^2``."""
    if y_sel.size < 2:
        return float("nan")
    var = np.sum((1.0 - pi_sel) / pi_sel**2 * y_sel**2) / n_pool**2
    return float(np.sqrt(max(var, 0.0)))


def _se_lure(y_sel: np.ndarray, q: np.ndarray, n_pool: int) -> float:
    """SE of the LURE mean from the spread of its own weighted terms."""
    m = y_sel.size
    if m < 2 or m >= n_pool:
        return float("nan")
    steps = np.arange(1, m + 1, dtype=float)
    weights = 1.0 + (n_pool - m) / (n_pool - steps) * (1.0 / ((n_pool - steps + 1.0) * q) - 1.0)
    return float(np.sqrt((weights * y_sel).var(ddof=1) / m))


def _tau(pred: np.ndarray, truth: np.ndarray) -> float:
    """Kendall tau between the estimated and true agent orderings (both: lower PR = better)."""
    n = len(truth)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = (pred[i] - pred[j]) * (truth[i] - truth[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    n_pairs = n * (n - 1) // 2
    return (concordant - discordant) / n_pairs if n_pairs else 0.0


def labels_to_match(rows: list[dict], baseline: str, target_k: int) -> dict[str, float]:
    """For each estimator, the budget at which it first reaches ``baseline``'s RMSE at ``target_k``.

    Interpolates linearly in log-K between measured points; ``nan`` when the estimator never
    reaches it inside the measured range.
    """
    ref = next(r["rmse"] for r in rows if r["estimator"] == baseline and r["k"] == target_k)
    out: dict[str, float] = {}
    for e in {r["estimator"] for r in rows}:
        pts = sorted(((r["k"], r["rmse"]) for r in rows if r["estimator"] == e), key=lambda t: t[0])
        hit = float("nan")
        for (k0, r0), (k1, r1) in zip(pts, pts[1:]):
            if (r0 - ref) * (r1 - ref) <= 0 and r0 != r1:  # bracket the crossing
                w = (r0 - ref) / (r0 - r1)
                hit = float(np.exp(np.log(k0) + w * (np.log(k1) - np.log(k0))))
                break
        if pts and pts[0][1] <= ref:  # already there at the smallest budget measured
            hit = float(pts[0][0])
        out[e] = hit
    return out


def predictor_correlation_range(
    cheap: ScoreCache, exp: ScoreCache, names: list[str], untrained
) -> tuple[float, float]:
    """Min and max Pearson correlation of 0-ply with 2-ply regret over the trained agents."""
    trained = [a for a in names if a not in untrained]
    corr = [float(np.corrcoef(cheap.losses[a], exp.losses[a])[0, 1]) for a in trained]
    if not corr:
        return float("nan"), float("nan")
    return min(corr), max(corr)
