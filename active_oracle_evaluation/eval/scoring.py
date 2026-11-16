"""Per-decision oracle scoring: EPM / PR.

Turns duplicate-dice :class:`DecisionLog` rows into a :class:`ScoreSummary`. For each unforced
checker decision the oracle gives the equity (to the mover) of every legal afterstate, and the
agent loses ``best_equity - chosen_equity`` (>= 0). EPM is the mean of those losses in
millipoints; PR is the mean loss on the engines' Performance Rating scale. The oracle is any
object exposing ``equity(board) -> float`` (see
:class:`~active_oracle_evaluation.eval.oracle.GnubgOracle`).
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field

from active_oracle_evaluation.eval.duplicate import DecisionLog

#: Performance Rating per point of mean equity loss per decision (paper, appendix "Statistical
#: protocol: derivations and definitions": 500).
_PR_PER_POINT = 500.0

#: EPM reports mean equity loss in millipoints (gnubg convention).
_MILLIPOINTS_PER_POINT = 1000.0

#: Worker processes for the parallel scorers (chosen for this study).
_WORKERS = 4


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate scoring of one or more agents' logged decisions."""

    n_checker: int
    epm: float  # mean checker equity-loss per unforced move, millipoints
    pr: float  # Performance Rating over all scored decisions
    by_agent: dict[str, "ScoreSummary"] = field(default_factory=dict)


def _mover_equity(after, oracle, cache: dict) -> float:
    """Oracle equity *to the mover* of reaching ``after`` (cached; afterstate is opp-to-move)."""
    value = cache.get(after)
    if value is None:
        value = -oracle.equity(after)
        cache[after] = value
    return value


def _checker_loss(log, oracle, cache: dict) -> float:
    """Equity the mover gave up on one checker decision: best afterstate minus the one taken."""
    equities = [_mover_equity(a, oracle, cache) for a in log.legal_afterstates]
    chosen = _mover_equity(log.chosen_afterstate, oracle, cache)
    return max(equities) - chosen


def _score(logs, oracle, cache: dict, *, by_agent: bool) -> ScoreSummary:
    """Aggregate checker losses over ``logs``, optionally split per agent."""
    checker_losses: list[float] = []
    per_agent: dict[str, list[DecisionLog]] = defaultdict(list)

    for log in logs:
        per_agent[log.agent].append(log)
        checker_losses.append(_checker_loss(log, oracle, cache))

    n_checker = len(checker_losses)
    epm = (sum(checker_losses) / n_checker * _MILLIPOINTS_PER_POINT) if n_checker else 0.0
    pr = (sum(checker_losses) / n_checker * _PR_PER_POINT) if n_checker else 0.0

    by: dict[str, ScoreSummary] = {}
    if by_agent:
        for agent, agent_logs in per_agent.items():
            by[agent] = _score(agent_logs, oracle, cache, by_agent=False)
    return ScoreSummary(n_checker, epm, pr, by)


def score_decisions(logs: list[DecisionLog], oracle) -> ScoreSummary:
    """Score logged decisions against ``oracle``. Returns overall + per-agent summaries."""
    return _score(logs, oracle, cache={}, by_agent=True)


# Parallel scoring: chunk the checker decisions across spawned workers, each with its own
# GnubgOracle. Workers return only small per-decision results. The "spawn" start method is
# chosen for this study.


def _score_chunk(packed):
    """Worker for :func:`score_decisions_parallel`: ``(agent, loss)`` per checker decision."""
    logs_chunk, oracle_cfg = packed
    from active_oracle_evaluation.eval.oracle import GnubgOracle  # lazy, per worker

    oracle = GnubgOracle(oracle_cfg)
    cache: dict = {}
    losses: list[tuple[str, float]] = []
    for log in logs_chunk:
        losses.append((log.agent, _checker_loss(log, oracle, cache)))
    return losses


def _detail_chunk(packed):
    """Worker for :func:`decision_losses_parallel`: ``(index, detail)`` per scored row."""
    idx_logs, oracle_cfg = packed
    from active_oracle_evaluation.eval.oracle import GnubgOracle  # lazy, per worker

    oracle = GnubgOracle(oracle_cfg)
    cache: dict = {}
    out: list[tuple[int, dict]] = []
    for i, log in idx_logs:
        equities = [_mover_equity(a, oracle, cache) for a in log.legal_afterstates]
        chosen = _mover_equity(log.chosen_afterstate, oracle, cache)
        chosen_index = log.legal_afterstates.index(log.chosen_afterstate)
        out.append(
            (
                i,
                {
                    "loss": max(equities) - chosen,
                    "chosen_index": chosen_index,
                    "legal_equities": tuple(equities),
                },
            )
        )
    return out


def decision_losses_parallel(logs: list[DecisionLog], oracle_cfg, workers: int = _WORKERS):
    """Per-decision detail, index-aligned with ``logs``.

    Each entry is ``{"loss", "chosen_index", "legal_equities"}``. Keeping the row index lets
    callers resample whole duplicate-dice trials (``DecisionLog.trial``).
    """
    detail: list[dict | None] = [None] * len(logs)
    indexed = list(enumerate(logs))
    if workers <= 1:
        for i, entry in _detail_chunk((indexed, oracle_cfg)):
            detail[i] = entry
        return detail
    chunks = [indexed[i::workers] for i in range(workers)]  # round-robin keeps chunks balanced
    packed = [(chunk, oracle_cfg) for chunk in chunks if chunk]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(packed)) as pool:
        for chunk_result in pool.map(_detail_chunk, packed):
            for i, entry in chunk_result:
                detail[i] = entry
    return detail


def _summary_from_losses(losses: list[tuple[str, float]], *, by_agent: bool) -> ScoreSummary:
    """Build a checker-only :class:`ScoreSummary` from ``(agent, loss)`` pairs."""
    values = [loss for _, loss in losses]
    n = len(values)
    epm = (sum(values) / n * _MILLIPOINTS_PER_POINT) if n else 0.0
    pr = (sum(values) / n * _PR_PER_POINT) if n else 0.0
    by: dict[str, ScoreSummary] = {}
    if by_agent:
        per: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for agent, loss in losses:
            per[agent].append((agent, loss))
        for agent, agent_losses in per.items():
            by[agent] = _summary_from_losses(agent_losses, by_agent=False)
    return ScoreSummary(n, epm, pr, by)


def score_decisions_parallel(
    logs: list[DecisionLog], oracle_cfg, workers: int = _WORKERS
) -> ScoreSummary:
    """Like :func:`score_decisions` but fans the oracle calls across ``workers`` processes.

    Takes an :class:`~active_oracle_evaluation.eval.oracle.OracleConfig`, since each worker
    builds its own
    ``GnubgOracle``.
    """
    if workers <= 1:
        from active_oracle_evaluation.eval.oracle import GnubgOracle

        return score_decisions(logs, GnubgOracle(oracle_cfg))
    chunks = [logs[i::workers] for i in range(workers)]  # round-robin keeps chunks balanced
    packed = [(chunk, oracle_cfg) for chunk in chunks if chunk]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(packed)) as pool:
        results = pool.map(_score_chunk, packed)
    all_losses = [pair for chunk in results for pair in chunk]
    return _summary_from_losses(all_losses, by_agent=True)
