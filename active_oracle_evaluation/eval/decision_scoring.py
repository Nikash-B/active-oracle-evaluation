"""Score agents on a reloaded oracle decision suite.

Selection, efficiency and tail-risk analyses share this module, so per-decision regret is
computed one way everywhere. Given labeled :class:`DecisionRecord` objects (each carrying the
oracle's mover equity for every legal afterstate), it recomputes an agent's per-decision
**equity loss** (regret >= 0) from the frozen labels. No oracle is called: this is the on-disk
analogue of :func:`active_oracle_evaluation.eval.suite.score_agent_on_suite`.

The downstream contract: callers first take an :func:`eligible_records` view (labeled,
unforced), then build :func:`score_matrices` over that *fixed* record
list so every agent's vector is index-aligned (paired across agents and positions).
"""

from __future__ import annotations

import random

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.eval.decision_suite import (
    DecisionRecord,
    afterstates_of,
    dice_of,
    serialize_board,
    to_board,
)

#: PR is 500 times the mean regret (paper, appendix "Statistical protocol: derivations and
#: definitions"; gnubg Performance Rating scale).
_PR_PER_EQUITY = 500.0


def eligible_records(records: list[DecisionRecord]) -> list[DecisionRecord]:
    """Filter to the scorable decisions, in input order (the shared, paired record view).

    This is the checker-PR view of the paper (appendix "Experimental protocol"): oracle-labeled,
    unforced decisions (a forced move carries no skill signal). All agents must be scored on the
    *same* returned list so their loss and choice vectors align position for position.
    """
    return [r for r in records if not r.forced_move and r.oracle_equities is not None]


def chosen_index(agent: Agent, record: DecisionRecord, *, seed: object = 0) -> int:
    """Index into ``record.legal_actions`` of the afterstate ``agent`` chooses at this position.

    Reconstructs the board + roll, asks the agent for its move, and matches the resulting afterstate
    to the record's legal actions by board equality (falling back to serialized-board equality).
    """
    afters = afterstates_of(record)
    if len(afters) <= 1:
        return 0
    # Seed the game RNG so agents that break ties at random choose reproducibly
    # (seed format chosen for this study).
    game = Game(board=to_board(record), rng=random.Random(f"{seed}"))
    game.set_dice(dice_of(record))
    move = agent.choose_move(game)
    try:
        return afters.index(move.afterstate)
    except ValueError:
        target = serialize_board(move.afterstate)
        for i, a in enumerate(record.legal_actions):
            if a == target:
                return i
        # The afterstate is not among the legal actions: fall back to action 0.
        return 0


def record_loss(agent: Agent, record: DecisionRecord, *, seed: object = 0) -> float:
    """Oracle equity loss (regret ≥ 0) of ``agent``'s choice. Requires a labeled record."""
    if record.oracle_equities is None or record.best_equity is None:
        raise ValueError("record is unlabeled (no oracle_equities)")
    idx = chosen_index(agent, record, seed=seed)
    return float(record.best_equity - record.oracle_equities[idx])


def score_agent_on_records(
    agent: Agent, records: list[DecisionRecord], *, seed: int = 0
) -> list[float]:
    """Per-record regret vector of ``agent`` over an :func:`eligible_records` view."""
    # Seeds are keyed by the index into ``records``, so a subset passed as its own list re-indexes.
    # The f"{seed}:{i}" format is chosen for this study.
    return [record_loss(agent, r, seed=f"{seed}:{i}") for i, r in enumerate(records)]


def pr_of_losses(losses: list[float]) -> float:
    """gnubg-scale Performance Rating from a regret vector (mean loss × 500)."""
    return (sum(losses) / len(losses) * _PR_PER_EQUITY) if losses else 0.0


def _choices(agent: Agent, records: list[DecisionRecord], seed: int) -> list[int]:
    """Chosen legal-action index per record, seeding record ``i`` with ``f"{seed}:{i}"``."""
    return [chosen_index(agent, r, seed=f"{seed}:{i}") for i, r in enumerate(records)]


def score_matrices(
    agents: dict[str, Agent], records: list[DecisionRecord], *, seed: int = 0
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Return regret and chosen-index matrices per agent, both from one choice per record.

    Both matrices describe the same choice, and every record is keyed like
    :func:`score_agent_on_records` (``f"{seed}:{i}"``, ``i`` the index into ``records``), so a loss
    row equals ``score_agent_on_records(agent, records, seed=seed)``.
    """
    losses: dict[str, list[float]] = {}
    choices: dict[str, list[int]] = {}
    for r in records:
        if r.oracle_equities is None or r.best_equity is None:
            raise ValueError(f"record {r.position_id} is unlabeled (no oracle_equities)")
    for name, agent in agents.items():
        idx = _choices(agent, records, seed)
        choices[name] = idx
        losses[name] = [float(r.best_equity - r.oracle_equities[j]) for r, j in zip(records, idx)]
    return losses, choices
