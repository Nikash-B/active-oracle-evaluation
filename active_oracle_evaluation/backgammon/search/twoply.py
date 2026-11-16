"""2-ply expectimax search.

From a candidate afterstate ``a`` (opponent on roll) we look one full ply deeper:
average, over all 21 distinct dice rolls weighted by probability, the opponent's
best *static* reply, then negate to bring the value back to the mover:

    two_ply_value_to_mover(a) = +terminal_points(a)            if a is terminal
                               -sum_d' P(d') * best_opp(a, d')  otherwise

    best_opp(a, d') = max over opponent replies b of value_to_opp(b)
    value_to_opp(b) = +terminal_points(b)  if b is terminal (opponent won)
                      -V(b)                 otherwise

The minus signs are the negamax bookkeeping: a board's value always belongs to
whoever is on roll there, and flips each ply.

Two things keep the (candidates) x 21 x (opponent replies) expansion tractable:

* a top-k 1-ply pre-filter (``FILTER_K``): only the most promising mover moves
  are searched deeply; the rest keep their 1-ply score;
* batched leaf evaluation: every non-terminal leaf across the whole tree is
  evaluated in a single network forward pass.
"""

from __future__ import annotations

import numpy as np

from active_oracle_evaluation.backgammon.core.dice import distinct_rolls
from active_oracle_evaluation.backgammon.core.moves import legal_moves
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.search.evaluator import (
    ModelEvaluator,
    is_terminal,
    terminal_points,
)
from active_oracle_evaluation.backgammon.search.onestep import one_ply_scores

# Mover moves kept by the 1-ply pre-filter of the "Network, 2-ply search" and "Network, selective
# 3-ply" agents (the paper's table of PR at four search depths, appendix "Validity of the
# oracle rating"); chosen for this study.
FILTER_K = 8

# A reply entry is (_TERM, points) or (_LEAF, index into leaf_boards).
_TERM = 0
_LEAF = 1


def two_ply_scores(moves: list[Move], evaluator: ModelEvaluator) -> np.ndarray:
    """Value-to-mover for each move; the top ``FILTER_K`` are searched to 2-ply.

    Moves outside the top-k keep their 1-ply score, so the returned array is a
    valid basis for selection over all moves.
    """
    rolls = distinct_rolls()
    scores = one_ply_scores(moves, evaluator)
    num_moves = len(moves)
    if num_moves <= 1:
        return scores

    num_candidates = min(max(1, FILTER_K), num_moves)
    candidate_idx = np.argpartition(scores, -num_candidates)[-num_candidates:]

    # Build the expansion plan and gather every non-terminal leaf board.
    leaf_boards = []
    plans: list[tuple] = []  # aligned with candidate_idx
    for candidate in candidate_idx:
        afterstate = moves[int(candidate)].afterstate
        if is_terminal(afterstate):
            plans.append(("term", terminal_points(afterstate)))
            continue
        roll_plans = []  # list of (prob, [(kind, payload), ...])
        for roll, prob in rolls:
            entries = []
            for reply in legal_moves(afterstate, roll):
                reply_board = reply.afterstate
                if is_terminal(reply_board):
                    entries.append((_TERM, terminal_points(reply_board)))
                else:
                    entries.append((_LEAF, len(leaf_boards)))
                    leaf_boards.append(reply_board)
            roll_plans.append((prob, entries))
        plans.append(("expand", roll_plans))

    # One batched forward pass over all leaves.
    leaf_values = evaluator.evaluate(leaf_boards) if leaf_boards else np.empty(0, np.float32)

    # Reduce the tree with the negamax/expectimax recursion above.
    for candidate, plan in zip(candidate_idx, plans):
        if plan[0] == "term":
            scores[int(candidate)] = plan[1]
            continue
        expected_opp = 0.0
        for prob, entries in plan[1]:
            best_opp = -np.inf
            for kind, payload in entries:
                # Value to the opponent of reaching this reply board.
                value_to_opp = payload if kind == _TERM else -leaf_values[payload]
                if value_to_opp > best_opp:
                    best_opp = value_to_opp
            expected_opp += prob * best_opp
        scores[int(candidate)] = -expected_opp
    return scores
