"""Selective 3-ply extension of the 2-ply value search.

Negamax value of an afterstate ``a`` to the player who just moved into it::

    score(a, p) = +terminal_points(a)                          if a terminal
                = -V(a)                                         if p <= 1  (static)
                = -sum_d P(d) * max_reply score(reply, p-1)     otherwise

``max`` is the opponent (on roll at ``a``) choosing its best reply; the leading minus
flips the value back to the mover. At ``p == 2`` this equals ``twoply.two_ply_scores``.
"""

from __future__ import annotations

import numpy as np
import torch

from active_oracle_evaluation.backgammon.agents._selection import select_by_score
from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.dice import Roll, distinct_rolls
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.moves import legal_moves
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.search.evaluator import (
    ModelEvaluator,
    is_terminal,
    terminal_points,
)

# Boards expanded per slice at depth >= 2; bounds peak memory (chosen for this study).
# Values are per-board independent, so slicing does not change results.
_EXPAND_CHUNK = 64

# Settings of the "Network, selective 3-ply" agent (the paper's table of PR at four search
# depths, appendix "Validity of the oracle rating").
# Near-tie threshold in equity between 2-ply scores (chosen for this study).
NEAR_TIE_DELTA = 0.03
# Opponent replies per roll kept by static value in the deeper search (chosen for this study).
REPLY_K = 1
# Top 2-ply moves checked for a near tie (chosen for this study).
EXTEND_TOP = 4
# Depth of the re-scoring search: the "selective 3-ply" agent of the paper's table of PR at
# four search depths.
EXTEND_PLIES = 3

# One roll's replies: (probability, [flat index into the reply list, ...]).
RollEntries = list[tuple[float, list[int]]]


def _expand_replies(
    live: list[Board], live_idx: list[int], rolls: tuple[tuple[Roll, float], ...]
) -> tuple[list[Board], list[tuple[int, RollEntries]]]:
    """Flatten every reply afterstate of every live board, grouped by board and roll."""
    reply_boards: list[Board] = []
    groups: list[tuple[int, RollEntries]] = []  # (out_index, roll entries)
    for j, board in enumerate(live):
        roll_entries: RollEntries = []
        for roll, prob in rolls:
            idxs = []
            for reply in legal_moves(board, roll):
                idxs.append(len(reply_boards))
                reply_boards.append(reply.afterstate)
            roll_entries.append((prob, idxs))
        groups.append((live_idx[j], roll_entries))
    return reply_boards, groups


def _top_replies(
    reply_boards: list[Board],
    groups: list[tuple[int, RollEntries]],
    evaluator: ModelEvaluator,
    rolls: tuple[tuple[Roll, float], ...],
    reply_k: int,
) -> list[int]:
    """Flat indices of the top ``reply_k`` replies per roll by static value, in order."""
    static = _afterstate_values(reply_boards, evaluator, 1, rolls, reply_k)
    keep = np.zeros(len(reply_boards), dtype=bool)
    for _out_i, roll_entries in groups:
        for _prob, idxs in roll_entries:
            if len(idxs) <= reply_k:
                for t in idxs:
                    keep[t] = True
            else:
                for t in sorted(idxs, key=lambda t: static[t], reverse=True)[:reply_k]:
                    keep[t] = True
    return [t for t in range(len(reply_boards)) if keep[t]]


def _reduce_replies(
    out: np.ndarray,
    groups: list[tuple[int, RollEntries]],
    kept_vals: np.ndarray,
    remap: dict[int, int],
) -> None:
    """Write the negated roll-weighted best kept reply value into ``out`` per board."""
    for out_i, roll_entries in groups:
        expected = 0.0
        for prob, idxs in roll_entries:
            best = -np.inf
            for t in idxs:
                pos = remap.get(t)
                if pos is not None and kept_vals[pos] > best:
                    best = kept_vals[pos]
            expected += prob * best
        out[out_i] = -expected


def _afterstate_values(
    boards: list[Board],
    evaluator: ModelEvaluator,
    plies: int,
    rolls: tuple[tuple[Roll, float], ...],
    reply_k: int,
) -> np.ndarray:
    """Negamax value to the player who moved into each board, searched to depth ``plies``.

    All leaves at one recursion level (within one slice of ``_EXPAND_CHUNK`` boards) are
    evaluated in a single batched forward pass.
    """
    if plies >= 2 and len(boards) > _EXPAND_CHUNK:
        return np.concatenate(
            [
                _afterstate_values(boards[i : i + _EXPAND_CHUNK], evaluator, plies, rolls, reply_k)
                for i in range(0, len(boards), _EXPAND_CHUNK)
            ]
        )
    out = np.empty(len(boards), dtype=np.float32)
    live_idx: list[int] = []
    live: list[Board] = []
    for i, board in enumerate(boards):
        if is_terminal(board):
            out[i] = terminal_points(board)
        else:
            live_idx.append(i)
            live.append(board)
    if not live:
        return out
    if plies <= 1:
        vals = evaluator.evaluate(live)
        for j, i in enumerate(live_idx):
            out[i] = -float(vals[j])  # value to mover = -V(opponent's afterstate)
        return out

    # Expand one layer: the player on roll at each live board replies.
    reply_boards, groups = _expand_replies(live, live_idx, rolls)
    # At depth 2 the replies are the static leaves, so the reply filter would not change
    # the max and is skipped.
    if plies >= 3 and reply_boards:
        keep_flat = _top_replies(reply_boards, groups, evaluator, rolls, reply_k)
    else:
        keep_flat = list(range(len(reply_boards)))

    remap = {t: pos for pos, t in enumerate(keep_flat)}
    kept_vals = _afterstate_values(
        [reply_boards[t] for t in keep_flat], evaluator, plies - 1, rolls, reply_k
    )
    _reduce_replies(out, groups, kept_vals, remap)
    return out


class SelectiveDepthAgent(Agent):
    """2-ply agent that re-scores near-tied top moves with a deeper search.

    When at least two of the ``EXTEND_TOP`` best 2-ply equities lie within ``NEAR_TIE_DELTA``
    of the best, those afterstates are re-scored at ``EXTEND_PLIES`` (with ``REPLY_K``
    statically filtered opponent replies), and the move is the argmax over the re-scored set
    only, because values at different depths are not on the same scale.
    """

    def __init__(self, model: torch.nn.Module, encoder: Encoder) -> None:
        """Wrap the model in an evaluator."""
        self.evaluator = ModelEvaluator(model, encoder)
        self.name = f"sel{EXTEND_PLIES}ply_d{NEAR_TIE_DELTA}_r{REPLY_K}"

    def choose_move(self, game: Game) -> Move:
        """Pick a move by 2-ply search, extending near-tied top moves to deeper search."""
        from active_oracle_evaluation.backgammon.search.twoply import two_ply_scores

        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        scores = two_ply_scores(moves, self.evaluator)
        order = np.argsort(scores)[::-1][:EXTEND_TOP]
        tied = [int(i) for i in order if scores[int(order[0])] - scores[int(i)] <= NEAR_TIE_DELTA]
        if len(tied) >= 2:
            deep = _afterstate_values(
                [moves[i].afterstate for i in tied],
                self.evaluator,
                EXTEND_PLIES,
                distinct_rolls(),
                REPLY_K,
            )
            return moves[tied[int(np.argmax(deep))]]
        return select_by_score(moves, scores)
