"""Board-level value evaluation for search.

Search needs to score *boards*, not moves, and it needs to do so for both value
heads without caring which one it has. :class:`ModelEvaluator` wraps a trained
network (a scalar-head
:class:`~active_oracle_evaluation.backgammon.nn.mlp.ValueMLP` or an outcome-head
:class:`~active_oracle_evaluation.backgammon.nn.outcome_mlp.OutcomeMLP`) and exposes a
single batched call:

    evaluate(boards) -> equity for the player to move at each board

This is exactly the quantity move selection already uses (``V(afterstate)`` is
the opponent's equity; the value to the mover of reaching it is ``-V``). The
encoder canonicalizes internally, so absolute boards may be passed directly.

Terminal positions are handled by :func:`terminal_points` rather than the
network: a board that a move just reached and won is worth its exact point value.
"""

from __future__ import annotations

import numpy as np
import torch

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.game import classify_result
from active_oracle_evaluation.backgammon.core.types import Player
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.nn.device import amp_autocast, default_torch_device


def terminal_points(board: Board) -> float:
    """Signed points for the side that just moved into a terminal ``board``.

    A checker play can only ever end the game in the *mover's* favour (you bear
    off your own checkers), so the side with all checkers borne off is the winner
    and this returns ``+1`` / ``+2`` / ``+3`` (single / gammon / backgammon).
    Raises if ``board`` is not terminal.
    """
    if board.has_won(Player.WHITE):
        winner = Player.WHITE
    elif board.has_won(Player.BLACK):
        winner = Player.BLACK
    else:
        raise ValueError("terminal_points called on a non-terminal board")
    return float(classify_result(board, winner).value)


def is_terminal(board: Board) -> bool:
    """True iff either side has borne off all 15 checkers."""
    return board.has_won(Player.WHITE) or board.has_won(Player.BLACK)


class ModelEvaluator:
    """Batched, head-agnostic ``board -> equity (for the player to move)``.

    ``ValueMLP`` returns its scalar output directly; ``OutcomeMLP`` returns the
    expected point-equity ``w . p`` of its outcome distribution. A
    per-evaluator cache (keyed by the immutable :class:`Board`) avoids recomputing
    leaves that recur across the branches of a single 2-ply search.
    """

    def __init__(self, model: torch.nn.Module, encoder: Encoder) -> None:
        """Move ``model`` to the default device in eval mode and set up the cache."""
        self.model = model
        self.encoder = encoder
        self.device = default_torch_device()
        self.model.to(self.device)
        self.model.eval()
        self._equity_fn = getattr(model, "equity", None)
        self._cache: dict[Board, float] = {}

    def _forward(self, boards: list[Board]) -> np.ndarray:
        """Run the network on ``boards`` and return mover equities as fp32."""
        feats = np.stack([self.encoder.encode(b) for b in boards])
        tensor = torch.from_numpy(feats).to(self.device)
        with torch.no_grad(), amp_autocast(self.device):  # bf16 inference on CUDA
            if callable(self._equity_fn):
                vals = self._equity_fn(tensor)
            else:
                vals = self.model(tensor).squeeze(-1)
        # NumPy has no bfloat16, so cast to fp32 before converting.
        return vals.float().cpu().numpy()

    # One 2-ply search touches ~30k boards; the bound keeps a long-lived agent's
    # cache from growing without limit across many games (chosen for this study).
    _CACHE_MAX = 200_000

    def evaluate(self, boards: list[Board]) -> np.ndarray:
        """Equity for the player to move at each board, shape ``(len(boards),)``."""
        if not boards:
            return np.empty(0, dtype=np.float32)
        if len(self._cache) > self._CACHE_MAX:
            self._cache = {}

        out = np.empty(len(boards), dtype=np.float32)
        todo_idx: list[int] = []
        todo_boards: list[Board] = []
        for i, b in enumerate(boards):
            cached = self._cache.get(b)
            if cached is None:
                todo_idx.append(i)
                todo_boards.append(b)
            else:
                out[i] = cached
        if todo_boards:
            vals = self._forward(todo_boards)
            for j, i in enumerate(todo_idx):
                out[i] = vals[j]
                self._cache[todo_boards[j]] = float(vals[j])
        return out
