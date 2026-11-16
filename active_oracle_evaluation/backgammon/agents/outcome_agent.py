"""Outcome-network agent: afterstate selection by expected point-equity.

Identical selection logic to
:class:`~active_oracle_evaluation.backgammon.agents.td_agent.TDAgent`, but the
per-afterstate score is the **expected point-equity** derived from the
six-class outcome head rather than a single scalar output:

    EV(afterstate) = sum_c CLASS_POINTS[c] * P(outcome = c | afterstate)

A candidate afterstate has the opponent on roll, so ``EV(afterstate)`` is the
opponent's equity; the value to the player who just moved is ``-EV(afterstate)``
and selection maximises that. Selection is shared with ``TDAgent`` via
:func:`active_oracle_evaluation.backgammon.agents._selection.select_by_score`.
"""

from __future__ import annotations

import numpy as np
import torch

from active_oracle_evaluation.backgammon.agents._selection import select_by_score
from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.nn.device import default_torch_device
from active_oracle_evaluation.backgammon.nn.outcome_mlp import OutcomeMLP


class OutcomeAgent(Agent):
    """Selects moves by the expected point-equity of a six-class outcome network."""

    name = "outcome"

    def __init__(
        self,
        model: OutcomeMLP,
        encoder: Encoder,
    ) -> None:
        """Wrap ``model`` and ``encoder``."""
        self.model = model
        self.encoder = encoder
        self.device = default_torch_device()
        self.model.to(self.device)

    def equities(self, moves: list[Move]) -> np.ndarray:
        """Expected point-equity ``EV(afterstate)`` for each move (opponent's view)."""
        feats = np.stack([self.encoder.encode(m.afterstate) for m in moves])
        tensor = torch.from_numpy(feats).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ev = self.model.equity(tensor)
        return ev.cpu().numpy()

    def choose_move(self, game: Game) -> Move:
        """Pick the legal move whose afterstate has the highest equity for the mover."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]

        # Value to the mover of reaching each afterstate is -EV(afterstate).
        scores = -self.equities(moves)
        return select_by_score(moves, scores)
