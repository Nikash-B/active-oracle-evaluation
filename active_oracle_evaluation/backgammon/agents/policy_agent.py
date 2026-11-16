"""Policy agent over legal afterstates.

The agent enumerates the legal moves, encodes only their afterstates and ranks them by
the policy head's logits, so illegal moves are never scored.
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
from active_oracle_evaluation.backgammon.nn.policy_value_mlp import PolicyValueMLP


class PolicyAgent(Agent):
    """Selects moves from a masked legal-afterstate policy head."""

    name = "policy"

    def __init__(
        self,
        model: PolicyValueMLP,
        encoder: Encoder,
    ) -> None:
        """Wrap a policy-value model."""
        self.model = model
        self.encoder = encoder
        self.device = default_torch_device()
        self.model.to(self.device)

    def logits(self, moves: list[Move]) -> np.ndarray:
        """Policy logits for each legal move candidate."""
        feats = np.stack([self.encoder.encode(m.afterstate) for m in moves])
        tensor = torch.from_numpy(feats).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.policy_logits(tensor)
        return logits.cpu().numpy()

    def choose_move(self, game: Game) -> Move:
        """Pick the move with the highest policy logit."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        scores = self.logits(moves)
        return select_by_score(moves, scores)
