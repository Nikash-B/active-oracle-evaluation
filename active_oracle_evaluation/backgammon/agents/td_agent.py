"""Value-network agent: greedy afterstate value play.

The agent picks the move whose afterstate has the highest value to the mover:

    m* = argmax_{m in legal}  value_to_mover(afterstate(m))

Value convention
----------------
The network scores a board as the **equity for the player to move in that
board**. A candidate afterstate has the *opponent* on roll, so its network value
``V(afterstate)`` is the opponent's equity; the value to the player who just
moved is therefore ``-V(afterstate)``. Selection maximises that, i.e. it picks
the afterstate the opponent likes least.
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


class TDAgent(Agent):
    """Selects moves by a learned afterstate value network."""

    name = "td"

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: Encoder,
    ) -> None:
        """Wrap a value network and encoder."""
        self.model = model
        self.encoder = encoder
        self.device = default_torch_device()
        self.model.to(self.device)

    def values(self, moves: list[Move]) -> np.ndarray:
        """Network equity ``V(afterstate)`` for each move (opponent's perspective)."""
        feats = np.stack([self.encoder.encode(m.afterstate) for m in moves])
        tensor = torch.from_numpy(feats).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(tensor).squeeze(-1)
        return out.cpu().numpy()

    def choose_move(self, game: Game) -> Move:
        """Return the best-scoring legal move."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]

        # Value to the mover of reaching each afterstate is -V(afterstate).
        scores = -self.values(moves)
        return select_by_score(moves, scores)
