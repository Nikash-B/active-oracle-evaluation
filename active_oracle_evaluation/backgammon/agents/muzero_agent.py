"""Agent that plays with a learned MuZero model by one-step latent lookahead.

The representation ``h`` encodes the board. For each legal afterstate, the dynamics ``g``
gives the next latent and reward, and the prediction ``f`` values that latent. The score to
the mover is ``reward - value(next_latent)``; winning afterstates use their exact points.
"""

from __future__ import annotations

import numpy as np
import torch

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.search.evaluator import is_terminal, terminal_points


class MuZeroAgent(Agent):
    """Plays by one-step lookahead with a learned MuZero value-equivalent model."""

    name = "muzero"

    def __init__(self, model: object, encoder: Encoder) -> None:
        """Store the model, the encoder and the model's device."""
        self.model = model
        self.encoder = encoder
        self.device = next(model.parameters()).device  # type: ignore[attr-defined]

    def choose_move(self, game: Game) -> Move:
        """Return the legal move with the highest one-step model score."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]

        board = torch.from_numpy(np.stack([self.encoder.encode(game.board)])).to(self.device)
        afterstates = torch.from_numpy(
            np.stack([self.encoder.encode(m.afterstate) for m in moves])
        ).to(self.device)

        self.model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            latent = self.model.represent(board).repeat(len(moves), 1)  # type: ignore[attr-defined]
            dynamics_step = self.model.dynamics_step  # type: ignore[attr-defined]
            next_latent, reward = dynamics_step(latent, afterstates)
            next_value = self.model.value(next_latent)  # type: ignore[attr-defined]

        # Negamax: the opponent moves after the afterstate.
        scores = reward.cpu().numpy() - next_value.cpu().numpy()
        for i, move in enumerate(moves):
            if is_terminal(move.afterstate):
                scores[i] = terminal_points(move.afterstate)
        return moves[int(np.argmax(scores))]
