"""Uniform-random agent: the weakest reference agent on the rating scale."""

from __future__ import annotations

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move


class RandomAgent(Agent):
    """Picks a uniformly random legal move.

    It uses the game's RNG, so a seeded
    :func:`~active_oracle_evaluation.backgammon.api.play_game` is reproducible without seeding each agent.
    """

    name = "random"

    def choose_move(self, game: Game) -> Move:
        """Return a uniformly random legal move for the side to play."""
        return game.rng.choice(game.legal_moves())
