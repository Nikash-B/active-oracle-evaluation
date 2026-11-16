"""Game loop that plays two agents against each other and returns the result."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.dice import Roll
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Player

#: Upper bound on plies per game, so a non-terminating loop raises instead of hanging
#: (chosen for this study).
MAX_PLIES = 10_000


@dataclass(frozen=True)
class GameResult:
    """The outcome of a completed game.

    ``points`` is the positive magnitude (1/2/3). :meth:`points_for` gives the signed score
    for either side.
    """

    winner: Player
    points: int

    def points_for(self, player: Player) -> int:
        """Return the signed points from ``player``'s perspective (+win / -loss)."""
        return self.points if player is self.winner else -self.points


def play_game(
    white: Agent,
    black: Agent,
    seed: int | None = None,
    game: Game | None = None,
    rolls: Iterable[Roll] | None = None,
) -> GameResult:
    """Play one full game between ``white`` and ``black`` and return the result.

    Pass ``rolls`` to take the dice from an external stream instead of the game's RNG,
    so that two games can share one roll sequence (duplicate-dice evaluation). The stream
    must yield at least as many rolls as the game has plies.
    """
    g = game if game is not None else Game(rng=random.Random(seed))
    agents = {Player.WHITE: white, Player.BLACK: black}
    roll_iter = iter(rolls) if rolls is not None else None

    plies = 0
    while not g.is_over:
        if plies >= MAX_PLIES:  # pragma: no cover - defensive guard
            raise RuntimeError("game exceeded MAX_PLIES without terminating")
        if roll_iter is None:
            g.roll()
        else:
            g.set_dice(next(roll_iter))
        move = agents[g.turn].choose_move(g)
        g.apply(move)
        plies += 1

    winner = g.winner
    assert winner is not None  # guaranteed once is_over is True
    kind = g.result_kind
    assert kind is not None
    return GameResult(
        winner=winner,
        points=kind.value,
    )


__all__ = [
    "GameResult",
    "play_game",
]
