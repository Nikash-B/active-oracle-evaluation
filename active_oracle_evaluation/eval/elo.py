"""Round-robin matches and Bradley-Terry Elo over a set of agents.

One round-robin gives the win and game matrices that the Bradley-Terry fit reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.api import play_game
from active_oracle_evaluation.backgammon.core.types import Player

# Seed strides that give every game and every pair its own seed (chosen for this study).
GAME_SEED_STRIDE = 1009
PAIR_SEED_STRIDE = 131
# Virtual games against a strength-1 opponent in the Bradley-Terry fit (chosen for this study).
BRADLEY_TERRY_PRIOR_GAMES = 2.0
# MM iteration cap and convergence tolerances (chosen for this study).
BRADLEY_TERRY_ITERATIONS = 500
BRADLEY_TERRY_RTOL = 1e-10
BRADLEY_TERRY_ATOL = 1e-12
# Standard Elo scale: 400 points per factor of ten in strength.
ELO_SCALE = 400.0


@dataclass(frozen=True)
class RoundRobin:
    """Pairwise results among named competitors."""

    names: list[str]
    games: np.ndarray  # (n, n) games played between i and j
    wins: np.ndarray  # (n, n) i's wins vs j


def play_pair(a: Agent, b: Agent, games: int, seed: int) -> tuple[int, int]:
    """Play ``games`` between ``a`` and ``b``, alternating colors.

    Returns ``(a_wins, b_wins)``.
    """
    a_wins = 0
    b_wins = 0
    for game_index in range(games):
        a_is_white = game_index % 2 == 0
        white, black = (a, b) if a_is_white else (b, a)
        result = play_game(white, black, seed=seed * GAME_SEED_STRIDE + game_index)
        a_seat = Player.WHITE if a_is_white else Player.BLACK
        if result.winner is a_seat:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins


def round_robin(competitors: dict[str, Agent], games_per_pair: int, seed: int = 0) -> RoundRobin:
    """Play every pair of competitors; record games and wins."""
    names = list(competitors)
    n = len(names)
    games = np.zeros((n, n))
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair_seed = seed + i * PAIR_SEED_STRIDE + j
            a_wins, b_wins = play_pair(
                competitors[names[i]], competitors[names[j]], games_per_pair, pair_seed
            )
            games[i, j] = games[j, i] = games_per_pair
            wins[i, j], wins[j, i] = a_wins, b_wins
    return RoundRobin(names, games, wins)


def bradley_terry_elo(
    rr: RoundRobin,
    *,
    anchor_name: str | None = None,
    anchor_elo: float = 0.0,
    prior: float = BRADLEY_TERRY_PRIOR_GAMES,
    iterations: int = BRADLEY_TERRY_ITERATIONS,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths (Bradley and Terry, 1952) by MM iteration and convert to Elo.

    A small ``prior`` (virtual games vs a strength-1 opponent, split evenly) keeps
    winless/unbeaten competitors bounded. The fit is deterministic. If ``anchor_name`` is
    given its Elo is pinned to ``anchor_elo``; otherwise ratings are centered at 0.
    """
    n = len(rr.names)
    total_wins = rr.wins.sum(axis=1) + prior / 2.0  # + virtual wins
    strength = np.ones(n)
    for _ in range(iterations):
        updated = np.empty(n)
        for i in range(n):
            denom = prior / (strength[i] + 1.0)  # virtual strength-1 opponent
            for j in range(n):
                if i != j and rr.games[i, j] > 0:
                    denom += rr.games[i, j] / (strength[i] + strength[j])
            updated[i] = total_wins[i] / denom if denom > 0 else strength[i]
        updated /= updated.mean()
        if np.allclose(updated, strength, rtol=BRADLEY_TERRY_RTOL, atol=BRADLEY_TERRY_ATOL):
            strength = updated
            break
        strength = updated
    elo = ELO_SCALE * np.log10(strength)
    if anchor_name is not None and anchor_name in rr.names:
        elo = elo - elo[rr.names.index(anchor_name)] + anchor_elo
    else:
        elo = elo - elo.mean()
    return {name: float(elo[k]) for k, name in enumerate(rr.names)}
