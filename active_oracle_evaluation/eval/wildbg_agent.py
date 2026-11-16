"""wildbg reference agent: plays the top move of a locally running wildbg server.

wildbg (see References in README.md) is an external Rust/ONNX engine. This
adapter queries wildbg's own binary over its HTTP API, so it re-implements no evaluation. On
every decision the chosen play must map onto exactly one of our legal afterstates, else
``WildbgParityError`` is raised.

Frame mapping: wildbg's absolute pips ``p1..p24`` (positive = player to move "x", moving
24 -> 1) equal our canonical mover frame ``canon.point(p)``. ``p25`` is x's bar (positive) and
``p0`` is o's bar (negative). Bear-off hops go ``to: 0``; bar entries come ``from: 25``.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move, Player

#: Per-request timeout in seconds for the wildbg HTTP API (chosen for this study).
_TIMEOUT = 30.0


class WildbgParityError(RuntimeError):
    """wildbg's chosen play did not map onto exactly one legal afterstate."""


def _mover_frame(canon: Board) -> tuple[tuple[int, ...], int, int]:
    """Canonical (mover-to-act) board -> (p1..p24 signed counts, x_bar, o_bar)."""
    vec = tuple(canon.point(p) for p in range(1, 25))
    return vec, canon.bar(Player.WHITE), canon.bar(Player.BLACK)


def _afterstate_mover_frame(after: Board) -> tuple[tuple[int, ...], int, int]:
    """Afterstate (opponent-to-act) -> the same frame as the player who just moved.

    In ``after.canonical()`` White is the *opponent* of x, so x's checkers are the
    Black side at mirrored numbering: ``x_frame[p] = -canon.point(25 - p)``.
    """
    canon = after.canonical()
    vec = tuple(-canon.point(25 - p) for p in range(1, 25))
    return vec, canon.bar(Player.BLACK), canon.bar(Player.WHITE)


def _apply_play(
    frame: tuple[tuple[int, ...], int, int], play: list[dict]
) -> tuple[tuple[int, ...], int, int]:
    """Apply wildbg's per-die hops in the x frame (hits send o checkers to o's bar)."""
    vec = list(frame[0])
    x_bar, o_bar = frame[1], frame[2]
    for hop in play:
        src, dst = hop["from"], hop["to"]
        if src == 25:
            x_bar -= 1
        else:
            vec[src - 1] -= 1
        if dst != 0:
            if vec[dst - 1] == -1:  # blot hit
                vec[dst - 1] = 0
                o_bar += 1
            vec[dst - 1] += 1
    return tuple(vec), x_bar, o_bar


class WildbgAgent(Agent):
    """Greedy agent playing wildbg's own top-ranked move (0-ply composite evaluator)."""

    name = "wildbg"

    def __init__(self, base_url: str) -> None:
        """Store the server URL."""
        self._base_url = base_url.rstrip("/")

    def _query(self, params: dict[str, int]) -> list[dict]:
        """Return wildbg's ranked moves for one position and roll."""
        url = f"{self._base_url}/move?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())["moves"]

    def choose_move(self, game: Game) -> Move:
        """Play wildbg's top-ranked move, checked to match exactly one legal move."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        die1, die2 = game.dice
        vec, x_bar, o_bar = _mover_frame(game.board.canonical())
        params = {"die1": die1, "die2": die2}
        for p in range(1, 25):
            if vec[p - 1]:
                params[f"p{p}"] = vec[p - 1]
        if x_bar:
            params["p25"] = x_bar
        if o_bar:
            params["p0"] = -o_bar
        ranked = self._query(params)
        if not ranked:
            raise WildbgParityError(f"wildbg returned no moves for {params}")
        target = _apply_play((vec, x_bar, o_bar), ranked[0]["play"])
        matches = [m for m in moves if _afterstate_mover_frame(m.afterstate) == target]
        if len(matches) != 1:
            raise WildbgParityError(
                f"wildbg play {ranked[0]['play']} mapped to {len(matches)} legal "
                f"afterstates (params {params})"
            )
        return matches[0]
