"""Duplicate-dice paired evaluation.

To compare agent ``a`` against agent ``b`` while controlling for dice luck and the
first-move/seat advantage, each *trial* is played twice on the same roll stream with
colors swapped:

    game 1:  a = White,  b = Black
    game 2:  b = White,  a = Black     (identical dice)

and we record ``a``'s mean points per game over the pair,
``(a_as_white + a_as_black) / 2``. Because both games saw the same dice, the luck that
helped whoever held White in game 1 helps whoever holds White in game 2, so it partly
cancels in the per-trial figure. ``PairedResult`` reports both the paired and the unpaired SD,
so the variance reduction is measured rather than assumed.

Dice come from a :class:`~active_oracle_evaluation.eval.dice_bank.DiceBank` via ``play_game(rolls=...)``,
which is independent of the agent-facing RNG, so stochastic agents never consume the
dice stream and the two games stay dice-identical.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.api import play_game
from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move, Player
from active_oracle_evaluation.eval.dice_bank import DiceBank


@dataclass(frozen=True)
class DecisionLog:
    """One decision, with everything needed to score it later against an oracle.

    ``agent`` is the stable label ("a"/"b") of who decided, regardless of seat.
    """

    agent: str
    board: Board  # pre-move position, mover to act
    dice: tuple[int, int]
    chosen_afterstate: Board
    legal_afterstates: tuple[Board, ...]
    #: Duplicate-dice trial. The trial is the bootstrap cluster label, since both games of a
    #: trial share one roll stream.
    trial: int


class _LoggingAgent(Agent):
    """Delegates to an inner agent and records each unforced decision into ``sink``."""

    def __init__(self, inner: Agent, label: str, sink: list[DecisionLog]) -> None:
        """Wrap ``inner`` under the stable ``label`` and append its decisions to ``sink``.

        :func:`play_duplicate_pair` sets ``trial`` before each duplicate-dice trial.
        """
        self.inner = inner
        self.label = label
        self.sink = sink
        self.trial = 0

    def choose_move(self, game: Game) -> Move:
        """Return the inner agent's move, logging the decision when it is not forced."""
        moves = game.legal_moves()
        chosen = self.inner.choose_move(game)
        if len(moves) > 1:
            self.sink.append(
                DecisionLog(
                    agent=self.label,
                    board=game.board,
                    dice=tuple(game.dice),
                    chosen_afterstate=chosen.afterstate,
                    legal_afterstates=tuple(m.afterstate for m in moves),
                    trial=self.trial,
                )
            )
        return chosen


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a duplicate-dice comparison of ``a`` vs ``b``.

    All point figures are from ``a``'s perspective, in points per game.
    """

    mean_diff: float  # a's mean points/game vs b (dice- & seat-controlled)
    paired_sd: float  # SD across trials of the per-trial per-game figure
    paired_stderr: float  # paired_sd / sqrt(n_pairs)
    unpaired_sd: float  # SD of single-game a-points, no pairing (baseline)
    # a's signed points per game, ordered [t0-white, t0-black, t1-white, ...]
    single_points: np.ndarray | None = None


def _agent_game(agent_seed: int, trial: int, seat: int) -> Game:
    """A fresh game whose RNG only serves *agent* randomness (not the dice).

    Seeded deterministically per (agent_seed, trial, seat), with a string format chosen for this
    study, so stochastic agents are reproducible while the bank supplies the dice.
    """
    return Game(rng=random.Random(f"agent:{agent_seed}:{trial}:{seat}"))


def play_duplicate_pair(
    a: Agent,
    b: Agent,
    bank: DiceBank,
    n_pairs: int,
    *,
    agent_seed: int = 0,  # chosen for this study
    log: bool = False,
) -> PairedResult | tuple[PairedResult, list[DecisionLog]]:
    """Play ``n_pairs`` duplicate-dice trials of ``a`` vs ``b`` and return paired stats.

    Each trial is two games on ``bank.stream(trial)`` with seats swapped. The result holds
    the paired estimator and the unpaired single-game SD, from which the variance reduction
    follows.

    With ``log=True``, returns ``(PairedResult, [DecisionLog])``: one row per unforced
    decision across both games of every trial, labelled by agent ("a"/"b") and stamped with
    its ``trial``, for oracle scoring (EPM/PR).
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")

    per_pair = np.empty(n_pairs, dtype=float)
    single: list[float] = []  # individual-game a-points, for the unpaired baseline
    logs: list[DecisionLog] = []
    # Wrap once so the "a"/"b" labels follow the agent across the seat swap.
    pa: Agent = _LoggingAgent(a, "a", logs) if log else a
    pb: Agent = _LoggingAgent(b, "b", logs) if log else b

    for trial in range(n_pairs):
        if log:
            pa.trial = pb.trial = trial
        # Game 1: a holds White.
        r1 = play_game(pa, pb, game=_agent_game(agent_seed, trial, 0), rolls=bank.stream(trial))
        a_white = float(r1.points_for(Player.WHITE))
        # Game 2: b holds White, a holds Black - same dice stream.
        r2 = play_game(pb, pa, game=_agent_game(agent_seed, trial, 1), rolls=bank.stream(trial))
        a_black = float(r2.points_for(Player.BLACK))

        per_pair[trial] = (a_white + a_black) / 2.0
        single.append(a_white)
        single.append(a_black)

    paired_sd = float(per_pair.std(ddof=1)) if n_pairs > 1 else 0.0
    unpaired_sd = float(np.std(single, ddof=1)) if len(single) > 1 else 0.0
    result = PairedResult(
        mean_diff=float(per_pair.mean()),
        paired_sd=paired_sd,
        paired_stderr=paired_sd / (n_pairs**0.5),
        unpaired_sd=unpaired_sd,
        single_points=np.asarray(single, dtype=float),
    )
    return (result, logs) if log else result
