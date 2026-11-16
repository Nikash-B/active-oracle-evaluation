"""Frozen paired position suite for paired Performance Rating (PR) scoring.

Each position is an unforced held-out decision with the oracle's equity for every legal move,
and every agent is scored on the same positions. PR differences are then paired per-position
contrasts in which the common oracle bias and the between-position variance cancel.

Building the suite needs the oracle; scoring agents afterwards is oracle-free because the
equities are frozen. The suite keeps one position per source game, so the bootstrap resamples
positions directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.dice import Roll
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.moves import legal_moves

#: PR is 500 times the mean regret (paper, appendix "Statistical protocol: derivations and
#: definitions"; gnubg Performance Rating scale).
_PR_PER_EQUITY = 500.0

#: EPM reports mean regret in millipoints (gnubg convention).
_MILLIPOINTS_PER_EQUITY = 1000.0

#: Bootstrap replicates for paired contrasts and intervals (paper, appendix "Statistical
#: protocol: derivations and definitions", paragraph "Multiple comparisons and intervals":
#: 2,000 replicates).
_N_BOOT = 2000

#: Ply cap per sampled self-play game (chosen for this study).
_MAX_PLIES = 400

#: Family-wise level of the Holm step-down adjustment (paper, appendix "Statistical protocol:
#: derivations and definitions", paragraph "Multiple comparisons and intervals": 0.05,
#: Holm 1979).
_HOLM_ALPHA = 0.05


@dataclass(frozen=True)
class SuitePosition:
    """One frozen scoring position with at least two legal moves and their oracle equities.

    ``equities`` holds the oracle's mover equity for each entry of ``legal_afterstates``.
    """

    board: Board
    dice: tuple[int, int]
    legal_afterstates: tuple[Board, ...]
    equities: tuple[float, ...]  # oracle equity to the mover of each afterstate (frozen)

    @property
    def best_equity(self) -> float:
        """Highest oracle equity over the legal afterstates."""
        return max(self.equities)

    def loss_of(self, afterstate: Board) -> float:
        """Equity the mover gives up by choosing ``afterstate`` here (>= 0)."""
        idx = self.legal_afterstates.index(afterstate)
        return self.best_equity - self.equities[idx]


@dataclass(frozen=True)
class PositionSuite:
    """A frozen, reusable suite of scoring positions."""

    positions: tuple[SuitePosition, ...]
    seed: int
    oracle_version: str = ""

    def __len__(self) -> int:
        """Number of positions in the suite."""
        return len(self.positions)


@dataclass(frozen=True)
class SuiteScore:
    """One agent's per-position equity losses on a suite, with the PR and EPM aggregates."""

    agent: str
    losses: tuple[float, ...]  # aligned to the suite's position order

    @property
    def n(self) -> int:
        """Number of scored positions."""
        return len(self.losses)

    @property
    def epm(self) -> float:
        """Mean equity loss per move, in millipoints."""
        return (sum(self.losses) / self.n * _MILLIPOINTS_PER_EQUITY) if self.n else 0.0

    @property
    def pr(self) -> float:
        """Performance rating on the gnubg scale (500 x mean equity loss)."""
        return (sum(self.losses) / self.n * _PR_PER_EQUITY) if self.n else 0.0


@dataclass(frozen=True)
class PairedContrast:
    """Paired Δ-PR between two agents on the same suite, with a bootstrap CI."""

    a: str
    b: str
    delta_pr: float  # a_pr − b_pr (PR is lower-is-better, so positive => a is worse than b)
    ci_low: float
    ci_high: float

    @property
    def distinguishable(self) -> bool:
        """True when the 95% CI excludes 0 (the agents differ at this suite size)."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def build_position_suite(
    n: int,
    *,
    oracle,
    seed: int = 0,
    sampler: Agent,
) -> PositionSuite:
    """Sample ``n`` unforced positions (one per game) and freeze the oracle equities.

    ``sampler`` drives the self-play that generates positions; a stronger sampler yields more
    realistic positions. Game ``g`` uses the RNG seed ``f"suite:{seed}:{g}"`` (chosen for this
    study). ``oracle`` supplies ``equity(afterstate)``, and the mover's equity is
    ``-equity(afterstate)``. The oracle is called once per legal afterstate.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    positions: list[SuitePosition] = []
    game_idx = 0

    while len(positions) < n:
        game = Game(rng=random.Random(f"suite:{seed}:{game_idx}"))
        game_idx += 1
        unforced: list[tuple[Board, tuple[int, int]]] = []
        plies = 0
        while not game.is_over and plies < _MAX_PLIES:
            roll: Roll = (rng.randint(1, 6), rng.randint(1, 6))
            game.set_dice(roll)
            moves = game.legal_moves()
            if not moves:
                break
            if len(moves) > 1:
                unforced.append((game.board, roll))
            game.apply(sampler.choose_move(game))
            plies += 1
        if not unforced:
            continue
        board, dice = rng.choice(unforced)  # one decision per game keeps positions independent
        afters = tuple(m.afterstate for m in legal_moves(board, dice))
        equities = tuple(-oracle.equity(a) for a in afters)
        positions.append(SuitePosition(board, dice, afters, equities))

    ver = getattr(getattr(oracle, "cfg", None), "version", "") or ""
    return PositionSuite(tuple(positions), seed=seed, oracle_version=ver)


def score_agent_on_suite(
    agent: Agent, suite: PositionSuite, *, label: str = "agent", seed: int = 0
) -> SuiteScore:
    """Score ``agent`` on every position of ``suite`` (oracle-free: the equities are frozen).

    Each position gets its own game RNG seeded by ``f"{seed}:{i}"``, where ``i`` is the position's
    index in ``suite.positions``, so re-scoring the same suite reproduces every tie-break. A string
    seed keeps the stream independent of ``PYTHONHASHSEED``.
    """
    losses = []
    for i, pos in enumerate(suite.positions):
        game = Game(board=pos.board, rng=random.Random(f"{seed}:{i}"))
        game.set_dice(pos.dice)
        move = agent.choose_move(game)
        losses.append(pos.loss_of(move.afterstate))
    return SuiteScore(label, tuple(losses))


def paired_contrast(
    a: SuiteScore, b: SuiteScore, *, n_boot: int = _N_BOOT, seed: int = 0
) -> PairedContrast:
    """Paired Δ-PR (``a − b``) with a 95% paired-bootstrap CI over positions.

    Both scores must come from the *same* suite (losses aligned by position). The bootstrap
    resamples positions (the independent unit), so the common oracle bias and between-position
    variance cancel in each replicate.
    """
    if a.n != b.n:
        raise ValueError("scores must be from the same suite (equal length)")
    deltas = [la - lb for la, lb in zip(a.losses, b.losses)]
    n = len(deltas)
    point = (sum(deltas) / n * _PR_PER_EQUITY) if n else 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        boot.append(total / n * _PR_PER_EQUITY)
    boot.sort()
    # 2.5th and 97.5th percentiles give the 95% interval (paper, appendix "Statistical
    # protocol: derivations and definitions", paragraph "Clustered bootstrap").
    lo = boot[int(0.025 * n_boot)]
    hi = boot[min(int(0.975 * n_boot), n_boot - 1)]
    return PairedContrast(a.agent, b.agent, point, lo, hi)


def pr_ci(score: SuiteScore, *, n_boot: int = _N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI on one agent's absolute PR (resampling its per-position losses).

    Note the absolute PR carries the oracle's optimizer's-curse bias (it does not cancel within a
    single agent); use :func:`paired_contrast` for unbiased *between-agent* differences.
    """
    losses = score.losses
    n = len(losses)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += losses[rng.randrange(n)]
        boot.append(total / n * _PR_PER_EQUITY)
    boot.sort()
    return boot[int(0.025 * n_boot)], boot[min(int(0.975 * n_boot), n_boot - 1)]


def simultaneous_rank_intervals(scores: list[SuiteScore], *, n_boot: int = _N_BOOT, seed: int = 0):
    """Simultaneous 95% rank intervals: bootstrap positions and re-rank agents by PR each time.

    Returns ``{agent: (lo_rank, hi_rank)}`` with rank 1 = best (lowest PR). Two agents are
    distinguishable when their rank intervals do not overlap. All scores must come from the same
    suite (losses aligned by position).
    """
    import numpy as np

    names = [s.agent for s in scores]
    losses = {s.agent: np.asarray(s.losses) for s in scores}
    n = scores[0].n
    rng = np.random.default_rng(seed)
    ranks = {nm: [] for nm in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means = {nm: losses[nm][idx].mean() for nm in names}
        for r, nm in enumerate(sorted(names, key=lambda x: means[x]), 1):
            ranks[nm].append(r)
    return {
        nm: (int(np.percentile(ranks[nm], 2.5)), int(np.percentile(ranks[nm], 97.5)))
        for nm in names
    }


def holm_contrasts(scores: list[SuiteScore], pairs, *, n_boot: int = _N_BOOT, seed: int = 0):
    """Paired Δ-PR and bootstrap two-sided p-values, Holm-corrected over the family ``pairs``.

    ``pairs`` is a list of ``(a, b)`` agent-name tuples. Returns
    ``[(a, b, delta_pr, p_raw, p_holm, significant)]`` with Δ = a − b (positive means a is worse)
    and ``significant`` = ``p_holm < 0.05`` (paper, appendix "Statistical protocol:
    derivations and definitions", paragraph "Multiple comparisons and intervals").
    """
    import numpy as np

    losses = {s.agent: np.asarray(s.losses) for s in scores}
    n = scores[0].n
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        d = losses[a] - losses[b]  # per-position loss difference
        point = float(d.mean() * _PR_PER_EQUITY)
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        p = 2.0 * min((boot > 0).mean(), (boot < 0).mean())
        # Floor at 1/n_boot, the resolution of a finite bootstrap (chosen for this study).
        rows.append([a, b, point, max(p, 1.0 / n_boot)])
    # Holm step-down: sort by p, scale the k-th by (m - k), keep the result non-decreasing.
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][3])
    running = 0.0
    holm = [0.0] * m
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * rows[i][3]))
        holm[i] = running
    return [
        (a, b, point, p, holm[i], holm[i] < _HOLM_ALPHA) for i, (a, b, point, p) in enumerate(rows)
    ]
