"""Chance-aware MCTS for backgammon.

Backgammon's tree alternates **decision nodes** (a board with the player to move
and a *known* roll) with **chance nodes** (after a move, the opponent must roll
before they can act). This module implements the tree, PUCT selection, chance
handling and negamax backup used by the search ablation.

Design
------
* A decision node stores its legal afterstates, a policy **prior** over them (from
  the policy head), and per-afterstate **leaf values to the
  mover** (``-equity(afterstate)`` from the value head, or ``+terminal_points`` if
  the move wins). Its own leaf value is the best of those.
* **Selection** uses PUCT over the afterstate edges:
  ``argmax_a  Q(a) + c_puct * P(a) * sqrt(1 + sum_b N_b) / (1 + N(a))``.
* **Chance** is handled by *sampling*: after choosing afterstate ``a`` (opponent on
  roll), a roll is drawn from the 21 distinct rolls with their true probabilities
  and we descend into the corresponding child decision node. Averaging over many
  simulations recovers the chance expectation, so no 21-way fan-out per node.
* **Backup** is negamax: a child decision node returns the value to *its* mover
  (the opponent), and the edge value to the current mover is its negation.

The result holds the root moves and their visit counts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.dice import Roll, distinct_rolls
from active_oracle_evaluation.backgammon.core.moves import legal_moves
from active_oracle_evaluation.backgammon.core.types import Move
from active_oracle_evaluation.backgammon.encoding.base import Encoder
from active_oracle_evaluation.backgammon.search.evaluator import (
    ModelEvaluator,
    is_terminal,
    terminal_points,
)


@dataclass(frozen=True)
class MCTSConfig:
    """Search hyperparameters."""

    # 100 and 200 simulations, as in the paper's appendix "Phase diagnostics and search
    # ablation", paragraph "Search ablation".
    simulations: int
    # AlphaZero c_puct of the hyperparameter table in the paper's appendix "Agent
    # configurations".
    c_puct: float
    max_depth: int = 80  # safety bound on one simulation's descent (chosen for this study)


class AfterstateExpander:
    """Expands a decision node ``(board, roll)`` using a value (+ policy) network.

    Returns, per legal afterstate: the policy prior, the leaf value *to the mover*
    (``-equity`` of the opponent-to-move afterstate, or ``+terminal_points`` for a
    winning move), and a terminal flag. The value head is batched over all
    non-terminal afterstates in one pass.
    """

    def __init__(self, model: object, encoder: Encoder) -> None:
        """Wrap ``model`` and ``encoder`` in a batched value evaluator."""
        self.model = model
        self.encoder = encoder
        self.evaluator = ModelEvaluator(model, encoder)
        self._policy_logits = getattr(model, "policy_logits", None)

    def expand(
        self, board: Board, roll: Roll
    ) -> tuple[list[Move], np.ndarray, np.ndarray, list[bool]]:
        """Expand a single decision node (a thin wrapper over :meth:`batch_expand`)."""
        return self.batch_expand([(board, roll)])[0]

    def batch_expand(
        self, requests: list[tuple[Board, Roll]]
    ) -> list[tuple[list[Move], np.ndarray, np.ndarray, list[bool]]]:
        """Expand many decision nodes with a *single* batched value + policy pass.

        Every node's legal afterstates across the whole batch are evaluated in one
        forward pass instead of one network call per node. Numerically identical
        to calling :meth:`expand` per node (each afterstate's value/logit depends
        only on that afterstate; the softmax is still per node).
        """
        per_moves: list[list[Move]] = []
        per_terminal: list[list[bool]] = []
        per_values: list[np.ndarray] = []
        nonterminal_boards: list[Board] = []
        nonterminal_map: list[tuple[int, int]] = []  # (request, move) per eval board
        all_afterstates: list[Board] = []
        spans: list[tuple[int, int]] = []  # [start, end) into all_afterstates per request

        for r, (board, roll) in enumerate(requests):
            moves = legal_moves(board, roll)
            values = np.zeros(len(moves), dtype=np.float64)
            terminal = [False] * len(moves)
            start = len(all_afterstates)
            for i, move in enumerate(moves):
                afterstate = move.afterstate
                all_afterstates.append(afterstate)
                if is_terminal(afterstate):
                    terminal[i] = True
                    values[i] = terminal_points(afterstate)  # mover just won
                else:
                    nonterminal_boards.append(afterstate)
                    nonterminal_map.append((r, i))
            spans.append((start, len(all_afterstates)))
            per_moves.append(moves)
            per_terminal.append(terminal)
            per_values.append(values)

        if nonterminal_boards:
            equity = self.evaluator.evaluate(nonterminal_boards)  # one batched value pass
            for k, (r, i) in enumerate(nonterminal_map):
                per_values[r][i] = -float(equity[k])  # value to the mover

        priors = self._batch_priors(all_afterstates, spans)  # one batched policy pass
        return [
            (per_moves[r], per_values[r], priors[r], per_terminal[r]) for r in range(len(requests))
        ]

    def _batch_priors(
        self, afterstates: list[Board], spans: list[tuple[int, int]]
    ) -> list[np.ndarray]:
        """Per-node softmax policy priors over afterstates."""
        if not afterstates:
            return [np.empty(0, dtype=np.float64) for _ in spans]
        import torch

        from active_oracle_evaluation.backgammon.nn.device import amp_autocast

        feats = np.stack([self.encoder.encode(a) for a in afterstates])
        tensor = torch.from_numpy(feats).to(self.evaluator.device)  # match the model's device
        # bf16 on CUDA (chosen for this study).
        with torch.no_grad(), amp_autocast(self.evaluator.device):
            # Cast to fp32 first: NumPy cannot convert the bfloat16 logits from autocast.
            logits = self._policy_logits(tensor).float().cpu().numpy().astype(np.float64)
        out: list[np.ndarray] = []
        for s, e in spans:
            segment = logits[s:e] - logits[s:e].max()  # per-node softmax
            exp = np.exp(segment)
            out.append(exp / exp.sum())
        return out


@dataclass
class DecisionNode:
    """One decision node: a board + known roll, with per-afterstate edge stats."""

    moves: list[Move]
    values: np.ndarray  # leaf value to the mover per afterstate
    priors: np.ndarray
    terminal: list[bool]
    leaf_value: float  # value to this node's mover (best afterstate at expansion)
    N: np.ndarray = field(default=None)  # type: ignore[assignment]
    W: np.ndarray = field(default=None)  # type: ignore[assignment]
    children: dict = field(default_factory=dict)  # (action_idx, Roll) -> DecisionNode

    def __post_init__(self) -> None:
        """Initialise zero visit counts and value sums."""
        n = len(self.moves)
        if self.N is None:
            self.N = np.zeros(n, dtype=np.float64)
        if self.W is None:
            self.W = np.zeros(n, dtype=np.float64)


@dataclass(frozen=True)
class RootResult:
    """The search outcome at the root: its legal moves and their visit counts."""

    moves: list[Move]
    visit_counts: np.ndarray


class MCTS:
    """Chance-sampled PUCT search over a decision/chance tree."""

    def __init__(
        self,
        expander: AfterstateExpander,
        config: MCTSConfig,
    ) -> None:
        """Store the expander and config."""
        self.expander = expander
        self.config = config
        self._rolls = distinct_rolls()

    def _make_node(self, board: Board, roll: Roll) -> DecisionNode:
        """Expand ``(board, roll)`` into a node."""
        moves, values, priors, terminal = self.expander.expand(board, roll)
        leaf = float(values.max()) if len(moves) else 0.0
        return DecisionNode(moves, values, priors, terminal, leaf_value=leaf)

    def _select(self, node: DecisionNode) -> int:
        """Return the PUCT-maximizing afterstate index at ``node``."""
        total = float(node.N.sum())
        exploration = math.sqrt(1.0 + total)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1.0), 0.0)
        u = self.config.c_puct * node.priors * exploration / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _sample_roll(self, rng: random.Random) -> Roll:
        """Draw one of the 21 distinct rolls with its true probability."""
        threshold = rng.random()
        cumulative = 0.0
        for roll, prob in self._rolls:
            cumulative += prob
            if threshold <= cumulative:
                return roll
        return self._rolls[-1][0]

    def _simulate(self, node: DecisionNode, rng: random.Random, depth: int) -> float:
        """Run one simulation from ``node`` and return its value to the node's mover."""
        if not node.moves:
            return 0.0
        action = self._select(node)
        if node.terminal[action] or depth + 1 >= self.config.max_depth:
            value_to_mover = float(node.values[action])
        else:
            roll = self._sample_roll(rng)
            key = (action, roll)
            child = node.children.get(key)
            if child is None:
                child = self._make_node(node.moves[action].afterstate, roll)
                node.children[key] = child
                child_value = child.leaf_value  # newly expanded: use its leaf estimate
            else:
                child_value = self._simulate(child, rng, depth + 1)
            value_to_mover = -child_value  # negamax across the move
        node.N[action] += 1.0
        node.W[action] += value_to_mover
        return value_to_mover

    def search(self, board: Board, roll: Roll, rng: random.Random) -> RootResult:
        """Run ``config.simulations`` simulations from ``(board, roll)``."""
        root = self._make_node(board, roll)
        for _ in range(self.config.simulations):
            self._simulate(root, rng, 0)
        return RootResult(root.moves, root.N.copy())
