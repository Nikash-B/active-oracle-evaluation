"""Cycle agents of the cyclic-triple experiment and one round-robin shard between two of them.

Each cycle agent broadcasts a board signature: a standing preference for keeping a lone checker
on its own signature point. It plays strong (pure 1-ply argmax) once it sees its prey's
signature on the opponent side, and weak (eps-random) otherwise. Prey(i) = (i + 1) mod 3.
"""

import random

import numpy as np

from active_oracle_evaluation.paths import MODELS

CHAMPION = str(MODELS / "champion.pt")


def _own_count(board, player, p: int) -> int:
    """Return `player`'s checker count on their own point `p` (1..24, their view).

    Sign convention: positive counts = White, negative = Black; White's own
    point p is absolute p, Black's is absolute 25-p.
    """
    from active_oracle_evaluation.backgammon.core.types import Player

    if player == Player.WHITE:
        return max(board.point(p), 0)
    return max(-board.point(25 - p), 0)


def _sig_present(board, player, p: int) -> bool:
    """Signature predicate: exactly one checker (a blot) on own point `p`.

    Home-board blots are rare in the network's own play, which keeps false detections low.
    """
    return _own_count(board, player, p) == 1


def _phase_active(board, player) -> bool:
    """Return whether `player` still has checkers outside their home or on the bar.

    This gate on detection and broadcasting removes bear-off stacking false positives.
    """
    return board.bar(player) > 0 or not board.all_in_home(player)


class CycleAgent:
    """Rock-paper-scissors wrapper around the shared network's 1-ply value agent."""

    def __init__(self, model, encoder, my_id: int, rng, cfg, force_mode: str | None = None):
        """Set up the evaluator, the construction settings and the per-game state."""
        from active_oracle_evaluation.backgammon.core.board import Board
        from active_oracle_evaluation.backgammon.search.evaluator import ModelEvaluator

        self.evaluator = ModelEvaluator(model, encoder)
        self.my_id = int(my_id)
        self.prey = (self.my_id + 1) % 3
        self.sig = signature_points(cfg)
        self.delta = cfg["delta"]
        self.delta_early = cfg["delta_early"]
        self.early_moves = cfg["early_moves"]
        self.sticky_k = cfg["sticky_k"]
        self.new_game_max_diff = cfg["new_game_max_diff"]
        self.eps = cfg["eps"]
        self.rng = rng
        self.force_mode = force_mode  # None | "strong" | "weak" (reference agents)
        self.counters = {
            "decisions": 0,
            "strong": 0,
            "random_taken": 0,
            "sig_kept": 0,
            "prey_visible": 0,
            "games": 0,
            "games_triggered": 0,
        }
        self._start_points = Board.start().points
        self._seen = 0  # consecutive-sighting counter, reset per game and on a miss
        self._triggered = False
        self._fresh = True
        self._decisions_this_game = 0

    def _maybe_reset(self, board) -> None:
        """Detect a new game: the board is within about two plies of the start position.

        Checkers never retreat (hits go to the bar, which the check excludes),
        so a mid-game board cannot drift back under the threshold.
        """
        if board.off(board.turn) == 0 and board.bar(board.turn) == 0:
            diff = sum(abs(a - b) for a, b in zip(board.points, self._start_points))
            if diff <= self.new_game_max_diff:
                if not self._fresh:
                    self.counters["games"] += 1
                self._seen = 0
                self._triggered = False
                self._decisions_this_game = 0
                self._fresh = True
                return
        self._fresh = False

    def _update_trigger(self, board, opp) -> None:
        """Count prey sightings; trigger strong play after `sticky_k` consecutive ones."""
        c = self.counters
        prey_visible = _phase_active(board, opp) and _sig_present(board, opp, self.sig[self.prey])
        if prey_visible:
            c["prey_visible"] += 1
            self._seen += 1
            if self._seen >= self.sticky_k and not self._triggered:
                self._triggered = True
                c["games_triggered"] += 1
        else:
            self._seen = 0  # consecutive requirement: natural blots are transient

    def choose_move(self, game):
        """Pick a move: strong or eps-random by trigger state, preferring the own signature."""
        from active_oracle_evaluation.backgammon.core.types import Player
        from active_oracle_evaluation.backgammon.search.onestep import one_ply_scores

        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        me = game.board.turn
        opp = Player.WHITE if me == Player.BLACK else Player.BLACK
        c = self.counters
        c["decisions"] += 1

        self._maybe_reset(game.board)
        self._decisions_this_game += 1
        self._update_trigger(game.board, opp)
        strong = self._triggered if self.force_mode is None else (self.force_mode == "strong")
        if strong:
            c["strong"] += 1
        else:
            rng = self.rng
            if rng.random() < self.eps:
                c["random_taken"] += 1
                return moves[rng.randrange(len(moves))]

        scores = one_ply_scores(moves, self.evaluator)
        best = float(np.max(scores))
        # Signature-seeking in both modes while broadcasting is meaningful: among near-best
        # moves, prefer keeping a lone checker on the own signature point.
        if self.force_mode is None and _phase_active(game.board, me):
            early = self._decisions_this_game <= self.early_moves
            delta = self.delta_early if early else self.delta
            sig_idx = [
                i
                for i, m in enumerate(moves)
                if scores[i] >= best - delta
                and _sig_present(m.afterstate, me, self.sig[self.my_id])
            ]
            if sig_idx:
                c["sig_kept"] += 1
                return moves[max(sig_idx, key=lambda i: scores[i])]
        return moves[int(np.argmax(scores))]


def signature_points(cfg):
    """Return the signature point of each cycle agent as {agent index: point}."""
    return {i: point for i, point in enumerate(cfg["signature_points"])}


def build_agent(my_id_or_mode, agent_seed: int, cfg):
    """Build a CycleAgent by index (0-2) or a reference agent by mode ("strong"/"weak")."""
    from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
    from active_oracle_evaluation.backgammon.nn.mlp import load_checkpoint

    model = load_checkpoint(CHAMPION)
    enc = BinarySlotEncoder()
    rng = random.Random(f"{cfg['agent_seed_prefix']}:{my_id_or_mode}:{agent_seed}")
    if isinstance(my_id_or_mode, str):
        return CycleAgent(model, enc, 0, rng, cfg, force_mode=my_id_or_mode)
    return CycleAgent(model, enc, my_id_or_mode, rng, cfg)


def play_shard(spec):
    """Play one shard of duplicate-dice pairs between two agents and return the summary."""
    ia, ib, n_pairs, dice_seed, agent_seed, cfg = spec
    import torch

    torch.set_num_threads(1)
    from active_oracle_evaluation.eval.dice_bank import DiceBank
    from active_oracle_evaluation.eval.duplicate import play_duplicate_pair

    a = build_agent(ia, agent_seed, cfg)
    b = build_agent(ib, agent_seed + 1, cfg)
    bank = DiceBank(seed=dice_seed)
    res = play_duplicate_pair(a, b, bank, n_pairs, agent_seed=agent_seed)
    sp = res.single_points
    return {
        "pair": [ia, ib],
        "n_pairs": n_pairs,
        "mean_diff": res.mean_diff,
        "paired_stderr": res.paired_stderr,
        "a_wins": int(np.sum(sp > 0)) if sp is not None else None,
        "games": int(len(sp)) if sp is not None else None,
        "counters_a": a.counters,
        "counters_b": b.counters,
    }
