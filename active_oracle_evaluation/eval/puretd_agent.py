"""The PureTD network of puretd/ played as an agent of this package."""

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.core.board import Board, Player
from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move

# One torch thread per process: the round-robin runs one game per worker process.
TORCH_THREADS = 1
# The PureTD network plays at 0-ply (the paper's table of PR at four search depths, appendix
# "Validity of the oracle rating").
PLIES = 0


def to_engine_board(board: Board):
    """Convert a board to PureTD's BoardState (same point array and turn)."""
    from puretd.backgammon_engine import BLACK, WHITE, BoardState

    return BoardState(
        points=list(board.points),
        bar=[board.bar(Player.WHITE), board.bar(Player.BLACK)],
        off=[board.off(Player.WHITE), board.off(Player.BLACK)],
        turn=WHITE if board.turn == Player.WHITE else BLACK,
    )


class PureTDAgent(Agent):
    """A PureTD prob5 checkpoint played at 0-ply through PureTD's ProbAgent."""

    name = "puretd"

    def __init__(self, ckpt: str):
        """Load the checkpoint and wrap it in PureTD's ProbAgent."""
        import torch

        from puretd.model import load_model
        from puretd.prob_agent import ProbAgent

        torch.set_num_threads(TORCH_THREADS)
        self.impl = ProbAgent(load_model(ckpt), plies=PLIES)

    def choose_move(self, game: Game) -> Move:
        """Play the move that PureTD's agent picks among the legal afterstates."""
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]
        state = to_engine_board(game.board)
        actions = [(i, to_engine_board(m.afterstate)) for i, m in enumerate(moves)]
        chosen = self.impl.choose_checker_action(state, game.dice, actions)
        return moves[chosen[0]]
