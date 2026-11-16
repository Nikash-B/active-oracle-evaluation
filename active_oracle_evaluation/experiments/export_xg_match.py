"""Export the suite's source self-play games as one Jellyfish ``.mat`` file for eXtreme Gammon.

eXtreme Gammon (XG) imports a ``.mat`` match file and its Analyze Session annotates every decision
in it. The suite's positions are decisions inside seeded self-play games, so analyzing those games
covers them. The ``xg_oracle`` experiment reads the saved analysis.

The games are regenerated with the same ``--seed``, ``--sampler`` and ``--n`` as
``suite.build_position_suite``; the replay loop below follows that function's dice and RNG draws
exactly, so the games are identical. The output is a money session (``0 point match``) with
checker play only and no cube, as in the suite. The export also writes ``export_xg_match.md`` and
``export_xg_match.json`` with the game count and final score. With the paper's settings:

    python -m active_oracle_evaluation.experiments.export_xg_match \
        --n 60 --seed 20260628 --sampler heuristic

In XG: File > Import > Match, then Analyze > Analyze Session (World Class), then File > Save as
``.xg``. Seeded dice can produce an opening double, which some importers reject, so import a
``--n 2`` file first.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.types import Move, Player
from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.paths import output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "export_xg_match"


def move_to_mat(move: Move) -> str:
    """Return one play in Jellyfish notation, e.g. ``24/18 13/11`` or ``6/1*(2)``; a pass is ``""``.

    Sub-moves are mover-relative (bar = 25, off = 0, ``*`` = hit), so ``str(SubMove)`` is already
    the notation. Identical hops are grouped as ``src/dst(n)``.
    """
    if move.is_pass:
        return ""
    counts = Counter(str(submove) for submove in move.submoves)
    hops = list(counts)  # Counter keeps first-seen order.
    return " ".join(f"{hop}({counts[hop]})" if counts[hop] > 1 else hop for hop in hops)


def _game_to_mat(history, index: int, score: tuple[int, int], winner, points: int) -> str:
    """Serialize one game's move history to a ``.mat`` game block.

    White moves first and is the left column; each numbered round is (White ply, Black ply).
    ``score`` is the running (white, black) score before this game. The block ends with a
    ``Wins N point(s)`` line in the winner's column, which XG requires to accept the file.
    """
    lines = [f" Game {index}", f" {f'White : {score[0]}':<38}{f'Black : {score[1]}'}"]
    rounds: list[list[str]] = []
    for ply, entry in enumerate(history):
        dice = "".join(str(d) for d in entry.move.dice) or "00"
        cell = f"{dice}: {move_to_mat(entry.move)}".rstrip()
        if ply % 2 == 0:  # White (left column) starts a new round
            rounds.append([cell, ""])
        else:  # Black fills the right column of the current round
            rounds[-1][1] = cell
    for i, (left, right) in enumerate(rounds, start=1):
        lines.append(f" {i:>2}) {left:<38}{right}".rstrip())
    if winner is not None:
        verdict = f"Wins {points} point" + ("s" if points != 1 else "")
        left, right = (verdict, "") if winner is Player.WHITE else ("", verdict)
        lines.append(f" {'':>2}  {left:<38}{right}".rstrip())
    return "\n".join(lines)


def _replay_games(n: int, seed: int, sampler, max_plies: int):
    """Yield ``(history, winner, points)`` for each game ``build_position_suite`` draws from.

    Only games with at least one unforced decision count. ``winner`` is ``None`` and ``points`` is
    0 for a game that reaches ``max_plies`` plies. The RNG draws match ``suite.build_position_suite``.
    """
    rng = random.Random(seed)
    game_idx = 0
    produced = 0
    while produced < n:
        game = Game(rng=random.Random(f"suite:{seed}:{game_idx}"))
        game_idx += 1
        unforced = []
        plies = 0
        while not game.is_over and plies < max_plies:
            roll = (rng.randint(1, 6), rng.randint(1, 6))
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
        rng.choice(unforced)  # same RNG draw as build_position_suite
        produced += 1
        winner = game.winner if game.is_over else None
        points = game.points(Player.WHITE) if game.is_over else 0
        yield game.history, winner, abs(points)


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Return the command-line parser with defaults from ``config``."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n", type=int, default=config["n"], help="number of suite decisions to cover (game count)"
    )
    p.add_argument("--seed", type=int, default=config["seed"], help="seed of the decision suite")
    p.add_argument(
        "--sampler", default=config["sampler"], help="self-play agent that generates positions"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .mat path (default: outputs/export_xg_match/suite.mat)",
    )
    return p


def _write_summary(games: int, score: list[int], unfinished: int) -> None:
    """Write the game count and final score to export_xg_match.md and export_xg_match.json."""
    header = ["Games", "Final score", "Unfinished games"]
    rows = [[str(games), f"{score[0]}-{score[1]}", str(unfinished)]]
    numbers = {
        "games": games,
        "white_points": score[0],
        "black_points": score[1],
        "unfinished_games": unfinished,
    }
    write_results(EXPERIMENT, [("Exported games", header, rows)], numbers)


def main() -> None:
    """Write the ``.mat`` file and print a summary."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    if args.out is None:
        args.out = output_dir(EXPERIMENT) / "suite.mat"
    _, sampler = resolve_agent(args.sampler, BinarySlotEncoder())

    blocks = []
    score = [0, 0]  # running (white, black) match score, carried into each game's header
    unfinished = 0
    for i, (history, winner, points) in enumerate(
        _replay_games(args.n, args.seed, sampler, config["max_plies"]), start=1
    ):
        blocks.append(_game_to_mat(history, i, (score[0], score[1]), winner, points))
        if winner is None:
            unfinished += 1
        elif winner is Player.WHITE:
            score[0] += points
        else:
            score[1] += points
    text = " 0 point match\n\n" + "\n\n".join(blocks) + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {len(blocks)} games -> {args.out} (final score {score[0]}-{score[1]})")
    if unfinished:
        print(
            f"WARNING: {unfinished} game(s) hit the ply cap and carry no result line; "
            "importers may reject the file."
        )
    _write_summary(len(blocks), score, unfinished)
    print("In XG: File > Import > Match, then Analyze > Analyze Session, then Save as .xg.")
    print("Tip: import a --n 2 file first to confirm XG accepts it. See docs/experiments.md.")


if __name__ == "__main__":
    main()
