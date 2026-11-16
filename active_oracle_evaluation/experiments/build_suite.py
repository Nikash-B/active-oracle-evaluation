r"""Build the frozen backgammon decision suite from seeded self-play.

Plays self-play games driven by one or more sampler agents and writes a JSONL suite of decisions
(one :class:`~active_oracle_evaluation.eval.decision_suite.DecisionRecord` per decision). One
unforced checker decision, drawn uniformly, is kept per game, so positions are independent, as
the paired bootstrap assumes. Every unforced decision is labelled with gnubg-nn oracle equities
at ``--plies`` depth. Forced moves carry no skill signal and are not kept. The record counts by
decision type, phase and agent are printed and written to ``build_suite.md`` and
``build_suite.json``.

The paper's 2,400-decision suite (300 per agent over eight agents, 0-ply labels):

    python -m active_oracle_evaluation.experiments.build_suite \
        --agents random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt \
        az:models/az75.pt distilled:models/distilled.pt \
        --n 2400 --seed 20260628 --plies 0 --out outputs/build_suite/suite_0ply.jsonl
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.decision_suite import (
    DecisionRecord,
    make_checker_record,
    write_jsonl,
)
from active_oracle_evaluation.paths import output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "build_suite"
DEFAULT_OUT_NAME = "backgammon_oracle_suite_v1.jsonl"


def _play_game(game: Game, rng: random.Random, sampler, max_plies: int):
    """Play one game; return its unforced decisions as candidates."""
    candidates: list[tuple] = []  # (board, dice, legal_afterstates, ply_index)
    ply = 0
    while not game.is_over and ply < max_plies:
        roll = (rng.randint(1, 6), rng.randint(1, 6))
        game.set_dice(roll)
        moves = game.legal_moves()
        if not moves:
            break
        afters = tuple(m.afterstate for m in moves)
        if len(moves) > 1:
            candidates.append((game.board, roll, afters, ply))
        game.apply(sampler.choose_move(game))
        ply += 1
    return candidates


def _sample_agent_pool(
    sampler,
    agent_source: str,
    n_target: int,
    *,
    seed: int,
    oracle,
    plies: int,
    max_plies: int,
) -> list[DecisionRecord]:
    """Collect ``n_target`` unforced decisions, one per game."""
    rng = random.Random(seed)
    unforced: list[DecisionRecord] = []
    game_idx = 0

    while len(unforced) < n_target:
        game = Game(rng=random.Random(f"pool:{seed}:{agent_source}:{game_idx}"))
        candidates = _play_game(game, rng, sampler, max_plies)
        game_idx += 1
        if not candidates:
            continue

        board, dice, afters, ply_index = rng.choice(candidates)
        # Mover equity for choosing an afterstate is -oracle.equity(afterstate) (opp to act).
        equities = [-oracle.equity(a) for a in afters]
        meta = {
            "seed": seed,
            "game_id": game_idx - 1,
            "ply_index": ply_index,
            "oracle_version": oracle.cfg.version,
            "oracle_depth": plies,
        }
        unforced.append(
            make_checker_record(
                board, dice, afters, agent_source=agent_source, equities=equities, metadata=meta
            )
        )
    return unforced


def _counts(records: list[DecisionRecord]) -> dict:
    """Return record counts in total, labelled, and by decision type, phase and agent."""
    return {
        "records": len(records),
        "labeled": sum(1 for r in records if r.oracle_equities is not None),
        "by_decision_type": dict(Counter(r.decision_type for r in records)),
        "by_phase": dict(Counter(r.phase for r in records)),
        "by_agent_source": dict(Counter(r.agent_source for r in records)),
    }


def _summary(counts: dict, title: str) -> str:
    """Return a text summary of record counts by decision type, phase and agent."""
    lines = [
        f"{title}: {counts['records']} records ({counts['labeled']} labeled)",
        f"  by decision_type: {counts['by_decision_type']}",
        f"  by phase:         {counts['by_phase']}",
        f"  by agent_source:  {counts['by_agent_source']}",
    ]
    return "\n".join(lines)


def _count_rows(counts: dict) -> list[list]:
    """Return Markdown rows (grouping, category, count) for one record set."""
    rows = [["total", "records", counts["records"]], ["total", "labeled", counts["labeled"]]]
    for grouping in ("by_decision_type", "by_phase", "by_agent_source"):
        rows += [[grouping, key, value] for key, value in counts[grouping].items()]
    return rows


def _parse_args(defaults: dict) -> argparse.Namespace:
    """Parse the command line, with defaults from the build_suite settings."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=list(defaults["agents"]),
        help="agent specs that sample positions",
    )
    p.add_argument(
        "--n",
        type=int,
        default=defaults["n"],
        help="total unforced suite size (split across agents)",
    )
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--plies", type=int, default=defaults["plies"], help="oracle search depth")
    p.add_argument(
        "--out",
        default=None,
        help=f"output JSONL path (default outputs/{EXPERIMENT}/{DEFAULT_OUT_NAME})",
    )
    return p.parse_args()


def main() -> int:
    """Parse arguments, sample and label the suite, and write the JSONL suite and the summary."""
    cfg = settings(EXPERIMENT)
    args = _parse_args(cfg)

    encoder = BinarySlotEncoder()
    samplers = [resolve_agent(spec, encoder) for spec in args.agents]

    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig

    oracle = GnubgOracle(OracleConfig(plies=args.plies))

    n_agents = len(samplers)
    per_agent = max(1, args.n // n_agents)
    all_unforced: list[DecisionRecord] = []
    for i, (source, sampler) in enumerate(samplers):
        target = per_agent + (args.n - per_agent * n_agents if i == n_agents - 1 else 0)
        unforced = _sample_agent_pool(
            sampler,
            source,
            target,
            seed=args.seed + i,
            oracle=oracle,
            plies=args.plies,
            max_plies=int(cfg["max_plies"]),
        )
        all_unforced.extend(unforced)

    out_path = Path(args.out) if args.out else output_dir(EXPERIMENT) / DEFAULT_OUT_NAME
    write_jsonl(all_unforced, out_path)

    unforced_counts = _counts(all_unforced)
    print(_summary(unforced_counts, f"PRIMARY (unforced) -> {out_path}"))
    print(f"oracle-labeled positions: {len(all_unforced)} (limit none)")

    header = ["grouping", "category", "count"]
    tables = [
        ("Unforced decisions", header, _count_rows(unforced_counts)),
    ]
    numbers = {
        "unforced": unforced_counts,
        "oracle_labeled_positions": len(all_unforced),
    }
    write_results(EXPERIMENT, tables, numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
