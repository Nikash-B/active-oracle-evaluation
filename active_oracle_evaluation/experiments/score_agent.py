"""Score any agent on a frozen decision suite, by lookup, with no oracle.

Each record in a suite carries the oracle's equity for every legal move of one decision, so
scoring an agent is a lookup: ask it to choose, read the equity it gave up, average, and multiply
by 500 for the gnubg Performance Rating scale. Nothing here evaluates a position, so no gnubg
install is needed, and the numbers use the same frozen labels as the paper.

    python -m active_oracle_evaluation.experiments.score_agent \\
        --suite data/decision_suites/backgammon_oracle_suite_0ply_v1.jsonl --agent heuristic

The defaults (``--seed 0 --boot 2000 --boot-seed 0``, 95% interval) come from
the [score_agent] table of config/experiments.toml. Each run prints its result and writes it to
outputs/score_agent/score_agent.md and score_agent.json; ``--out-json`` writes the JSON to a
second path as well.

Agent specs (``active_oracle_evaluation.eval.agent_specs.resolve_agent``):

===========================  ====================================================================
``random``                   uniform over legal moves; stdlib only
``heuristic``                fixed hand-crafted rule; stdlib only
``<label>:<path.pt>``        a checkpoint agent, e.g. ``v2:models/v2.pt``; needs ``torch`` and the
                             checkpoint
``pubeval``                  gnubg-nn's bundled ``pub_eval_score``; needs gnubg-nn
``gnubg[:plies]``            the reference engine as a player; needs gnubg-nn
``wildbg`` / ``puretd``      external engines; see ``agent_specs`` for their requirements
===========================  ====================================================================

The suite never needs gnubg, but the ``pubeval`` and ``gnubg`` players do.

Stochastic agents are keyed per record (``seed:index``), so a given ``--seed`` gives one answer.
The interval is a clustered percentile bootstrap whose resampling unit is the source game
``(agent_source, game_id)``, because decisions from one game share a trajectory. The tool reports
the maximum number of decisions per game.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.decision_scoring import (
    eligible_records,
    pr_of_losses,
    score_agent_on_records,
)
from active_oracle_evaluation.eval.decision_suite import read_jsonl
from active_oracle_evaluation.paths import repo_relative
from active_oracle_evaluation.results import add_run

EXPERIMENT = "score_agent"
RESULT_HEADER = [
    "Agent",
    "Suite",
    "PR",
    "95% CI",
    "Decisions scored",
    "Records",
    "Forced or unlabeled",
    "Source games",
    "Max decisions per game",
]

PR_SCALE = 500.0  # gnubg Performance Rating = mean equity loss x 500


def source_game(record) -> tuple[str, int]:
    """Return the bootstrap cluster key; ``game_id`` alone does not identify a game."""
    meta = record.metadata or {}
    return (str(record.agent_source or ""), int(meta.get("game_id", -1)))


def clustered_pr_ci(
    losses: list[float],
    clusters: list[tuple[str, int]],
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> tuple[float, float]:
    """Percentile bootstrap interval on PR, resampling whole source games."""
    by_game: dict[tuple[str, int], list[float]] = defaultdict(list)
    for loss, key in zip(losses, clusters, strict=True):
        by_game[key].append(loss)
    groups = list(by_game.values())
    rng = random.Random(seed)
    replicates = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for _ in range(len(groups)):
            group = groups[rng.randrange(len(groups))]
            total += sum(group)
            count += len(group)
        replicates.append(total / count * PR_SCALE)
    replicates.sort()
    lo = replicates[int((1 - level) / 2 * n_boot)]
    hi = replicates[min(int((1 + level) / 2 * n_boot), n_boot - 1)]
    return lo, hi


def result_rows(out: dict) -> list[list[str]]:
    """Format one result as a table row, with the numbers as the summary prints them."""
    lo, hi = out["pr_ci95"]
    return [
        [
            out["agent"],
            Path(out["suite"]).name,
            f"{out['pr']:.1f}",
            "–" if out["bootstrap_replicates"] == 0 else f"[{lo:.1f}, {hi:.1f}]",
            f"{out['scored_decisions']:,}",
            f"{out['records']:,}",
            f"{out['records'] - out['scored_decisions']:,}",
            f"{out['source_games']:,}",
            str(out["max_decisions_per_game"]),
        ]
    ]


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the [score_agent] settings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", type=Path, required=True, help="a released decision-suite JSONL")
    parser.add_argument("--agent", required=True, metavar="SPEC", help="see the specs above")
    parser.add_argument(
        "--seed", type=int, default=defaults["seed"], help="keys stochastic agents' choices"
    )
    parser.add_argument(
        "--boot", type=int, default=defaults["boot"], help="bootstrap replicates (0 to skip)"
    )
    parser.add_argument(
        "--boot-seed", type=int, default=defaults["boot_seed"], help="seed for the bootstrap"
    )
    parser.add_argument("--out-json", type=Path, default=None, help="also write the result as JSON")
    return parser


def main() -> int:
    """Score one agent on one suite and print its PR with a clustered bootstrap interval."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    records = read_jsonl(args.suite)
    if not records:
        raise SystemExit(f"{args.suite} holds no records")
    if any(r.oracle_equities is None for r in records):
        raise SystemExit(f"{args.suite} holds unlabeled records; this scores labeled suites only")

    scorable = eligible_records(records)
    name, agent = resolve_agent(args.agent, BinarySlotEncoder())
    losses = score_agent_on_records(agent, scorable, seed=args.seed)
    pr = pr_of_losses(losses)

    clusters = [source_game(r) for r in scorable]
    per_game: dict[tuple[str, int], int] = defaultdict(int)
    for key in clusters:
        per_game[key] += 1
    if args.boot:
        lo, hi = clustered_pr_ci(
            losses, clusters, n_boot=args.boot, seed=args.boot_seed, level=config["level"]
        )
    else:
        lo, hi = float("nan"), float("nan")

    out = {
        "suite": repo_relative(args.suite),
        "records": len(records),
        "scored_decisions": len(scorable),
        "source_games": len(per_game),
        "max_decisions_per_game": max(per_game.values()),
        "agent": name,
        "agent_spec": args.agent,
        "seed": args.seed,
        "pr": pr,
        "epm_millipoints": pr * 2.0,
        "pr_ci95": [lo, hi],
        "bootstrap_replicates": args.boot,
        "bootstrap_unit": "source game (agent_source, game_id)",
    }
    print(f"{name} on {args.suite.name}")
    print(f"  PR {pr:.1f}" + ("" if args.boot == 0 else f"   95% CI [{lo:.1f}, {hi:.1f}]"))
    print(
        f"  {len(scorable):,} decisions scored of {len(records):,} records "
        f"({len(records) - len(scorable):,} forced or unlabeled)"
    )
    print(f"  {len(per_game):,} source games, at most {max(per_game.values())} decision(s) each")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"  wrote {args.out_json}")
    key = f"{name}|{Path(out['suite']).name}"
    add_run(EXPERIMENT, key, out, result_rows, "Performance Rating", RESULT_HEADER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
