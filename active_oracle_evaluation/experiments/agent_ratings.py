"""Performance Rating (PR) and checker error per move (EPM) of each agent on its own games.

Plays a round-robin of duplicate-dice pairings among the given agents, logs every unforced
checker decision, scores the decisions with the gnubg-nn oracle and aggregates them by agent.
Writes a Markdown report (per-agent PR and EPM plus a head-to-head points-per-game matrix) and
a JSON file with the same numbers next to it.

Agent specs: ``random`` / ``heuristic`` / ``pubeval`` (built-ins) or ``label:path`` (a
checkpoint; the agent type is inferred from its head). Needs gnubg-nn (the ``[oracle]`` extra).
The learning-curve round-robin of the paper (AlphaZero after 25, 50 and 75 iterations):

    python -m active_oracle_evaluation.experiments.agent_ratings \
        --agents az25:models/az25.pt az50:models/az50.pt az75:models/az75.pt pubeval random \
        --pairs 60 --plies 0 --seed 20260628

Defaults come from the [agent_ratings] table of the settings.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import DiceBank, play_duplicate_pair, score_decisions
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "agent_ratings"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings."""
    config = settings(EXPERIMENT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        metavar="SPEC",
        help="random | heuristic | pubeval | label:path (>= 2)",
    )
    parser.add_argument(
        "--pairs", type=int, default=config["pairs"], help="duplicate-dice pairs per pairing"
    )
    parser.add_argument("--seed", type=int, default=config["seed"], help="dice-bank seed")
    parser.add_argument(
        "--plies", type=int, default=config["plies"], help="oracle eval depth (0 = fast bulk)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output Markdown file (default: outputs/agent_ratings/agent_ratings.md); "
        "the JSON file is written next to it with the suffix .json",
    )
    return parser


def agent_rows(names, summary, games_per_agent):
    """Return (agent, games, decisions, EPM, PR) per agent, sorted by PR, lowest first."""
    rows = []
    for name in names:
        agent_summary = summary.by_agent.get(name)
        rows.append(
            (
                name,
                games_per_agent[name],
                agent_summary.n_checker if agent_summary else 0,
                agent_summary.epm if agent_summary else 0.0,
                agent_summary.pr if agent_summary else 0.0,
            )
        )
    rows.sort(key=lambda r: r[4])
    return rows


def ppg_value(ppg, row_name, col_name):
    """Return the row agent's mean points per game minus the column agent's."""
    if (row_name, col_name) in ppg:
        return ppg[(row_name, col_name)]
    return -ppg[(col_name, row_name)]


def render(args, names, rows, summary, ppg, t_games, t_score, oracle_cfg) -> str:
    """Render the per-agent PR / EPM table and the head-to-head ppg matrix as Markdown."""
    lines = [
        "# Agent PR table",
        "",
        "Round-robin duplicate-dice + per-decision gnubg-oracle scoring. PR = performance "
        "rating; EPM = mean checker equity-loss per unforced move (millipoints); lower = better "
        "for both.",
        "",
        "## Setup",
        "",
        f"- Agents ({len(names)}): {', '.join(names)}; 1-ply head play.",
        f"- Round-robin: {len(names) * (len(names) - 1) // 2} pairings × {args.pairs} pairs; seed "
        f"{args.seed}. Oracle: gnubg-nn {oracle_cfg.plies}-ply cubeless.",
        f"- Wall: {t_games:.0f}s games + {t_score:.0f}s scoring (1 worker(s)).",
        "",
        "## PR / EPM by agent (sorted best → worst)",
        "",
        "| rank | agent | games | decisions | EPM (mpts) | PR |",
        "|---|---|---|---|---|---|",
    ]
    for rank, (name, n_games, n_decisions, epm, pr) in enumerate(rows, 1):
        lines.append(f"| {rank} | {name} | {n_games:,} | {n_decisions:,} | {epm:.1f} | {pr:.1f} |")
    lines += [
        "",
        "## Head-to-head ppg (row − column, dice/seat-controlled)",
        "",
        "| | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 1),
    ]
    for row_name in names:
        cells = []
        for col_name in names:
            if row_name == col_name:
                cells.append("—")
            else:
                cells.append(f"{ppg_value(ppg, row_name, col_name):+.3f}")
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"(Checker decisions only; cube decisions are not scored. "
        f"Total decisions: {summary.n_checker:,}.)",
        "",
    ]
    return "\n".join(lines)


def summary_numbers(args, names, rows, summary, ppg, oracle_cfg):
    """Return the report's numbers, rounded as printed, as a JSON-serialisable dict."""
    return {
        "agents": names,
        "pairs": args.pairs,
        "seed": args.seed,
        "oracle_plies": oracle_cfg.plies,
        "total_decisions": summary.n_checker,
        "by_agent": [
            {
                "rank": rank,
                "agent": name,
                "games": n_games,
                "decisions": n_decisions,
                "epm": float(f"{epm:.1f}"),
                "pr": float(f"{pr:.1f}"),
            }
            for rank, (name, n_games, n_decisions, epm, pr) in enumerate(rows, 1)
        ],
        "head_to_head_ppg": {
            row_name: {
                col_name: float(f"{ppg_value(ppg, row_name, col_name):+.3f}")
                for col_name in names
                if col_name != row_name
            }
            for row_name in names
        },
    }


def play_round_robin(agents, names, bank, pairs):
    """Play every pairing on duplicate dice; return decision logs, ppg and games per agent."""
    all_logs = []
    ppg: dict[tuple[str, str], float] = {}
    games_per_agent = {n: 0 for n in names}
    for (name_a, agent_a), (name_b, agent_b) in combinations(agents.items(), 2):
        result, logs = play_duplicate_pair(agent_a, agent_b, bank, pairs, log=True)
        for log in logs:
            all_logs.append(replace(log, agent=(name_a if log.agent == "a" else name_b)))
        ppg[(name_a, name_b)] = result.mean_diff  # name_a minus name_b
        games_per_agent[name_a] += pairs * 2
        games_per_agent[name_b] += pairs * 2
    return all_logs, ppg, games_per_agent


def main() -> None:
    """Play the round-robin, score every decision and write the report and its JSON."""
    args = build_parser().parse_args()
    encoder = BinarySlotEncoder()
    agents = dict(resolve_agent(s, encoder) for s in args.agents)
    names = list(agents)
    if len(names) < 2:
        raise SystemExit("need >= 2 agents")

    bank = DiceBank(args.seed)
    print(f"round-robin: {names}, {args.pairs} pairs/pairing ...")
    t0 = time.perf_counter()
    all_logs, ppg, games_per_agent = play_round_robin(agents, names, bank, args.pairs)
    t_games = time.perf_counter() - t0

    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig

    cfg = OracleConfig(plies=args.plies)
    oracle = GnubgOracle(cfg)
    print(f"scoring {len(all_logs):,} decisions, {args.plies}-ply, 1 worker(s) ...")
    t1 = time.perf_counter()
    summary = score_decisions(all_logs, oracle)
    t_score = time.perf_counter() - t1

    rows = agent_rows(names, summary, games_per_agent)
    report = render(args, names, rows, summary, ppg, t_games, t_score, oracle.cfg)
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    numbers = summary_numbers(args, names, rows, summary, ppg, oracle.cfg)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(numbers, indent=1) + "\n")
    print(f"\nwrote {out}\nwrote {json_path}\n")
    print(report)


if __name__ == "__main__":
    main()
