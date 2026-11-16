"""Agent-panel PR table on the paired panel suite.

Builds the panel suite from seeded self-play of the sampler agents, labels every position with
the gnubg-nn oracle and scores every panel agent on the same positions. For each position the
oracle's mover-equity over all legal moves is the ground truth, and an agent loses
``best - chosen``. Because the positions are shared, between-agent PR differences are paired
per-position contrasts in which the oracle's selection bias and the between-position variance
cancel. The report gives each agent's PR with a bootstrap CI, simultaneous rank intervals, the
adjacent paired contrasts and their Holm-corrected p-values. It also gives the Kendall tau
between the panel's PR order and the anchored-Elo order of the shipped round-robin
(data/round_robins/elo_validity.json), with the pairs the two order differently, and, with
``--compare-to``, the tau against the PR order of another panel_ratings run (for example the
0-ply run when this one is at 2 ply). It is written to outputs/panel_ratings/panel_ratings.md,
with the same numbers in panel_ratings.json.

The paper's settings (the [panel_ratings] table of config/experiments.toml) are the defaults:

    python -m active_oracle_evaluation.experiments.panel_ratings \
        --agents random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt \
            az:models/az75.pt distilled:models/distilled.pt \
        --samplers heuristic v2:models/v2.pt --positions 1000 --seed 20260628 --plies 0

The 2-ply run compares its order with the 0-ply run's JSON:

    python -m active_oracle_evaluation.experiments.panel_ratings --plies 2 \
        --out outputs/panel_ratings/panel_ratings_2ply.md \
        --out-json outputs/panel_ratings/panel_ratings_2ply.json \
        --compare-to outputs/panel_ratings/panel_ratings.json
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.suite import (
    PositionSuite,
    build_position_suite,
    holm_contrasts,
    paired_contrast,
    pr_ci,
    score_agent_on_suite,
    simultaneous_rank_intervals,
)
from active_oracle_evaluation.eval.validity import kendall_tau
from active_oracle_evaluation.experiments.panel_ratings_report import (
    agreement_lines,
    render,
    summary_numbers,
    write_panel_csv,
)
from active_oracle_evaluation.paths import DATA, output_dir, repo_relative

EXPERIMENT = "panel_ratings"
ROUND_ROBIN_JSON = DATA / "round_robins" / "elo_validity.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, with defaults from the experiment's settings."""
    cfg = settings(EXPERIMENT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agents", nargs="+", default=list(cfg["agents"]), metavar="SPEC", help="agents (>=2)"
    )
    p.add_argument(
        "--samplers",
        nargs="+",
        default=list(cfg["samplers"]),
        help="agents whose self-play generates the held-out positions (a style mix)",
    )
    p.add_argument(
        "--positions", type=int, default=cfg["positions"], help="suite size (split across samplers)"
    )
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--plies", type=int, default=cfg["plies"], help="oracle ground-truth depth")
    p.add_argument(
        "--out", type=Path, default=None, help="Markdown report (default: outputs/panel_ratings/)"
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="JSON summary of the report's numbers (default: outputs/panel_ratings/)",
    )
    p.add_argument(
        "--dump-suite",
        type=Path,
        default=None,
        metavar="JSONL",
        help="write the built suite as labeled decision records, scoreable without the "
        "oracle or the samplers",
    )
    p.add_argument(
        "--out-panel-csv",
        type=Path,
        default=None,
        help="also write each agent's PR, CI and rank interval as CSV",
    )
    p.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        metavar="JSON",
        help="JSON of another panel_ratings run; report the Kendall tau between its PR order "
        "and this run's",
    )
    return p


def order_agreement(order: list[str], reference: list[str]) -> dict:
    """Kendall tau between two best-first orders, with the pairs they place differently."""
    rank = {name: i for i, name in enumerate(reference)}
    shared = [name for name in order if name in rank]
    discordant = [[a, b] for a, b in combinations(shared, 2) if rank[a] > rank[b]]
    return {"kendall_tau": round(kendall_tau(order, reference), 3), "discordant": discordant}


def elo_order(path: Path) -> list[str]:
    """Agents of a round-robin JSON, highest anchored Elo first."""
    elo = json.loads(path.read_text())["elo"]
    return sorted(elo, key=lambda name: -elo[name])


def panel_order(path: Path) -> list[str]:
    """Agents of a panel_ratings JSON, best PR rank first."""
    agents = json.loads(path.read_text())["agents"]
    return [a["agent"] for a in sorted(agents, key=lambda a: a["rank"])]


def agreements(order: list[str], compare_to: Path | None) -> dict:
    """Agreement of the PR order with anchored Elo and, if given, with another run."""
    out = {
        "anchored_elo": {
            "source": repo_relative(ROUND_ROBIN_JSON),
            **order_agreement(order, elo_order(ROUND_ROBIN_JSON)),
        }
    }
    if compare_to is not None:
        out["compared_run"] = {
            "source": repo_relative(compare_to),
            **order_agreement(order, panel_order(compare_to)),
        }
    return out


def build_merged_suite(
    samplers, n: int, oracle, seed: int, seed_stride: int
) -> tuple[PositionSuite, list[str]]:
    """Pool ``n`` positions across the sampler agents, one position per game.

    Also returns the sampler name behind each position, aligned to ``suite.positions``.
    """
    per = max(1, n // len(samplers))
    positions: list = []
    sources: list[str] = []
    for i, (name, agent) in enumerate(samplers):
        sub = build_position_suite(
            per, oracle=oracle, seed=seed + seed_stride * (i + 1), sampler=agent
        )
        positions.extend(sub.positions)
        sources.extend([name] * len(sub.positions))
    ver = getattr(getattr(oracle, "cfg", None), "version", "") or ""
    return PositionSuite(tuple(positions), seed=seed, oracle_version=ver), sources


def dump_suite_jsonl(
    suite: PositionSuite,
    sources: list[str],
    path: Path,
    *,
    sampler_specs: list[str],
    plies: int,
    suite_label: str,
) -> int:
    """Write the panel suite as labeled decision records, scoreable without gnubg.

    Uses the decision-suite JSONL schema. ``game_id`` counts within a sampler, so the clustered
    bootstrap's unit ``(agent_source, game_id)`` is unique per game.
    """
    from active_oracle_evaluation.eval.decision_suite import record_from_suite_position, write_jsonl

    per_source: dict[str, int] = {}
    records = []
    for pos, src in zip(suite.positions, sources, strict=True):
        gid = per_source.get(src, 0)
        per_source[src] = gid + 1
        records.append(
            record_from_suite_position(
                pos,
                agent_source=src,
                metadata={
                    "seed": suite.seed,
                    "game_id": gid,
                    "sampler": src,
                    "samplers": list(sampler_specs),
                    "oracle_version": suite.oracle_version,
                    "oracle_depth": int(plies),
                    "suite": suite_label,
                },
            )
        )
    write_jsonl(records, path)
    return len(records)


def repeat_draws(agents, suite, seed: int, n_draws: int) -> dict[str, list[float]]:
    """Re-score every agent on the same suite under n_draws other tie-break seeds -> PRs."""
    return {
        name: [
            score_agent_on_suite(agent, suite, label=name, seed=seed + k).pr
            for k in range(1, n_draws + 1)
        ]
        for name, agent in agents
    }


def repeat_lines(scores, repeats: dict[str, list[float]]) -> str:
    """Render each agent's PR range over the repeat draws and whether the rank order changes."""
    n_draws = len(next(iter(repeats.values())))
    if n_draws == 0:
        return ""
    order = [s.agent for s in scores]
    same_order = all(sorted(order, key=lambda a: repeats[a][k]) == order for k in range(n_draws))
    lines = [
        "",
        f"## Repeat evaluations ({n_draws} other tie-break draws on the same suite)",
        "",
        "| agent | PR | PR over the other draws | changes |",
        "|---|---|---|---|",
    ]
    for s in scores:
        r = repeats[s.agent]
        changes = "yes" if any(abs(x - s.pr) > 1e-9 for x in r) else "no"
        lines.append(f"| {s.agent} | {s.pr:.1f} | {min(r):.1f} to {max(r):.1f} | {changes} |")
    lines += ["", f"Rank order unchanged in every draw: {'yes' if same_order else 'no'}.", ""]
    return "\n".join(lines)


def main() -> None:
    """Build the suite, score every agent and write the report (and optional suite/CSV)."""
    args = build_parser().parse_args()
    cfg = settings(EXPERIMENT)
    n_boot = cfg["n_boot"]
    encoder = BinarySlotEncoder()
    agents = [resolve_agent(s, encoder) for s in args.agents]
    samplers = [resolve_agent(s, encoder) for s in args.samplers]
    if len(agents) < 2:
        raise SystemExit("need >= 2 agents")

    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig  # lazy

    oracle = GnubgOracle(OracleConfig(plies=args.plies))
    sampler_names = [n for n, _ in samplers]
    print(f"building suite: {args.positions} positions from {sampler_names} ...")
    t0 = time.perf_counter()
    suite, sources = build_merged_suite(
        samplers, args.positions, oracle, args.seed, cfg["sampler_seed_stride"]
    )
    t_build = time.perf_counter() - t0

    if args.dump_suite is not None:
        n = dump_suite_jsonl(
            suite,
            sources,
            args.dump_suite,
            sampler_specs=args.samplers,
            plies=args.plies,
            suite_label=cfg["suite_label"],
        )
        print(f"wrote {args.dump_suite} ({n} labeled records; scoreable without gnubg)")

    print(f"scoring {len(agents)} agents on {len(suite)} positions ...")
    t1 = time.perf_counter()
    scores = [
        score_agent_on_suite(agent, suite, label=name, seed=args.seed) for name, agent in agents
    ]
    t_score = time.perf_counter() - t1

    scores.sort(key=lambda s: s.pr)
    cis = {s.agent: pr_ci(s, n_boot=n_boot, seed=args.seed) for s in scores}
    # Worse minus better, so each delta-PR is positive.
    contrasts = [
        paired_contrast(scores[i + 1], scores[i], n_boot=n_boot, seed=args.seed)
        for i in range(len(scores) - 1)
    ]
    ranks = simultaneous_rank_intervals(scores, n_boot=n_boot, seed=args.seed)
    adjacent_pairs = [(scores[i + 1].agent, scores[i].agent) for i in range(len(scores) - 1)]
    holm = holm_contrasts(scores, adjacent_pairs, n_boot=n_boot, seed=args.seed)
    agreement = agreements([s.agent for s in scores], args.compare_to)
    repeats = repeat_draws(agents, suite, args.seed, cfg["repeat_draws"])

    report = render(
        args, scores, cis, contrasts, suite, sampler_names, t_build, t_score, ranks, holm, n_boot
    )
    report += agreement_lines(agreement)
    report += repeat_lines(scores, repeats)
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    out_json = args.out_json
    if out_json is None:
        out_json = output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    if args.out_panel_csv is not None:
        write_panel_csv(args.out_panel_csv, scores, cis, ranks)
    numbers = summary_numbers(args, scores, cis, contrasts, ranks, holm, len(suite))
    numbers["order_agreement"] = agreement
    numbers["repeat_draws"] = repeats
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(numbers, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"\nwrote {out}\n")
    print(report)


if __name__ == "__main__":
    main()
