"""Count the suite decisions that fall inside the exact bear-off table's domain.

The table needs both sides fully home, nothing on the bar and at least one checker borne off on
each side, so that gammons are impossible and the race factorises. This is narrower than the
``bearoff`` phase label, which requires only the mover to be home. The experiment writes
``bearoff_domain.md`` (counts, phase mix and source games) and ``bearoff_domain.json``.

    python -m active_oracle_evaluation.experiments.bearoff_domain \
        --suite data/decision_suites/backgammon_oracle_suite_0ply_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from active_oracle_evaluation.backgammon.core.types import Player
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.decision_suite import deserialize_board
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import markdown_table

EXPERIMENT = "bearoff_domain"
DEFAULT_SUITE = DATA / "decision_suites" / "backgammon_oracle_suite_0ply_v1.jsonl"


def in_domain(board, *, min_off: int) -> bool:
    """Return whether both sides are home, the bar is empty and each side has ``min_off`` off."""
    for player in (Player.WHITE, Player.BLACK):
        if board.bar(player) > 0 or not board.all_in_home(player):
            return False
        if board.off(player) < min_off:
            return False
    return True


def source_games(records) -> collections.Counter:
    """Count decisions per source game, keyed by ``(agent_source, game_id)``.

    Each per-agent pool numbers its games from 0, so ``game_id`` alone does not identify a game.
    """
    return collections.Counter(
        (record.get("agent_source"), record.get("metadata", {}).get("game_id"))
        for record in records
    )


def count_domain(records, *, min_off: int, phase_label: str) -> dict:
    """Return the suite, label, both-home and domain counts plus the domain games and phases."""
    boards = [(record, deserialize_board(record["board"])) for record in records]
    labelled = [record for record, _ in boards if record.get("phase") == phase_label]
    both_home = [record for record, board in boards if in_domain(board, min_off=0)]
    domain = [record for record, board in boards if in_domain(board, min_off=min_off)]
    games = source_games(domain)
    return {
        "decisions": len(records),
        "labelled": len(labelled),
        "both_home": len(both_home),
        "domain": len(domain),
        "source_games": len(games),
        "max_per_game": max(games.values(), default=0),
        "phase_mix": dict(collections.Counter(record.get("phase") for record in domain)),
    }


def report_lines(suite_name: str, counts: dict, phase_label: str) -> list[str]:
    """Return the Markdown report for one suite's domain counts."""
    pct = 100.0 * counts["domain"] / counts["decisions"]
    labelled_pct = 100.0 * counts["domain"] / max(1, counts["labelled"])
    rows = [
        ["whole suite", f"{counts['decisions']:,}"],
        [f"labelled `{phase_label}` phase", f"{counts['labelled']:,}"],
        ["both sides home, nothing on the bar", f"{counts['both_home']:,}"],
        ["exact bear-off domain (both home, ≥1 checker off each)", f"{counts['domain']:,}"],
    ]
    return [
        "# Exact bear-off coverage of the decision suite",
        "",
        f"Suite: `{suite_name}`.",
        "",
        markdown_table(["set", "decisions"], rows),
        "",
        f"{counts['domain']} of {counts['decisions']:,} decisions — {pct:.1f}% of the suite — "
        f"from {counts['source_games']} source games.",
        "",
        f"Phase mix of the in-domain set: {counts['phase_mix']}.",
        "",
        f"The phase label requires only the mover to be home and allows an opposing back "
        "checker, whereas the table requires both sides home with a checker already off. "
        f"{labelled_pct:.0f}% of the {counts['labelled']} labelled `{phase_label}` decisions "
        "qualify.",
        "",
        f"Resampling units: {counts['source_games']} source games, at most "
        f"{counts['max_per_game']} decision(s) each.",
    ]


def summary(suite_name: str, counts: dict) -> dict:
    """Return the JSON summary of one suite's domain counts."""
    return {
        "suite": suite_name,
        **counts,
        "domain_pct": round(100.0 * counts["domain"] / counts["decisions"], 1),
        "labelled_qualify_pct": round(100.0 * counts["domain"] / max(1, counts["labelled"])),
    }


def main() -> None:
    """Write the domain-coverage report and JSON summary for one suite."""
    config = settings(EXPERIMENT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report path (default outputs/bearoff_domain/bearoff_domain.md); "
        "the JSON summary is written beside it",
    )
    parser.add_argument("--min-checkers-off", type=int, default=config["min_checkers_off"])
    parser.add_argument("--phase-label", default=config["phase_label"])
    args = parser.parse_args()
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"

    records = [json.loads(line) for line in args.suite.read_text().splitlines() if line.strip()]
    counts = count_domain(records, min_off=args.min_checkers_off, phase_label=args.phase_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report_lines(args.suite.name, counts, args.phase_label)) + "\n")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(summary(args.suite.name, counts), indent=1) + "\n")
    pct = 100.0 * counts["domain"] / counts["decisions"]
    print(
        f"{counts['domain']}/{counts['decisions']} in domain ({pct:.1f}%), "
        f"{counts['source_games']} source games -> {out}"
    )


if __name__ == "__main__":
    main()
