"""Markdown report of the Yahtzee rank-recovery experiment."""

from __future__ import annotations

from pathlib import Path

from active_oracle_evaluation.results import markdown_table
from active_oracle_evaluation.yahtzee.suite import DECISIONS_PER_GAME


def truth_table(results: dict) -> tuple[str, list[str], list[list[str]]]:
    """Return the table of exact per-game and per-decision deficits."""
    header = ["Agent", "Delta", f"Delta / {DECISIONS_PER_GAME}"]
    rows = [
        [name, f"{delta:.4f}", f"{per_decision:.4f}"]
        for name, delta, per_decision in zip(
            results["agents"], results["truth_delta"], results["truth_per_decision"]
        )
    ]
    return "Exact deficit per agent", header, rows


def estimand_summary_table(results: dict) -> tuple[str, list[str], list[list[str]]]:
    """Return the per-suite table of Kendall tau between full-suite regret and truth."""
    header = ["Suite", "Nodes", "Kendall tau (R_suite_full, truth)", "Discordant pairs"]
    rows = [
        [
            suite_name,
            f"{suite['n_pool']:,}",
            f"{suite['tau_estimand']:.4f}",
            str(len(suite["estimand_discordant_pairs"])),
        ]
        for suite_name, suite in results["suites"].items()
    ]
    return "Ranking by full-suite regret", header, rows


def estimand_gap_table(results: dict, suite_name: str) -> tuple[str, list[str], list[list[str]]]:
    """Return the per-agent table of full-suite regret, truth and their gap on one suite."""
    suite = results["suites"][suite_name]
    header = ["Agent", "R_suite_full", f"Truth / {DECISIONS_PER_GAME}", "Gap"]
    rows = [
        [name, f"{r_full:.4f}", f"{truth:.4f}", f"{gap:+.4f}"]
        for name, r_full, truth, gap in zip(
            results["agents"], suite["r_suite_full"], results["truth_per_decision"], suite["gap"]
        )
    ]
    return f"Estimand gap, {suite_name} suite", header, rows


def sampling_table(results: dict, suite_name: str) -> tuple[str, list[str], list[list[str]]]:
    """Return the per-rule, per-budget table of sampled rank recovery on one suite."""
    sampling = results["suites"][suite_name]["sampling"]
    header = ["Rule", "K", "Mean tau", "Tau interval", "Top-1", "Hajek MAE", "Tie rate"]
    rows = []
    for rule, by_budget in sampling.items():
        for k, cell in by_budget.items():
            rows.append(
                [
                    rule,
                    k,
                    f"{cell['tau_mean']:.4f}",
                    f"[{cell['tau_lo']:.3f},{cell['tau_hi']:.3f}]",
                    f"{cell['top1']:.3f}",
                    f"{cell['hajek_mae']:.4f}",
                    f"{cell['tie_rate']:.2f}",
                ]
            )
    title = f"Rank recovery from {results['seeds']} draws, {suite_name} suite"
    return title, header, rows


def margin_table(results: dict, suite_name: str) -> tuple[str, list[str], list[list[str]]]:
    """Return the per-budget table of one rule's tau lead over the baseline on one suite."""
    margin = results["suites"][suite_name]["selection_margin"]
    header = ["K", "Tau difference", "Outcome", "Label ratio"]
    rows = [
        [
            str(row["K"]),
            f"{row['tau_difference']:+.4f}",
            row["outcome"],
            "–" if row["label_ratio"] is None else f"{row['label_ratio']:.2f}",
        ]
        for row in margin["by_budget"]
    ]
    title = f"{margin['rule']} minus {margin['baseline']}, {suite_name} suite"
    return title, header, rows


def write_report(results: dict, path: Path, title: str) -> None:
    """Write the result tables of one run as a Markdown file."""
    tables = [truth_table(results), estimand_summary_table(results)]
    tables += [estimand_gap_table(results, name) for name in results["suites"]]
    tables += [sampling_table(results, name) for name in results["suites"]]
    tables += [margin_table(results, name) for name in results["suites"]]
    parts = [f"# {title}"]
    for table_title, header, rows in tables:
        parts += [f"## {table_title}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
