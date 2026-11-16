r"""Tail statistics and per-phase regret of every agent, from the cached suite scores.

Loads the per-agent regret cache and, for every agent, reports the tail of the regret
distribution (mean, median, p90, p95, p99, with a decision-level bootstrap interval on p95),
the catastrophic-error rates (regret above 0.05, 0.10 and 0.20 equity) and PR, plus a
per-(agent, phase) table of mean regret with game-clustered bootstrap intervals and p95. No agent
is re-scored.

Writes, in outputs/tail_regret/ (or --out-dir):
    tail_risk_table.csv        one row per agent
    phase_breakdown_table.csv  one row per (agent, phase)
    tail_regret.md, tail_regret.json  the result tables

The paper's settings (0-ply suite; pass full_suite_scores_2ply.npz for the 2-ply re-score):

    python -m active_oracle_evaluation.experiments.tail_regret \
        --cache data/decision_suites/full_suite_scores_0ply.npz --n-boot 2000 --boot-seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.score_cache import ScoreCache
from active_oracle_evaluation.eval.tail_risk import (
    bootstrap_p95_ci,
    catastrophic_rate,
    game_bootstrap_indices,
    percentile_ci,
    phase_breakdown,
    phase_mean_replicates,
    tail_stats,
)
from active_oracle_evaluation.experiments.tail_regret_report import report_tables
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "tail_regret"
DEFAULT_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
PHASE_COLUMNS = ["agent", "phase", "n", "n_games", "mean", "mean_ci_lo", "mean_ci_hi", "p95"]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="per-agent regret score cache (.npz with names/phases/losses/...)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=config["n_boot"], help="clustered bootstrap replicates"
    )
    parser.add_argument(
        "--boot-seed", type=int, default=config["boot_seed"], help="clustered bootstrap seed"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="output directory (default outputs/tail_regret)"
    )
    return parser


def cat_columns(thresholds: tuple[float, ...]) -> list[str]:
    """Return the CSV column name of each catastrophic-error threshold."""
    return [f"cat_gt_{t:g}" for t in thresholds]


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as a CSV file with the given columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tail_table(path: Path, cache: ScoreCache, config: dict) -> list[dict]:
    """Write the per-agent tail table (percentiles in equity) and return its rows."""
    thresholds = tuple(config["catastrophic_thresholds"])
    rows = []
    for name in cache.names:
        regrets = cache.losses[name]
        stats = tail_stats(regrets)
        cat = catastrophic_rate(regrets, thresholds)
        ci_lo, ci_hi = bootstrap_p95_ci(
            regrets, n_boot=config["p95_n_boot"], alpha=config["alpha"], seed=config["p95_seed"]
        )
        row = {
            "agent": name,
            "n": int(regrets.size),
            "mean": f"{stats['mean']:.6f}",
            "median": f"{stats['median']:.6f}",
            "p90": f"{stats['p90']:.6f}",
            "p95": f"{stats['p95']:.6f}",
            "p99": f"{stats['p99']:.6f}",
            "p95_ci_lo": f"{ci_lo:.6f}",
            "p95_ci_hi": f"{ci_hi:.6f}",
            **{f"cat_gt_{t:g}": f"{cat[t]:.4f}" for t in thresholds},
            "pr": f"{cache.pr[name]:.1f}",
            "cube_epc": "n/a",  # the checker-only suite has no cube decisions
        }
        rows.append(row)
    columns = [
        "agent",
        "n",
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "p95_ci_lo",
        "p95_ci_hi",
        *cat_columns(thresholds),
        "pr",
        "cube_epc",
    ]
    write_csv(path, columns, rows)
    return rows


def write_phase_table(
    path: Path, cache: ScoreCache, *, n_boot: int, seed: int, alpha: float
) -> list[dict]:
    """Write the per-(agent, phase) regret table with game-clustered intervals; return its rows.

    ``mean_ci_lo``/``mean_ci_hi`` bound the per-phase mean with a bootstrap that resamples whole
    games, keyed on ``cache.cluster_ids`` (the ``(agent_source, game_id)`` pair). ``n_games`` is
    the number of source games behind the phase.
    """
    indices = game_bootstrap_indices(cache.cluster_ids, n_boot=n_boot, seed=seed)
    rows = []
    for name in cache.names:
        breakdown = phase_breakdown(cache.losses[name], cache.phases)
        phases = sorted(breakdown)
        replicates = phase_mean_replicates(cache.losses[name], cache.phases, indices, phases)
        for j, phase in enumerate(phases):
            lo, hi = percentile_ci(replicates[:, j], alpha=alpha)
            n_games = int(len(np.unique(cache.cluster_ids[cache.phases == phase])))
            rows.append(
                {
                    "agent": name,
                    "phase": phase,
                    "n": breakdown[phase]["n"],
                    "n_games": n_games,
                    "mean": f"{breakdown[phase]['mean']:.6f}",
                    "mean_ci_lo": f"{lo:.6f}",
                    "mean_ci_hi": f"{hi:.6f}",
                    "p95": f"{breakdown[phase]['p95']:.6f}",
                }
            )
    write_csv(path, PHASE_COLUMNS, rows)
    return rows


def run(cache: ScoreCache, args, config: dict) -> None:
    """Write both CSV tables and the Markdown and JSON results."""
    out_dir = args.out_dir if args.out_dir is not None else output_dir(EXPERIMENT)
    tail_path = out_dir / "tail_risk_table.csv"
    phase_path = out_dir / "phase_breakdown_table.csv"
    tail_rows = write_tail_table(tail_path, cache, config)
    phase_rows = write_phase_table(
        phase_path, cache, n_boot=args.n_boot, seed=args.boot_seed, alpha=config["alpha"]
    )
    print(f"wrote {tail_path}")
    print(f"wrote {phase_path}")
    thresholds = tuple(config["catastrophic_thresholds"])
    tables, numbers = report_tables(
        cache, tail_rows, phase_rows, cat_columns(thresholds), config["similar_mean_tol"]
    )
    md_path, json_path = write_results(EXPERIMENT, tables, numbers, out_dir=out_dir)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


def main() -> None:
    """Load the score cache and write the tail-regret outputs."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    if not args.cache.exists():
        raise SystemExit(f"score cache not found: {args.cache}")
    cache = ScoreCache(args.cache)
    print(
        f"[full] {cache.n_pos:,} decisions from {cache.n_games} games, "
        f"{len(cache.names)} agents: {', '.join(cache.names)}"
    )
    run(cache, args, config)
    print("done.")


if __name__ == "__main__":
    main()
