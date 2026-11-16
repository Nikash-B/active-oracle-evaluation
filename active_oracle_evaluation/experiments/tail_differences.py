"""Paired clustered bootstrap intervals on differences in tail statistics between agents.

Every agent is scored on the same frozen suite, so each replicate resamples the positions once and
recomputes both agents on that draw, which cancels position difficulty. Resampling is clustered by
source game through ``ScoreCache.cluster_ids``. Holm's correction is applied across all tests.

Statistics: the catastrophic-error rate at each threshold, and the upper percentiles of the regret
distribution. The results go to outputs/tail_differences/tail_differences.md and .json.

    python -m active_oracle_evaluation.experiments.tail_differences --n-boot 10000 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.score_cache import ScoreCache
from active_oracle_evaluation.eval.tail_risk import (
    bootstrap_p_two_sided,
    game_bootstrap_indices,
    holm_adjust,
    percentile_ci,
)
from active_oracle_evaluation.paths import DATA, output_dir, repo_relative
from active_oracle_evaluation.results import markdown_table

EXPERIMENT = "tail_differences"
DEFAULT_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
TABLE_HEADER = [
    "Statistic",
    "First",
    "Second",
    "Difference",
    "95% interval",
    "p",
    "Holm",
    "Resolved",
]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--n-boot", type=int, default=config["n_boot"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="JSON summary path (default outputs/tail_differences/tail_differences.json); "
        "tail_differences.md is written next to it",
    )
    return parser


def stats_of(regrets: np.ndarray, thresholds: list, percentiles: list) -> dict[str, float]:
    """Return the tail statistics of one regret vector."""
    out = {f"cat_gt_{t:g}": float((regrets > t).mean()) for t in thresholds}
    for q in percentiles:
        out[f"p{q}"] = float(np.percentile(regrets, q))
    return out


def compare_pair(regrets_a, regrets_b, a, b, indices, thresholds, percentiles):
    """Return the observed statistics, interval and p-value of every statistic for one pair."""
    obs_a = stats_of(regrets_a, thresholds, percentiles)
    obs_b = stats_of(regrets_b, thresholds, percentiles)
    reps = {k: np.empty(len(indices)) for k in obs_a}
    for i, idx in enumerate(indices):
        stats_a = stats_of(regrets_a[idx], thresholds, percentiles)
        stats_b = stats_of(regrets_b[idx], thresholds, percentiles)
        for k in reps:
            reps[k][i] = stats_a[k] - stats_b[k]
    entry = {}
    for k in obs_a:
        lo, hi = percentile_ci(reps[k])
        p_value = bootstrap_p_two_sided(reps[k])
        entry[k] = {
            a: obs_a[k],
            b: obs_b[k],
            "diff": obs_a[k] - obs_b[k],
            "lo": lo,
            "hi": hi,
            "p": p_value,
            "resolved": bool(lo > 0 or hi < 0),
        }
    return entry


def result_rows(a: str, b: str, entry: dict, adj: dict) -> list[list[str]]:
    """Return one row of formatted strings per statistic, as printed."""
    rows = []
    for k, v in entry.items():
        rows.append(
            [
                k,
                f"{v[a]:.4f}",
                f"{v[b]:.4f}",
                f"{v['diff']:+.4f}",
                f"[{v['lo']:+.4f}, {v['hi']:+.4f}]",
                f"{v['p']:.4f}",
                f"{adj[f'{a}-{b}:{k}']:.4f}",
                "RESOLVED" if v["resolved"] else "-",
            ]
        )
    return rows


def print_results(results: dict, adj: dict) -> list:
    """Print every pair's results and return the Markdown tables."""
    tables = []
    for pair, entry in results.items():
        a, b = pair.split(" - ")
        rows = result_rows(a, b, entry, adj)
        print(f"=== {pair} ===")
        for row in rows:
            print(
                f"  {row[0]:<12} {row[1]} vs {row[2]}   diff {row[3]} {row[4]}  "
                f"p={row[5]}  holm={row[6]}  {row[7]}"
            )
        print()
        tables.append((pair, rows))
    return tables


def write_markdown(path: Path, heading: str, tables: list) -> None:
    """Write the titled result tables of every pair."""
    parts = [f"# {EXPERIMENT}", heading]
    for pair, rows in tables:
        parts += [f"## {pair}", markdown_table(TABLE_HEADER, rows)]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    """Compute, print and write paired intervals for every agent pair and statistic."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    agent_pairs = [tuple(pair) for pair in config["agent_pairs"]]
    thresholds, percentiles = config["cat_thresholds"], config["tail_percentiles"]
    cache = ScoreCache(args.cache)

    missing = {agent for pair in agent_pairs for agent in pair} - set(cache.losses)
    if missing:
        raise SystemExit(f"cache lacks agent(s) {sorted(missing)}; has {cache.names}")

    heading = (
        f"{cache.n_pos:,} decisions, {cache.n_games:,} clusters "
        f"(mean size {cache.n_pos / cache.n_games:.2f}); "
        f"{args.n_boot:,} replicates, seed {args.seed}"
    )
    print(heading)
    # One index list, reused for every agent and every statistic, makes the comparison paired.
    indices = game_bootstrap_indices(cache.cluster_ids, n_boot=args.n_boot, seed=args.seed)

    n_tail = int(round(cache.n_pos * config["tail_note_fraction"]))
    print(
        f"NOTE: p{percentiles[-1]} on {cache.n_pos:,} decisions rests on ~{n_tail} points "
        "above the threshold.\n"
    )

    results, flat_p = {}, []
    for a, b in agent_pairs:
        entry = compare_pair(
            cache.losses[a], cache.losses[b], a, b, indices, thresholds, percentiles
        )
        flat_p += [(f"{a}-{b}:{k}", v["p"]) for k, v in entry.items()]
        results[f"{a} - {b}"] = entry

    adj = dict(zip([k for k, _ in flat_p], holm_adjust([p for _, p in flat_p])))
    tables = print_results(results, adj)

    out_json = args.out_json or output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "cache": repo_relative(args.cache),
        "n_decisions": cache.n_pos,
        "n_clusters": cache.n_games,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "p99_tail_points": n_tail,
        "pairs": results,
        "holm": adj,
    }
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    out_md = out_json.parent / f"{EXPERIMENT}.md"
    write_markdown(out_md, heading, tables)
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
