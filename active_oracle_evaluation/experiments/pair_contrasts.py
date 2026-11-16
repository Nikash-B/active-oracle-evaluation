"""Paired PR contrasts between agent pairs on the shipped score caches.

Each contrast is a first-minus-second PR difference on the same decisions, with a 95% paired
bootstrap interval over decisions (eval.suite.paired_contrast). Every row uses the same bootstrap
replicates and seed. PR is lower-is-better, so a negative difference places the first agent ahead.
The rows are pubeval against the warm-start network (v2init) on the 2,400-decision suite and on
the panel suite, the AlphaZero checkpoints after 25, 50 and 75 iterations, and the same
contrasts with the suite re-labelled at 2 ply. The experiment also reports the PR order of the
AlphaZero checkpoints in each cache.

    python -m active_oracle_evaluation.experiments.pair_contrasts

The results go to outputs/pair_contrasts/pair_contrasts.md and pair_contrasts.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY, ScoreCache
from active_oracle_evaluation.eval.suite import SuiteScore, paired_contrast
from active_oracle_evaluation.paths import DATA, repo_relative
from active_oracle_evaluation.results import write_results

EXPERIMENT = "pair_contrasts"
DEFAULT_CACHE_DIR = DATA / "decision_suites"


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="folder of the score caches"
    )
    parser.add_argument(
        "--n-boot", type=int, default=config["n_boot"], help="bootstrap replicates per interval"
    )
    parser.add_argument("--seed", type=int, default=config["seed"], help="bootstrap seed")
    return parser


def pr_of(cache, name):
    """Return one agent's PR in a cache, computed from its regrets rather than a rounded sidecar."""
    return float(cache.losses[name].mean() * PR_PER_EQUITY)


def contrast_row(cache, first, second, n_boot, seed):
    """Return the paired PR contrast of ``first`` minus ``second`` on one cache as a dict."""
    scores = [SuiteScore(name, tuple(cache.losses[name])) for name in (first, second)]
    contrast = paired_contrast(scores[0], scores[1], n_boot=n_boot, seed=seed)
    return {
        "cache": cache.path.name,
        "a": first,
        "b": second,
        "n_decisions": cache.n_pos,
        "pr_a": scores[0].pr,
        "pr_b": scores[1].pr,
        "delta_pr": contrast.delta_pr,
        "ci_low": contrast.ci_low,
        "ci_high": contrast.ci_high,
        "excludes_zero": bool(contrast.distinguishable),
    }


def order_row(cache, agents):
    """Return the PR of each agent in one cache and the agents sorted best first."""
    pr = {name: pr_of(cache, name) for name in agents}
    return {"cache": cache.path.name, "pr": pr, "order": sorted(agents, key=pr.__getitem__)}


def tables_of(contrasts, orders, decimals):
    """Return the Markdown tables of the contrasts and of the PR orders."""
    signed = f"{{:+.{decimals}f}}"
    contrast_rows = [
        [
            row["cache"],
            row["n_decisions"],
            f"{row['a']} - {row['b']}",
            signed.format(row["delta_pr"]),
            f"[{signed.format(row['ci_low'])}, {signed.format(row['ci_high'])}]",
            "yes" if row["excludes_zero"] else "no",
        ]
        for row in contrasts
    ]
    order_rows = [
        [
            row["cache"],
            ", ".join(f"{name} {pr:.{decimals}f}" for name, pr in row["pr"].items()),
            " < ".join(row["order"]),
        ]
        for row in orders
    ]
    contrast_header = ["cache", "decisions", "contrast", "delta PR", "95% interval", "excludes 0"]
    return [
        ("Paired PR contrasts (first minus second)", contrast_header, contrast_rows),
        ("PR order (lower is better)", ["cache", "PR", "order"], order_rows),
    ]


def main() -> None:
    """Compute every configured contrast and PR order and write the results."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    names = {name for name, _, _ in config["contrasts"]} | set(config["order_caches"])
    caches = {name: ScoreCache(args.cache_dir / name) for name in sorted(names)}
    contrasts = [
        contrast_row(caches[name], first, second, args.n_boot, args.seed)
        for name, first, second in config["contrasts"]
    ]
    orders = [order_row(caches[name], config["order_agents"]) for name in config["order_caches"]]
    decimals = config["decimals"]
    for row in contrasts:
        print(
            f"{row['cache']}: {row['a']} - {row['b']} = {row['delta_pr']:+.{decimals}f} "
            f"[{row['ci_low']:+.{decimals}f}, {row['ci_high']:+.{decimals}f}]"
        )
    for row in orders:
        print(f"{row['cache']}: order {' < '.join(row['order'])}")
    numbers = {
        "cache_dir": repo_relative(args.cache_dir),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "contrasts": contrasts,
        "orders": orders,
    }
    md_path, json_path = write_results(EXPERIMENT, tables_of(contrasts, orders, decimals), numbers)
    print(f"wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
