"""Trials to resolve a skill difference: outcome scoring against per-decision scoring.

Yahtzee's optimal policy is solvable, so the efficiency of per-decision scoring over outcome
scoring can be computed rather than estimated against an imperfect oracle. Every quantity below
is exact from the DP or comes from on-policy simulation whose mean is checked against an exact
value before use.

To decide which of two agents is better:

* **outcome scoring** compares mean final score, with the score SD as noise (59.6 points at
  optimal play);
* **per-decision scoring** compares mean per-game oracle regret. By the performance-difference
  lemma its expectation is ``V*(start) - V^pi(start)``, so it estimates the same gap with the
  regret SD as noise.

Both estimate the same quantity, so the ratio of required trials is a variance ratio:

    n = z^2 * (var_i + var_j) / gap^2        ->   ratio = (score vars) / (regret vars)

Trial counts are written for two criteria: the 95% interval excludes zero, and 80% power at
two-sided 5%. The second multiplies every count by ``((1.96 + 0.8416) / 1.96)^2 = 2.04``.

    python -m active_oracle_evaluation.experiments.yahtzee_trials --games 5000

The run writes yahtzee_trials_to_resolve.csv (one row per agent pair), yahtzee_trials.md and
yahtzee_trials.json to outputs/yahtzee_trials/.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import write_results
from active_oracle_evaluation.yahtzee import OPTIMAL_EXPECTED_SCORE, OPTIMAL_SCORE_SD
from active_oracle_evaluation.yahtzee.agents import build_panel
from active_oracle_evaluation.yahtzee.dp import START, solve_optimal
from active_oracle_evaluation.yahtzee.suite import simulate_on_policy

EXPERIMENT = "yahtzee_trials"
CSV_NAME = "yahtzee_trials_to_resolve.csv"
AGENT_HEADER = ["agent", "Delta", "sd_score", "sd_regret", "z_score", "z_regret"]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings file."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--games", type=int, default=config["games"], help="on-policy games per agent")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=list(config["exclude"]),
        help="agents to drop. `fixed_order` ranks categories without looking at the dice, so its "
        "options tie almost everywhere; the exact evaluator and the simulator break those ties "
        "differently and describe different policies.",
    )
    p.add_argument("--seed", type=int, default=config["seed"], help="simulation seed")
    p.add_argument(
        "--panel-seed", type=int, default=config["panel_seed"], help="seed of the agent panel"
    )
    p.add_argument("--vstar", type=Path, default=None, help="cached optimal table (.npy)")
    p.add_argument(
        "--panel-json",
        type=Path,
        default=DATA / "yahtzee" / "yahtzee_panel.json",
        help="exact panel evaluation written by "
        "python -m active_oracle_evaluation.experiments.yahtzee_panel; supplies the gap to "
        "optimal and the exact score SD",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=f"per-pair trial counts and variance ratios (default outputs/{EXPERIMENT}/{CSV_NAME})",
    )
    return p


def load_panel(args, config: dict):
    """Solve or load V*, check its anchor and return it with the panel minus excluded agents."""
    vstar = np.load(args.vstar) if args.vstar and args.vstar.exists() else solve_optimal()
    assert abs(vstar[START] - OPTIMAL_EXPECTED_SCORE) < config["v_star_tolerance"], vstar[START]
    panel = {
        k: v
        for k, v in build_panel(vstar, seed=args.panel_seed).items()
        if k not in set(args.exclude)
    }
    if args.exclude:
        print(f"excluded: {', '.join(args.exclude)} (see --help)\n")
    return vstar, panel


def measure_agent(vstar, spec, exact_row, n_games: int, seed: int) -> dict:
    """Simulate one agent on-policy, check both means against exact values, return its SDs."""
    delta = float(exact_row["gap_vs_optimal"])
    expected_score = float(exact_row["expected_score"])
    sd_score_exact = float(exact_row["sd"])
    sim = simulate_on_policy(vstar, spec["vhat"], spec["preference"], n_games, seed=seed)
    n = n_games
    # Both means are exact in expectation; check the simulator against them before use.
    z_mean = (sim["score"].mean() - expected_score) / (sim["score"].std(ddof=1) / np.sqrt(n))
    z_reg = (
        (sim["regret"].mean() - delta) / (sim["regret"].std(ddof=1) / np.sqrt(n))
        if sim["regret"].std(ddof=1) > 0
        else 0.0
    )
    return {
        "delta": delta,
        "sd_score": sd_score_exact,
        "sd_regret": float(sim["regret"].std(ddof=1)),
        "z_score": z_mean,
        "z_regret": z_reg,
    }


def measure_panel(vstar, panel: dict, exact: dict, args, config: dict):
    """Measure every agent in panel order; print and return the stats and the table rows."""
    z_limit = config["z_limit"]
    stat: dict[str, dict] = {}
    table_rows = []
    print(f"{'agent':<14}{'Delta':>9}{'sd_score':>10}{'sd_regret':>11}{'checks':>28}")
    for i, (name, spec) in enumerate(panel.items()):
        measured = measure_agent(vstar, spec, exact[name], args.games, args.seed + i)
        z_mean, z_reg = measured.pop("z_score"), measured.pop("z_regret")
        assert abs(z_mean) < z_limit, (name, "score mean off the exact value", z_mean)
        assert abs(z_reg) < z_limit, (
            name,
            "regret mean off the performance-difference lemma",
            z_reg,
        )
        stat[name] = measured
        delta, sd_score_exact = measured["delta"], measured["sd_score"]
        print(
            f"{name:<14}{delta:>9.4f}{sd_score_exact:>10.3f}{stat[name]['sd_regret']:>11.3f}"
            f"{f'  z_score={z_mean:+.2f} z_regret={z_reg:+.2f}':>28}",
            flush=True,
        )
        table_rows.append(
            [
                name,
                f"{delta:.4f}",
                f"{sd_score_exact:.3f}",
                f"{stat[name]['sd_regret']:.3f}",
                f"{z_mean:+.2f}",
                f"{z_reg:+.2f}",
            ]
        )
        stat[name]["z_score"] = float(z_mean)
        stat[name]["z_regret"] = float(z_reg)
    return stat, table_rows


def pair_rows(stat: dict, z95, power80) -> list[dict]:
    """Return the trial counts and variance ratio of every agent pair."""
    rows = []
    for a, b in itertools.combinations(stat, 2):
        sa, sb = stat[a], stat[b]
        gap = abs(sa["delta"] - sb["delta"])
        n_out = (z95**2) * (sa["sd_score"] ** 2 + sb["sd_score"] ** 2) / gap**2
        n_dec = (z95**2) * (sa["sd_regret"] ** 2 + sb["sd_regret"] ** 2) / gap**2
        rows.append(
            {
                "agent_a": a,
                "agent_b": b,
                "gap_points_per_game": round(gap, 6),
                "sd_score_a": round(sa["sd_score"], 4),
                "sd_score_b": round(sb["sd_score"], 4),
                "sd_regret_a": round(sa["sd_regret"], 4),
                "sd_regret_b": round(sb["sd_regret"], 4),
                "n_outcome_ci95": round(n_out, 1),
                "n_perdecision_ci95": round(n_dec, 1),
                "n_outcome_power80": round(n_out * power80, 1),
                "n_perdecision_power80": round(n_dec * power80, 1),
                "ratio": round(n_out / n_dec, 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the per-pair rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(stat: dict, agent_rows: list, rows: list[dict], summary: dict, power80):
    """Write yahtzee_trials.md and yahtzee_trials.json; return both paths."""
    pair_header = list(rows[0])
    pair_table = [[str(r[key]) for key in pair_header] for r in rows]
    summary_rows = [
        [
            str(summary["pairs"]),
            f"{summary['ratio_median']:.2f}",
            f"{summary['ratio_min']:.2f}",
            f"{summary['ratio_max']:.2f}",
            f"{power80:.4f}",
        ]
    ]
    summary_header = ["pairs", "median ratio", "min ratio", "max ratio", "power80 multiplier"]
    tables = [
        ("Score and regret SD per agent", AGENT_HEADER, agent_rows),
        ("Trials to resolve each agent pair", pair_header, pair_table),
        ("Efficiency ratio over all pairs", summary_header, summary_rows),
    ]
    numbers = {"agents": stat, "pairs": rows, "summary": summary}
    return write_results(EXPERIMENT, tables, numbers)


def main() -> None:
    """Measure the score and regret SDs per agent and write trial counts per agent pair."""
    start_time = time.perf_counter()
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    out_csv = args.out_csv if args.out_csv is not None else output_dir(EXPERIMENT) / CSV_NAME
    z95 = stats.norm.ppf(config["interval_quantile"])  # 95% interval excludes zero
    power80 = ((z95 + stats.norm.ppf(config["power"])) / z95) ** 2  # multiplier for 80% power
    vstar, panel = load_panel(args, config)

    exact = json.loads(args.panel_json.read_text())["panel"]
    stat, agent_rows = measure_panel(vstar, panel, exact, args, config)

    optimal_sd = stat["optimal"]["sd_score"]
    assert abs(optimal_sd - OPTIMAL_SCORE_SD) < config["sd_tolerance"], optimal_sd
    print(f"\noptimal score SD {optimal_sd:.4f} matches the published {OPTIMAL_SCORE_SD} anchor\n")

    rows = pair_rows(stat, z95, power80)
    ratios = np.array([r["ratio"] for r in rows])
    print(
        f"{len(rows)} agent pairs | efficiency ratio: median {np.median(ratios):.2f}x, "
        f"range {ratios.min():.2f}-{ratios.max():.2f}x"
    )
    print(f"(the ratio is criterion-free: 80% power multiplies BOTH columns by {power80:.4f})")

    write_csv(out_csv, rows)
    summary = {
        "pairs": len(rows),
        "ratio_median": float(np.median(ratios)),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "power80_multiplier": float(power80),
        "optimal_sd_score": optimal_sd,
    }
    md_path, json_path = write_report(stat, agent_rows, rows, summary, power80)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"[{time.perf_counter() - start_time:.0f}s]")


if __name__ == "__main__":
    main()
