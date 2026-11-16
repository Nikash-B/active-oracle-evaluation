"""Evaluate the Yahtzee agent panel exactly against the solved game.

The experiment solves solitaire Yahtzee (or loads a cached optimal table V*) and evaluates each
panel agent with one backward pass (:func:`active_oracle_evaluation.yahtzee.policy.evaluate`).
The pass gives the exact expected score, the SD from the exact second moment, and the expected
cumulative oracle regret along the agent's own state distribution. The run checks the
performance-difference identity for every agent, each agent's plausibility band, and the
published anchors V*(start) = 254.5877 and SD = 59.61.

    python -m active_oracle_evaluation.experiments.yahtzee_panel --save-vstar vstar.npy

The run writes yahtzee_panel.json (the panel) and yahtzee_panel.md (the printed table) to
outputs/yahtzee_panel/. The table reports the measured wall time per agent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.paths import output_dir
from active_oracle_evaluation.results import markdown_table
from active_oracle_evaluation.yahtzee import OPTIMAL_EXPECTED_SCORE, OPTIMAL_SCORE_SD
from active_oracle_evaluation.yahtzee.agents import build_panel, check_plausibility
from active_oracle_evaluation.yahtzee.dp import START, solve_optimal
from active_oracle_evaluation.yahtzee.policy import evaluate, identity_error, score_sd

EXPERIMENT = "yahtzee_panel"
TABLE_HEADER = ["agent", "E[score]", "SD", "V*-V_pi", "regret", "id err", "s"]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Command-line options for the panel evaluation."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the panel as JSON here (default: outputs/yahtzee_panel/yahtzee_panel.json)",
    )
    p.add_argument(
        "--vstar", type=Path, default=None, help="cached optimal table (.npy); solved if absent"
    )
    p.add_argument(
        "--save-vstar", type=Path, default=None, help="save the solved optimal table (.npy)"
    )
    p.add_argument("--seed", type=int, default=config["seed"], help="seed for the noisy agents")
    return p


def load_or_solve_vstar(args: argparse.Namespace) -> np.ndarray:
    """Load the cached optimal table, or solve it and optionally save it."""
    if args.vstar and args.vstar.exists():
        vstar = np.load(args.vstar)
        print(f"loaded V* from {args.vstar}")
    else:
        t = time.perf_counter()
        vstar = solve_optimal()
        print(f"solved V* in {time.perf_counter() - t:.0f}s")
        if args.save_vstar:
            np.save(args.save_vstar, vstar)
    return vstar


def format_row(name: str, row: dict) -> list[str]:
    """Format one agent's results as the printed table cells."""
    return [
        name,
        f"{row['expected_score']:.4f}",
        f"{row['sd']:.4f}",
        f"{row['gap_vs_optimal']:.4f}",
        f"{row['on_policy_regret']:.4f}",
        f"{row['identity_error']:.1e}",
        f"{row['seconds']:.0f}",
    ]


def evaluate_panel(panel: dict, vstar: np.ndarray) -> dict[str, dict]:
    """Evaluate every agent exactly, printing one table row per agent."""
    rows: dict[str, dict] = {}
    header = (
        f"{'agent':<14}{'E[score]':>11}{'SD':>9}{'V*-V_pi':>11}{'regret':>11}{'id err':>10}{'s':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, spec in panel.items():
        t = time.perf_counter()
        v_pi, m_pi, r_pi = evaluate(spec["vhat"], vstar, preference=spec["preference"])
        dt = time.perf_counter() - t
        row = {
            "expected_score": float(v_pi[START]),
            "sd": score_sd(v_pi, m_pi),
            "gap_vs_optimal": float(vstar[START] - v_pi[START]),
            "on_policy_regret": float(r_pi[START]),
            "identity_error": identity_error(v_pi, r_pi, vstar),
            "seconds": dt,
        }
        rows[name] = row
        print(
            f"{name:<14}{row['expected_score']:11.4f}{row['sd']:9.4f}"
            f"{row['gap_vs_optimal']:11.4f}{row['on_policy_regret']:11.4f}"
            f"{row['identity_error']:10.1e}{dt:7.0f}",
            flush=True,
        )
    return rows


def write_markdown(path: Path, rows: dict[str, dict], v_star_start: float, worst: float) -> None:
    """Write the panel table and the anchor checks as Markdown, numbers formatted as printed."""
    summary = [
        ["V*(start)", f"{v_star_start:.6f}", f"{OPTIMAL_EXPECTED_SCORE}"],
        ["optimal SD", f"{rows['optimal']['sd']:.6f}", f"{OPTIMAL_SCORE_SD}"],
        ["worst identity error", f"{worst:.2e}", "-"],
    ]
    parts = [
        f"# {EXPERIMENT}",
        "## Exact evaluation of the panel",
        markdown_table(TABLE_HEADER, [format_row(name, row) for name, row in rows.items()]),
        "## Anchors and checks",
        markdown_table(["quantity", "value", "anchor"], summary),
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    """Solve or load V*, evaluate every panel agent, check the anchors and write the results."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    if args.out is None:
        args.out = output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"

    vstar = load_or_solve_vstar(args)

    # V* must match the published optimal expected score.
    assert abs(vstar[START] - OPTIMAL_EXPECTED_SCORE) < config["v_star_tolerance"], vstar[START]
    print(f"V*(start) = {vstar[START]:.6f}  (anchor {OPTIMAL_EXPECTED_SCORE})\n")

    panel = build_panel(vstar, seed=args.seed)
    rows = evaluate_panel(panel, vstar)

    worst = max(r["identity_error"] for r in rows.values())
    print(f"\nworst identity error across the panel: {worst:.2e}")
    assert worst < config["identity_tolerance"], "performance-difference identity does not hold"

    # The identity checks the evaluator; this checks that each agent scores in its expected band.
    implausible = check_plausibility({k: v["expected_score"] for k, v in rows.items()})
    if implausible:
        raise SystemExit(
            "implausible agent scores — the identity cannot catch this:\n  "
            + "\n  ".join(implausible)
        )
    print("plausibility: every agent inside its expected band")
    sd_opt = rows["optimal"]["sd"]
    assert abs(sd_opt - OPTIMAL_SCORE_SD) < config["sd_tolerance"], sd_opt
    print(f"optimal SD = {sd_opt:.6f}  (anchor {OPTIMAL_SCORE_SD})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "panel": rows,
                "seed": args.seed,
                "v_star_start": float(vstar[START]),
                "worst_identity_error": worst,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    md_path = args.out.with_suffix(".md")
    write_markdown(md_path, rows, float(vstar[START]), worst)
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
