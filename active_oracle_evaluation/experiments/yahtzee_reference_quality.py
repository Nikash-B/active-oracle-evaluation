"""Rating bias as a function of reference quality in solitaire Yahtzee.

The exact reference ``V*`` is replaced by a degraded one, ``V_ref = V* + N(0, sigma)``, and every
panel agent's regret is scored against it. The suite's nodes (driven by ``optimal``) and the
agents' actions are unchanged, so only the pricing of mistakes degrades.

By default the reference at each sigma uses the same noise draw as the panel's ``noisy_s<sigma>``
agent, its twin, which it scores at zero regret. The sweep therefore reports Kendall's tau both over
the whole panel and with the twin removed. With ``--independent`` the reference uses a noise seed
that no panel agent shares, over ``--draws`` independent draws per sigma.

The reference's quality is reported as ``Delta(ref)``, its own points-per-game shortfall from
optimal, on the same scale as the agents it ranks. The sweep is written as a JSON file and a
Markdown table beside it (``yahtzee_reference_quality.json`` and ``.md`` by default).

    python -m active_oracle_evaluation.experiments.yahtzee_reference_quality --games 3000
    python -m active_oracle_evaluation.experiments.yahtzee_reference_quality --games 3000 \\
        --independent --draws 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.rank_recovery import kendall_vs_full, pairwise_accuracy
from active_oracle_evaluation.eval.validity import tied_values
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import markdown_table
from active_oracle_evaluation.yahtzee.agents import (
    NOISE_SIGMAS,
    build_panel,
    noise_seed,
    noisy_value_table,
)
from active_oracle_evaluation.yahtzee.dp import START, solve_optimal
from active_oracle_evaluation.yahtzee.policy import evaluate
from active_oracle_evaluation.yahtzee.suite import DECISIONS_PER_GAME, generate

EXPERIMENT = "yahtzee_reference_quality"
PANEL_JSON = DATA / "yahtzee" / "yahtzee_panel.json"
SWEEP_HEADER = [
    "sigma",
    "draw",
    "Delta(ref)",
    "Delta(ref)/dec",
    "tau",
    "tau_ex_twin",
    "pairwise",
    "offset",
    "slope",
    "misordered",
    "twin",
    "twin R_ref",
    "twin rank",
    "twin true rank",
]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Command-line options for the reference-degradation sweep."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=config["games"])
    parser.add_argument("--suite-seed", type=int, default=config["suite_seed"])
    parser.add_argument("--panel-seed", type=int, default=config["panel_seed"])
    parser.add_argument("--vstar", type=Path, default=None, help="cached optimal table (.npy)")
    parser.add_argument(
        "--panel-json",
        type=Path,
        default=PANEL_JSON,
        help="exact panel evaluation written by "
        "python -m active_oracle_evaluation.experiments.yahtzee_panel",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="degrade the reference with a noise seed that no panel agent shares",
    )
    parser.add_argument("--indep-seed", type=int, default=config["indep_seed"])
    parser.add_argument(
        "--draws",
        type=int,
        default=config["draws"],
        help="with --independent: number of independent noise draws per sigma",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default: outputs/yahtzee_reference_quality/"
        "yahtzee_reference_quality.json, or yahtzee_reference_quality_independent.json "
        "with --independent); the Markdown table goes beside it",
    )
    return parser


def order_of(values: np.ndarray, names: list[str], what: str) -> list[int]:
    """Agent indices sorted by ascending value; exits if any two values tie."""
    ties = tied_values({names[i]: float(v) for i, v in enumerate(values)})
    if ties:
        raise SystemExit(f"tie in {what}: {ties}")
    return list(np.argsort(values, kind="stable"))


def reference_table(sigma, draw, args, vstar, rows, tolerance):
    """Return the degraded reference table at one sigma and draw, and its skill gap."""
    if sigma == 0.0:
        return vstar, 0.0
    if args.independent:
        seed = noise_seed(sigma, args.indep_seed + 1000 * draw)
        v_ref = noisy_value_table(vstar, sigma, seed)
        v_pi_ref, _, r_pi_ref = evaluate(v_ref, vstar)
        ref_delta = float(vstar[START] - v_pi_ref[START])
        # On-policy regret must equal the skill gap exactly.
        assert abs(r_pi_ref[START] - ref_delta) < tolerance, (sigma, r_pi_ref[START], ref_delta)
        return v_ref, ref_delta
    v_ref = noisy_value_table(vstar, sigma, noise_seed(sigma, args.panel_seed))
    # The reference is the panel's noisy_s<sigma> agent, so its skill gap is in the panel.
    return v_ref, rows[f"noisy_s{sigma:g}"]["gap_vs_optimal"]


def score_reference(sigma, draw, v_ref, ref_delta, args, panel, names, truth, true_order):
    """Rank the panel by regret against one reference, print the line and return the row."""
    truth_per_decision = truth / DECISIONS_PER_GAME
    start = time.perf_counter()
    suite = generate(v_ref, panel, args.games, args.suite_seed)
    r_ref = suite["regret"].mean(axis=1)
    seconds = time.perf_counter() - start

    ref_order = order_of(r_ref, names, f"R_ref at sigma={sigma}")
    tau = kendall_vs_full(ref_order, true_order)
    acc = pairwise_accuracy(ref_order, true_order)

    # The same ranking with the reference's twin removed.
    twin = None if (sigma == 0.0 or args.independent) else names.index(f"noisy_s{sigma:g}")
    keep = [i for i in range(len(names)) if i != twin]
    sub = [names[i] for i in keep]
    tau_ex = kendall_vs_full(
        order_of(r_ref[keep], sub, f"R_ref ex-twin at sigma={sigma}"),
        order_of(truth[keep], sub, "truth ex-twin"),
    )
    # Regress R_ref on per-decision truth: slope 1 means mistakes are priced at their true
    # size, and the offset is the common-mode error that cancels in pairwise comparisons.
    slope, offset = np.polyfit(truth_per_decision[keep], r_ref[keep], 1)
    wrong = [
        (names[a], names[b])
        for ia, a in enumerate(ref_order)
        for b in ref_order[ia + 1 :]
        if truth[a] > truth[b]
    ]
    print(
        f"sigma={sigma:<5g} d{draw} ref Delta={ref_delta:8.4f} "
        f"({ref_delta / DECISIONS_PER_GAME:.4f}/dec)"
        f"  tau={tau:.4f}  tau_ex_twin={tau_ex:.4f}  pairwise={acc:.4f}"
        f"  offset={offset:7.4f} slope={slope:.3f}  misordered={len(wrong)}",
        flush=True,
    )
    if twin is not None:
        print(
            f"           twin {names[twin]}: R_ref={r_ref[twin]:.4f} "
            f"at rank {ref_order.index(twin)} (true rank {true_order.index(twin)})",
            flush=True,
        )
    return {
        "sigma": sigma,
        "draw": draw,
        "ref_delta": ref_delta,
        "tau_excluding_twin": float(tau_ex),
        "twin_r_ref": None if twin is None else r_ref[twin],
        "twin_rank": None if twin is None else ref_order.index(twin),
        "common_mode_offset": float(offset),
        "slope_vs_truth": float(slope),
        "ref_delta_per_decision": ref_delta / DECISIONS_PER_GAME,
        "tau_vs_truth": float(tau),
        "pairwise_accuracy": float(acc),
        "r_ref": r_ref.tolist(),
        "misordered_pairs": wrong,
        "seconds": seconds,
    }


def sweep_table_row(row, names, true_order):
    """Format one sweep row as the Markdown cells, with the numbers as printed."""
    twin = None if row["twin_rank"] is None else names.index(f"noisy_s{row['sigma']:g}")
    return [
        f"{row['sigma']:g}",
        str(row["draw"]),
        f"{row['ref_delta']:.4f}",
        f"{row['ref_delta'] / DECISIONS_PER_GAME:.4f}",
        f"{row['tau_vs_truth']:.4f}",
        f"{row['tau_excluding_twin']:.4f}",
        f"{row['pairwise_accuracy']:.4f}",
        f"{row['common_mode_offset']:.4f}",
        f"{row['slope_vs_truth']:.3f}",
        str(len(row["misordered_pairs"])),
        "–" if twin is None else names[twin],
        "–" if twin is None else f"{row['twin_r_ref']:.4f}",
        "–" if twin is None else str(row["twin_rank"]),
        "–" if twin is None else str(true_order.index(twin)),
    ]


def write_markdown(path, names, true_order, independent, results):
    """Write the sweep as a titled Markdown table."""
    sampling = "independent" if independent else "shared_draw"
    parts = [
        f"# {EXPERIMENT}",
        f"Noise sampling: {sampling}. True order: {' < '.join(names[i] for i in true_order)}.",
        "## Panel ranked against degraded references",
        markdown_table(SWEEP_HEADER, [sweep_table_row(row, names, true_order) for row in results]),
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    """Score the panel against references of decreasing quality and write the sweep."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    if args.out is None:
        suffix = "_independent" if args.independent else ""
        args.out = output_dir(EXPERIMENT) / f"yahtzee_reference_quality{suffix}.json"
    vstar = np.load(args.vstar) if args.vstar and args.vstar.exists() else solve_optimal()
    panel = build_panel(vstar, seed=args.panel_seed)
    names = list(panel)

    rows = json.loads(args.panel_json.read_text())["panel"]
    truth = np.array([rows[n]["gap_vs_optimal"] for n in names])
    truth_per_decision = truth / DECISIONS_PER_GAME
    true_order = order_of(truth, names, "truth")
    print(
        f"truth from {args.panel_json.name}; true order: "
        f"{' < '.join(names[i] for i in true_order)}\n"
    )

    # sigma = 0 is the exact reference, which recovers tau = 1 by construction.
    n_draws = args.draws if args.independent else 1
    jobs = [(0.0, 0)] + [(sigma, draw) for sigma in NOISE_SIGMAS for draw in range(n_draws)]
    results = []
    for sigma, draw in jobs:
        v_ref, ref_delta = reference_table(
            sigma, draw, args, vstar, rows, config["regret_tolerance"]
        )
        results.append(
            score_reference(sigma, draw, v_ref, ref_delta, args, panel, names, truth, true_order)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "agents": names,
                "truth_delta": truth.tolist(),
                "truth_per_decision": truth_per_decision.tolist(),
                "games": args.games,
                "suite_seed": args.suite_seed,
                "noise_sampling": "independent" if args.independent else "shared_draw",
                "sweep": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    md_path = args.out.with_suffix(".md")
    write_markdown(md_path, names, true_order, args.independent, results)
    print(f"\nwrote {args.out}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
