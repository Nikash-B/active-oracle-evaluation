"""Label efficiency of `active_rank` under different design sources.

`active_rank` steers its sampling design on pairwise regret importance. This experiment builds that
design either from the labels being estimated (the skyline) or from the regret matrix degraded by
Gaussian noise of sigma times its own standard deviation, and compares both with the label-free
rules:

    labels (sigma = 0)  the skyline: design built from the labels being estimated
    noisy:sigma         design built from a noisier reference
    phase_balanced      the sigma -> infinity limit of the noisy family
    disagreement        label-free: steers on the agents' chosen moves
    random              uniform sampling

Each row of the output CSV is one (design, strategy, budget K, seed) cell, scored against the
full-pool ranking. The mean Kendall's tau per design and budget goes to cheap_steering.md and
cheap_steering.json in outputs/cheap_steering/. The paper's setting (400 draws on the full panel):

    python -m active_oracle_evaluation.experiments.cheap_steering --seeds 400
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import rank_recovery as rr
from active_oracle_evaluation.eval.eval_select import select
from active_oracle_evaluation.paths import DATA, OUTPUTS
from active_oracle_evaluation.results import write_results

EXPERIMENT = "cheap_steering"


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA / "decision_suites" / "full_suite_scores_0ply.npz",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=list(config["ks"]))
    parser.add_argument("--seeds", type=int, default=config["seeds"])
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=OUTPUTS / EXPERIMENT / f"{EXPERIMENT}.csv",
    )
    return parser


def variants(sigmas: list[float]) -> list[tuple[str, str]]:
    """Return the (design, strategy) pairs; the design matters only for active_rank."""
    return (
        [("labels", "active_rank")]
        + [(f"noisy:{sigma:g}", "active_rank") for sigma in sigmas]
        + [("none", "phase_balanced"), ("none", "disagreement"), ("none", "random")]
    )


def degrade(losses: np.ndarray, sigma: float, seed: int, seed_base: int) -> np.ndarray:
    """Add N(0, sigma x SD) noise to the regret matrix and clip at 0 (regret is non-negative)."""
    rng = np.random.default_rng(seed_base + int(sigma * 1000) + seed)
    noised = losses + rng.normal(0.0, sigma * float(losses.std()), size=losses.shape)
    return np.clip(noised, 0.0, None)


def score_cell(design, strategy, budget, seed, data, full_order, seed_base) -> dict:
    """Select one subset and score its ranking against the full-pool order."""
    losses = data["losses"]
    if design in ("labels", "none"):
        importance_losses = None
    else:
        sigma = float(design.split(":")[1])
        importance_losses = degrade(losses, sigma, seed, seed_base)
    selection = select(
        strategy,
        budget,
        losses=losses,
        choices=data["choices"],
        phases=data["phases"],
        seed=seed,
        importance_losses=importance_losses,
        importance="skyline" if design == "labels" else None,
    )
    indices = list(selection.indices)
    subset_pr = rr.pr_subset(losses, indices)
    subset_order = rr.ranking(subset_pr)
    # `rr.ranking` breaks ties by declaration order, so record whether any tie occurred.
    return {
        "panel": data["panel"],
        "n_agents": len(data["names"]),
        "tied": int(len(np.unique(subset_pr)) < len(subset_pr)),
        "design": design,
        "strategy": strategy,
        "K": budget,
        "seed": seed,
        "kendall_tau": rr.kendall_vs_full(subset_order, full_order),
        "pairwise_acc": rr.pairwise_accuracy(subset_order, full_order),
        "top1": float(rr.top1_recovered(subset_order, full_order)),
        "n_labels": float(len(indices)),
    }


def mean_taus(rows: list[dict], design: str, strategy: str, ks: list[int]) -> dict:
    """Return the mean Kendall's tau of one (design, strategy) pair at each budget."""
    return {
        budget: np.mean(
            [
                row["kendall_tau"]
                for row in rows
                if row["design"] == design and row["strategy"] == strategy and row["K"] == budget
            ]
        )
        for budget in ks
    }


def write_summary(data, n_pool, args, summaries) -> None:
    """Write the mean-tau table to cheap_steering.md and cheap_steering.json."""
    header = ["Design"] + [f"K={budget}" for budget in args.ks]
    rows = [[label] + [f"{tau:.4f}" for tau in taus.values()] for label, taus in summaries]
    title = f"Mean Kendall's tau against the full-pool order over {args.seeds} draws per budget"
    numbers = {
        "panel": data["panel"],
        "n_agents": len(data["names"]),
        "n_pool": n_pool,
        "seeds": args.seeds,
        "ks": list(args.ks),
        "mean_kendall_tau": {
            label: {str(budget): float(tau) for budget, tau in taus.items()}
            for label, taus in summaries
        },
    }
    write_results(EXPERIMENT, [(title, header, rows)], numbers)


def main() -> None:
    """Run every design variant over all budgets and seeds and write one CSV row per cell."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    sigmas = [float(sigma) for sigma in config["sigmas"]]
    seed_base = int(config["noise_seed_base"])
    cache = np.load(args.cache, allow_pickle=True)
    names = list(cache["names"])
    losses, choices, phases = cache["losses"], cache["choices"], list(cache["phases"])
    panel = "full"
    data = {"names": names, "losses": losses, "choices": choices, "phases": phases}
    data["panel"] = panel

    n_pool = losses.shape[1]
    full_pr = losses.mean(axis=1) * rr.PR_SCALE
    full_order = rr.ranking(full_pr)
    print(
        f"panel ({panel}, {len(names)} agents): {names}\npool: {n_pool} positions | "
        f"full-pool order: {' < '.join(names[i] for i in full_order)}\n"
    )

    rows = []
    summaries = []
    for design, strategy in variants(sigmas):
        for budget in args.ks:
            for seed in range(args.seeds):
                rows.append(score_cell(design, strategy, budget, seed, data, full_order, seed_base))
        taus = mean_taus(rows, design, strategy, args.ks)
        label = f"{strategy}[{design}]" if design != "none" else strategy
        summaries.append((label, taus))
        summary = "  ".join(f"K={budget}:{tau:.4f}" for budget, tau in taus.items())
        print(f"  {label:<26} {summary}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(data, n_pool, args, summaries)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
