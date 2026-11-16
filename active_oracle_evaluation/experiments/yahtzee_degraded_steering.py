"""Regret-importance selection steered by a degraded reference.

Pairwise regret importance computed from exact regret is a skyline: it reads the labels the budget
is meant to save. This experiment builds the selection design from a cheaper reference instead, a
value table V* + N(0, sigma) drawn with a seed no panel agent shares, and then scores the selected
nodes with exact regret. Each design is compared with uniform and disagreement selection on the
near-tied three-agent panel, by mean Kendall tau against the exact ranking at each label budget K.

The reference deficit Delta(ref), the reference's own points-per-game shortfall from optimal, is
read from the output of yahtzee_reference_quality --independent. The degradation is i.i.d. across
states. Run with the paper's settings:

    python -m active_oracle_evaluation.experiments.yahtzee_degraded_steering \\
        --games 3000 --seeds 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.eval_select import (
    _pps_sample,
    floor_mix,
    inclusion_probs,
    pairwise_regret_importance,
)
from active_oracle_evaluation.eval.rank_recovery import kendall_vs_full
from active_oracle_evaluation.experiments.yahtzee_rank_recovery import inclusion, order_of
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import markdown_table
from active_oracle_evaluation.yahtzee.agents import build_panel, noise_seed, noisy_value_table
from active_oracle_evaluation.yahtzee.dp import solve_optimal
from active_oracle_evaluation.yahtzee.suite import DECISIONS_PER_GAME, generate

EXPERIMENT = "yahtzee_degraded_steering"
PANEL_JSON = DATA / "yahtzee" / "yahtzee_panel.json"
REFERENCE_JSON = DATA / "yahtzee" / "yahtzee_reference_quality_independent.json"


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Command-line arguments, with defaults from the settings table."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--games", type=int, default=config["games"])
    p.add_argument("--seeds", type=int, default=config["seeds"])
    p.add_argument("--suite-seed", type=int, default=config["suite_seed"])
    p.add_argument("--panel-seed", type=int, default=config["panel_seed"])
    p.add_argument("--vstar", type=Path, default=None, help="cached V* table (.npy)")
    p.add_argument("--panel-json", type=Path, default=PANEL_JSON)
    p.add_argument(
        "--reference-json",
        type=Path,
        default=REFERENCE_JSON,
        help="output of yahtzee_reference_quality --independent (reference deficit per sigma)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output JSON (default: outputs/{EXPERIMENT}/{EXPERIMENT}.json)",
    )
    return p


def mean_tau(pi, exact, order_true, seeds, n_names, jitter_seed_base):
    """Mean Kendall tau vs truth over `seeds` PPS draws from `pi`, scored on `exact` regret."""
    taus = []
    for s in range(seeds):
        idx = _pps_sample(pi, seed=s)
        est = exact[:, idx].mean(axis=1)
        jitter = np.random.default_rng(jitter_seed_base + s).permutation(n_names)
        taus.append(
            kendall_vs_full(
                [int(jitter[i]) for i in np.argsort(est[jitter], kind="stable")], order_true
            )
        )
    return float(np.mean(taus))


def sweep(order_true, exact, design_regret, seeds, n_names, budgets, jitter_seed_base):
    """Mean tau per budget for a design built on `design_regret`, scored on `exact`."""
    out = {}
    for k in budgets:
        pi = inclusion_probs(floor_mix(pairwise_regret_importance(design_regret)), k)
        out[str(k)] = mean_tau(pi, exact, order_true, seeds, n_names, jitter_seed_base)
    return out


def report_curve(tag, tau_by_k, budgets, rows, flush=False):
    """Print one tau-by-budget line and append it as a Markdown row."""
    print(
        f"  {tag:<26} " + "  ".join(f"K={k}:{tau_by_k[str(k)]:.3f}" for k in budgets),
        flush=flush,
    )
    rows.append([tag] + [f"{tau_by_k[str(k)]:.3f}" for k in budgets])


def run_suite(mixture, vstar, panel, order_true, deltas, args, config):
    """Run the baselines and the degraded-design sweep on one suite; return curves and tables."""
    budgets = config["budgets"]
    jitter_base = config["jitter_seed_base"]
    n_names = len(panel)
    label = "mixture" if mixture else "optimal"
    t = time.perf_counter()
    base = generate(vstar, panel, args.games, args.suite_seed, mixture=mixture)
    exact = base["regret"]
    print(f"[{label}] exact suite in {time.perf_counter() - t:.0f}s")

    curves = {}
    rows = []
    imp_exact = pairwise_regret_importance(exact)
    moved: dict[str, float] = {}
    # Baselines that read no reference.
    for rule in ("uniform", "disagreement"):
        tau_by_k = {}
        for k in budgets:
            pi = inclusion(rule, k, exact, base["action"])
            tau_by_k[str(k)] = mean_tau(pi, exact, order_true, args.seeds, n_names, jitter_base)
        curves[rule] = {"ref_delta": None, "tau": tau_by_k}
        report_curve(rule, tau_by_k, budgets, rows)

    # sigma = 0 is the skyline; larger sigma is a cheaper, worse reference.
    for sigma in (0.0, *config["reference_sigmas"]):
        if sigma == 0.0:
            design = exact
        else:
            seed = noise_seed(sigma, config["reference_seed_base"])
            v_ref = noisy_value_table(vstar, sigma, seed)
            design = generate(v_ref, panel, args.games, args.suite_seed, mixture=mixture)["regret"]
            # The degraded design must differ from the exact one; record their correlation.
            assert not np.allclose(design, exact), f"sigma={sigma}: design == exact regret"
            corr = float(np.corrcoef(pairwise_regret_importance(design), imp_exact)[0, 1])
            moved[f"{sigma:g}"] = corr
        taus = sweep(order_true, exact, design, args.seeds, n_names, budgets, jitter_base)
        curves[f"design_s{sigma:g}"] = {"ref_delta": deltas.get(sigma), "tau": taus}
        if sigma == 0.0:
            tag = "skyline (exact design)"
        else:
            tag = f"design from ref D={deltas[sigma]:.2f}"
        report_curve(tag, taus, budgets, rows, flush=True)
    print(
        "  design-signal correlation with the exact-importance vector: "
        + ", ".join(f"D={deltas[float(k)]:.2f}:{v:.3f}" for k, v in moved.items())
    )
    print()
    curves["_importance_corr_vs_exact"] = moved
    corr_rows = [[f"{deltas[float(k)]:.2f}", f"{v:.3f}"] for k, v in moved.items()]
    tables = [
        (
            f"{label} suite: mean Kendall tau by label budget",
            ["selection"] + [f"K={k}" for k in budgets],
            rows,
        ),
        (
            f"{label} suite: design-signal correlation with the exact-importance vector",
            ["reference D", "correlation"],
            corr_rows,
        ),
    ]
    return label, curves, tables


def write_markdown(tables, path):
    """Write the titled result tables as one Markdown file."""
    parts = [f"# {EXPERIMENT}"]
    for title, header, rows in tables:
        parts += [f"## {title}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n")


def main() -> None:
    """Run the baselines and the degraded-design sweep on both suites and write the results."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    vstar = np.load(args.vstar) if args.vstar and args.vstar.exists() else solve_optimal()
    full = build_panel(vstar, seed=args.panel_seed)
    panel = {k: full[k] for k in config["agents"]}
    names = list(panel)
    rows = json.loads(args.panel_json.read_text())["panel"]
    truth = np.array([rows[n]["gap_vs_optimal"] for n in names]) / DECISIONS_PER_GAME
    order_true = order_of(truth, names, "truth")
    deltas = {
        r["sigma"]: r["ref_delta"]
        for r in json.loads(args.reference_json.read_text())["sweep"]
        if r.get("draw", 0) == 0
    }
    print(f"panel: {names}   true order: {' < '.join(names[i] for i in order_true)}\n")

    out = {
        "agents": names,
        "budgets": list(config["budgets"]),
        "seeds": args.seeds,
        "games": args.games,
        "suites": {},
    }
    tables = []
    for mixture in (False, True):
        label, curves, suite_tables = run_suite(
            mixture, vstar, panel, order_true, deltas, args, config
        )
        out["suites"][label] = curves
        tables += suite_tables

    out_dir = output_dir(EXPERIMENT)
    out_path = args.out if args.out is not None else out_dir / f"{EXPERIMENT}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_markdown(tables, out_dir / f"{EXPERIMENT}.md")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
