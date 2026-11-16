"""Label efficiency against the exact Yahtzee answer: estimator error versus estimand error.

For each selection rule and label budget, the experiment measures the error of the Hajek estimate
against two targets: the full-pool mean suite regret (estimator error) and the exact per-decision
regret (estimator plus estimand error). The first error falls with the budget. The second stops at
the floor |full-pool mean - truth|, which no budget can reduce. It also runs a leave-one-agent-out
check: the design is built without agent j, and the experiment counts how often j keeps its true
rank. It writes yahtzee_label_efficiency.json and a Markdown table file beside it. Run with the
paper's settings:

    python -m active_oracle_evaluation.experiments.yahtzee_label_efficiency \\
        --games 3000 --seeds 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.estimators import hajek, horvitz_thompson
from active_oracle_evaluation.eval.eval_select import _pps_sample
from active_oracle_evaluation.eval.rank_recovery import kendall_vs_full
from active_oracle_evaluation.experiments.yahtzee_rank_recovery import inclusion, order_of
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import markdown_table
from active_oracle_evaluation.yahtzee.agents import build_panel
from active_oracle_evaluation.yahtzee.dp import solve_optimal
from active_oracle_evaluation.yahtzee.suite import DECISIONS_PER_GAME, generate

EXPERIMENT = "yahtzee_label_efficiency"
PANEL_JSON = DATA / "yahtzee" / "yahtzee_panel.json"


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Return the command-line parser, with defaults from the settings table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=config["games"])
    parser.add_argument("--seeds", type=int, default=config["seeds"])
    parser.add_argument("--suite-seed", type=int, default=config["suite_seed"])
    parser.add_argument("--panel-seed", type=int, default=config["panel_seed"])
    parser.add_argument("--vstar", type=Path, default=None)
    parser.add_argument("--panel-json", type=Path, default=PANEL_JSON)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output JSON (default outputs/{EXPERIMENT}/{EXPERIMENT}.json)",
    )
    return parser


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute difference between two arrays."""
    return float(np.mean(np.abs(a - b)))


def tie_broken_order(estimates: np.ndarray, seed: int) -> list[int]:
    """Return agent indices in ascending order of ``estimates``, breaking ties at random."""
    jitter = np.random.default_rng(seed).permutation(len(estimates))
    return [int(jitter[i]) for i in np.argsort(estimates[jitter], kind="stable")]


def budget_cell(config, regret, action, rule, k, truth, r_full, true_order, n_seeds) -> dict:
    """Return the mean errors and Kendall tau of one rule at budget ``k`` over the draws."""
    n_agents, n_pool = regret.shape
    pi = inclusion(rule, k, regret, action)
    vs_full, vs_truth, vs_truth_ht, taus = [], [], [], []
    for s in range(n_seeds):
        idx = _pps_sample(pi, seed=s)
        pi_s = pi[idx]
        est_ht = np.array([horvitz_thompson(regret[a, idx], pi_s, n_pool) for a in range(n_agents)])
        est_h = np.array([hajek(regret[a, idx], pi_s) for a in range(n_agents)])
        vs_full.append(mae(est_h, r_full))
        vs_truth.append(mae(est_h, truth))
        vs_truth_ht.append(mae(est_ht, truth))
        order = tie_broken_order(est_h, config["tie_seed_offset"] + s)
        taus.append(kendall_vs_full(order, true_order))
    return {
        "mae_vs_full_pool": float(np.mean(vs_full)),
        "mae_vs_truth": float(np.mean(vs_truth)),
        "mae_vs_truth_ht": float(np.mean(vs_truth_ht)),
        "tau_vs_truth": float(np.mean(taus)),
    }


def leave_one_agent_out(config, regret, action, names, true_order, n_seeds) -> list[dict]:
    """Return, per agent, how often it keeps its true rank when left out of the design."""
    n_agents = len(names)
    loao = []
    for j in range(n_agents):
        keep = [a for a in range(n_agents) if a != j]
        pi = inclusion(config["loao_rule"], config["loao_budget"], regret[keep], action[keep])
        exact = 0
        for s in range(n_seeds):
            idx = _pps_sample(pi, seed=s)
            est = np.array([hajek(regret[a, idx], pi[idx]) for a in range(n_agents)])
            order = tie_broken_order(est, config["loao_tie_seed_offset"] + s)
            exact += int(order.index(j) == true_order.index(j))
        loao.append({"agent": names[j], "rank_exact_rate": exact / n_seeds})
    return loao


def run_suite(config, args, vstar, panel, label, truth, true_order) -> dict:
    """Generate one suite, then run the budget sweep and the leave-one-agent-out check on it."""
    names = list(panel)
    start = time.perf_counter()
    suite = generate(vstar, panel, args.games, args.suite_seed, mixture=label == "mixture")
    regret, action = suite["regret"], suite["action"]
    n_pool = regret.shape[1]
    r_full = regret.mean(axis=1)
    # The full-pool value's own error against truth: a property of the suite, not the sample.
    floor = mae(r_full, truth)
    print(
        f"[{label}] {n_pool} nodes in {time.perf_counter() - start:.0f}s | "
        f"estimand floor |r_full - truth| = {floor:.4f}"
    )
    cells: dict = {}
    for rule in config["rules"]:
        cells[rule] = {}
        for k in config["budgets"]:
            cell = budget_cell(
                config, regret, action, rule, k, truth, r_full, true_order, args.seeds
            )
            cells[rule][str(k)] = cell
            print(
                f"  {rule:<18} K={k:<5} MAE vs full-pool {cell['mae_vs_full_pool']:7.4f} | "
                f"vs truth {cell['mae_vs_truth']:7.4f} | tau {cell['tau_vs_truth']:.4f}",
                flush=True,
            )
    loao = leave_one_agent_out(config, regret, action, names, true_order, args.seeds)
    worst = min(loao, key=lambda r: r["rank_exact_rate"])
    print(
        f"  LOAO @ K={config['loao_budget']}: worst agent {worst['agent']} keeps its exact true "
        f"rank on {100 * worst['rank_exact_rate']:.0f}% of seeds\n"
    )
    return {
        "n_pool": n_pool,
        "r_suite_full": r_full.tolist(),
        "estimand_floor": floor,
        "by_rule": cells,
        "loao": loao,
    }


def markdown_report(out: dict, loao_budget: int) -> str:
    """Return the result tables of ``out`` as Markdown, numbers formatted as printed."""
    parts = [f"# {EXPERIMENT}"]
    floor_rows = [
        [label, suite["n_pool"], f"{suite['estimand_floor']:.4f}"]
        for label, suite in out["suites"].items()
    ]
    parts += [
        "## Estimand floor |r_full - truth| per suite",
        markdown_table(["suite", "nodes", "estimand floor"], floor_rows),
    ]
    for label, suite in out["suites"].items():
        rows = [
            [
                rule,
                k,
                f"{cell['mae_vs_full_pool']:.4f}",
                f"{cell['mae_vs_truth']:.4f}",
                f"{cell['tau_vs_truth']:.4f}",
            ]
            for rule, by_k in suite["by_rule"].items()
            for k, cell in by_k.items()
        ]
        header = ["rule", "K", "MAE vs full-pool", "MAE vs truth", "tau vs truth"]
        parts += [f"## Hajek error by rule and budget, {label} suite", markdown_table(header, rows)]
        loao_rows = [
            [row["agent"], f"{100 * row['rank_exact_rate']:.0f}%"] for row in suite["loao"]
        ]
        parts += [
            f"## Leave-one-agent-out exact-rank rate at K={loao_budget}, {label} suite",
            markdown_table(["agent", "exact true rank"], loao_rows),
        ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    """Run the label-efficiency and leave-one-agent-out analyses on both suites."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    out_path = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    vstar = np.load(args.vstar) if args.vstar and args.vstar.exists() else solve_optimal()
    panel = build_panel(vstar, seed=args.panel_seed)
    panel = {k: v for k, v in panel.items() if k not in config["drop"]}
    names = list(panel)
    rows = json.loads(args.panel_json.read_text())["panel"]
    truth = np.array([rows[n]["gap_vs_optimal"] for n in names]) / DECISIONS_PER_GAME
    true_order = order_of(truth, names, "truth")
    print(f"panel: {names}\ntrue order: {' < '.join(names[i] for i in true_order)}\n")

    out = {
        "agents": names,
        "truth_per_decision": truth.tolist(),
        "budgets": list(config["budgets"]),
        "seeds": args.seeds,
        "games": args.games,
        "suites": {},
    }
    for label in config["suites"]:
        out["suites"][label] = run_suite(config, args, vstar, panel, label, truth, true_order)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(markdown_report(out, config["loao_budget"]), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
