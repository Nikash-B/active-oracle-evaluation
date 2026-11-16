"""Rank recovery and estimand bias of the suite rating against exact Yahtzee regret.

Truth is the exact per-game deficit ``Delta(pi) = V*(start) - V^pi(start)``. Every Yahtzee game has
exactly ``DECISIONS_PER_GAME`` decisions, so ``Delta(pi) / DECISIONS_PER_GAME`` is the per-decision
quantity directly comparable with mean suite regret.

Three analyses run on each of two suites (optimal play and a panel mixture):

- Estimand: with no sampling, compare the order induced by full-suite regret (``R_suite_full``)
  with the true order, and report the per-agent gap between the two magnitudes.
- Sampling: for each selection rule and label budget K, sample decisions, estimate every agent's
  suite regret, and compare the estimated order with the true order.
- Selection margin: at each budget, the mean tau of disagreement minus that of uniform sampling,
  and the uniform budget that reaches disagreement's mean tau, as a ratio to K.

Run with the paper's settings on all ten agents, on the paper's nine-agent panel (every agent
but ``fixed_order``) and on the near-tied panel:

    python -m active_oracle_evaluation.experiments.yahtzee_rank_recovery
    python -m active_oracle_evaluation.experiments.yahtzee_rank_recovery --preset estimand
    python -m active_oracle_evaluation.experiments.yahtzee_rank_recovery \\
        --agents optimal noisy_s0.5 noisy_s1 --tag near_tied

Each run writes ``yahtzee_rank_recovery<_tag>.json`` and ``yahtzee_rank_recovery<_tag>.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.estimators import hajek, horvitz_thompson
from active_oracle_evaluation.eval.eval_select import (
    _pps_sample,
    disagreement_importance,
    floor_mix,
    inclusion_probs,
    pairwise_regret_importance,
)
from active_oracle_evaluation.eval.rank_recovery import (
    kendall_vs_full,
    pairwise_accuracy,
    pr_mae,
    top1_recovered,
)
from active_oracle_evaluation.eval.validity import tied_values
from active_oracle_evaluation.experiments.yahtzee_rank_recovery_report import write_report
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.yahtzee.agents import build_panel
from active_oracle_evaluation.yahtzee.dp import START, solve_optimal
from active_oracle_evaluation.yahtzee.policy import evaluate
from active_oracle_evaluation.yahtzee.suite import DECISIONS_PER_GAME, generate

EXPERIMENT = "yahtzee_rank_recovery"
# Two mean taus closer than this count as tied.
TIE_TOL = 1e-9


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings table."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--games", type=int, default=cfg["games"])
    p.add_argument("--seeds", type=int, default=cfg["seeds"], help="repeats per (rule, K)")
    p.add_argument("--suite-seed", type=int, default=cfg["suite_seed"])
    p.add_argument("--vstar", type=Path, default=None)
    p.add_argument(
        "--panel-json",
        type=Path,
        default=DATA / "yahtzee" / "yahtzee_panel.json",
        help="read Delta(pi) from a saved yahtzee_panel output instead of "
        "evaluating every agent; evaluated exactly when the file is absent",
    )
    p.add_argument("--out-dir", type=Path, default=None, help=f"default: outputs/{EXPERIMENT}")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    p.add_argument(
        "--agents",
        nargs="+",
        default=None,
        metavar="NAME",
        help="restrict the panel to these agents, e.g. the near-tied subset "
        "`--agents optimal noisy_s0.5 noisy_s1`",
    )
    p.add_argument(
        "--preset",
        choices=sorted(cfg["presets"]),
        default=None,
        help="restrict the panel to a named agent subset; `estimand` is the nine agents "
        "other than fixed_order. Sets --tag to the preset name unless given",
    )
    return p


def order_of(values: np.ndarray, names: list[str], what: str) -> list[int]:
    """Return agent indices in ascending order of ``values``, exiting on any exact tie.

    ``kendall_tau`` takes orderings, so a tie would be broken arbitrarily before it runs.
    """
    ties = tied_values({names[i]: float(v) for i, v in enumerate(values)})
    if ties:
        raise SystemExit(f"tie in {what}: {ties} — the tau convention would be arbitrary")
    return list(np.argsort(values, kind="stable"))


def inclusion(rule: str, k: int, regret: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return inclusion probabilities for budget ``k`` under one selection rule.

    Regret importance is used directly, without the phase stratification of ``active_rank``.
    """
    if rule == "uniform":
        return inclusion_probs(np.ones(regret.shape[1]), k)
    if rule == "disagreement":
        return inclusion_probs(floor_mix(disagreement_importance(action)), k)
    if rule == "regret_importance":
        return inclusion_probs(floor_mix(pairwise_regret_importance(regret)), k)
    raise ValueError(rule)


def load_truth(args, panel: dict, names: list[str], vstar: np.ndarray) -> np.ndarray:
    """Return exact Delta(pi) per agent, read from a saved panel run or evaluated."""
    truth = np.empty(len(names))
    if args.panel_json and args.panel_json.exists():
        blob = json.loads(args.panel_json.read_text())
        rows = blob["panel"]
        missing = [n for n in names if n not in rows]
        if missing:
            raise SystemExit(f"{args.panel_json} has no Delta for {missing}")
        for i, name in enumerate(names):
            truth[i] = rows[name]["gap_vs_optimal"]
        print(f"truth reused from {args.panel_json.name}")
    else:
        print("evaluating the panel for truth ...", flush=True)
        for i, name in enumerate(names):
            v_pi, _, _ = evaluate(panel[name]["vhat"], vstar, preference=panel[name]["preference"])
            truth[i] = vstar[START] - v_pi[START]
    for i, name in enumerate(names):
        print(f"  {name:<13} Delta = {truth[i]:8.4f}", flush=True)
    return truth


def run_estimand(regret, names, truth, truth_per_decision, true_order, suite_name):
    """Compare full-suite regret with truth, with no sampling; print and return the results."""
    r_full = regret.mean(axis=1)
    suite_order = order_of(r_full, names, f"{suite_name} R_suite_full")
    tau_estimand = kendall_vs_full(suite_order, true_order)
    gap = r_full - truth_per_decision
    disagreements = [
        (names[a], names[b])
        for ia, a in enumerate(suite_order)
        for b in suite_order[ia + 1 :]
        if (truth[a] > truth[b])
    ]
    print(
        f"  kendall(R_suite_full, truth) = {tau_estimand:.4f}"
        f"   {len(disagreements)} discordant pair(s)"
    )
    for i, nm in enumerate(names):
        print(
            f"    {nm:<13} R_suite_full={r_full[i]:8.4f}"
            f"  truth/{DECISIONS_PER_GAME}={truth_per_decision[i]:8.4f}"
            f"  gap={gap[i]:+8.4f}"
        )
    return r_full, gap, tau_estimand, disagreements


def sample_budget(rule, k, regret, action, n_agents, true_order, r_full, n_seeds, cfg):
    """Estimate the agent order from ``n_seeds`` PPS draws at one rule and budget; print it."""
    n_pool = regret.shape[1]
    lo_pct, hi_pct = cfg["tau_percentiles"]
    pi = inclusion(rule, k, regret, action)
    taus, accs, tops, maes, tied = [], [], [], [], 0
    for s in range(n_seeds):
        idx = _pps_sample(pi, seed=s)
        pi_s = pi[idx]
        est = np.array([horvitz_thompson(regret[a, idx], pi_s, n_pool) for a in range(n_agents)])
        est_h = np.array([hajek(regret[a, idx], pi_s) for a in range(n_agents)])
        # Break ties in the estimates at random: a stable sort would return the panel's
        # declaration order, which matches the true order, and so would score a tie as tau = 1.
        if len(np.unique(est)) < len(est):
            tied += 1
        jitter = np.random.default_rng(cfg["jitter_seed_base"] + s).permutation(len(est))
        sub_order = [int(jitter[i]) for i in np.argsort(est[jitter], kind="stable")]
        taus.append(kendall_vs_full(sub_order, true_order))
        accs.append(pairwise_accuracy(sub_order, true_order))
        tops.append(top1_recovered(sub_order, true_order))
        maes.append(pr_mae(est_h, r_full))
    cell = {
        "tau_mean": float(np.mean(taus)),
        "tau_lo": float(np.percentile(taus, lo_pct)),
        "tau_hi": float(np.percentile(taus, hi_pct)),
        "pairwise_accuracy": float(np.mean(accs)),
        "top1": float(np.mean(tops)),
        "hajek_mae": float(np.mean(maes)),
        "tie_rate": tied / n_seeds,
    }
    print(
        f"  {rule:<18} K={k:<5} tau={np.mean(taus):.4f}"
        f" [{np.percentile(taus, lo_pct):.3f},{np.percentile(taus, hi_pct):.3f}]"
        f"  top1={np.mean(tops):.3f}  mae={np.mean(maes):.4f}"
        f"  ties={tied / n_seeds:.2f}",
        flush=True,
    )
    return cell


def matching_budget(tau: float, budgets: list[int], taus: list[float]) -> float | None:
    """Return the budget at which ``taus`` first reaches ``tau``, interpolated linearly in log K."""
    for k1, k2, t1, t2 in zip(budgets, budgets[1:], taus, taus[1:]):
        if t1 <= tau <= t2 and t2 > t1:
            return k1 * 2 ** (math.log2(k2 / k1) * (tau - t1) / (t2 - t1))
    return None


def first_exact_budget(by_budget: dict[int, dict]) -> int | None:
    """Return the smallest budget whose mean tau is 1, or None if no budget reaches it."""
    exact = [k for k, cell in by_budget.items() if cell["tau_mean"] >= 1 - TIE_TOL]
    return min(exact) if exact else None


def selection_margin(sampling: dict, rule: str = "disagreement", baseline: str = "uniform"):
    """Compare ``rule`` with ``baseline`` at each budget: tau difference and label-equivalence ratio.

    The ratio is the baseline budget that reaches the rule's mean tau, divided by K. It is given
    only where the rule leads, and only when the baseline reaches that tau at a larger budget.
    """
    budgets = sorted(sampling[baseline])
    base_tau = [sampling[baseline][k]["tau_mean"] for k in budgets]
    rows = []
    for i, k in enumerate(budgets):
        diff = sampling[rule][k]["tau_mean"] - base_tau[i]
        outcome = "tie" if abs(diff) < TIE_TOL else ("lead" if diff > 0 else "trail")
        diff = 0.0 if outcome == "tie" else diff
        ratio = None
        if outcome == "lead":
            match = matching_budget(sampling[rule][k]["tau_mean"], budgets[i:], base_tau[i:])
            ratio = None if match is None else match / k
        rows.append({"K": k, "tau_difference": diff, "outcome": outcome, "label_ratio": ratio})
    ratios = [row["label_ratio"] for row in rows if row["label_ratio"] is not None]
    return {
        "rule": rule,
        "baseline": baseline,
        "by_budget": rows,
        "label_ratio_min": min(ratios) if ratios else None,
        "label_ratio_max": max(ratios) if ratios else None,
        "first_exact_budget": {r: first_exact_budget(by_k) for r, by_k in sampling.items()},
    }


def print_margin(margin: dict) -> None:
    """Print the selection margin of one suite."""
    print(f"  {margin['rule']} minus {margin['baseline']}:")
    for row in margin["by_budget"]:
        ratio = "-" if row["label_ratio"] is None else f"{row['label_ratio']:.2f}x"
        print(
            f"    K={row['K']:<5} diff={row['tau_difference']:+.4f}"
            f"  {row['outcome']:<5}  label ratio={ratio}"
        )
    first = ", ".join(f"{r} {k}" for r, k in margin["first_exact_budget"].items())
    print(f"    first budget with mean tau 1: {first}")


def run_suite(args, cfg, vstar, panel, names, truth, true_order, suite_name, mixture):
    """Generate one suite and run the estimand and sampling analyses on it."""
    truth_per_decision = truth / DECISIONS_PER_GAME
    print(f"\n=== suite: {suite_name} ===", flush=True)
    t = time.perf_counter()
    suite = generate(vstar, panel, args.games, args.suite_seed, mixture=mixture)
    regret, action = suite["regret"], suite["action"]
    n_pool = regret.shape[1]
    print(f"  {n_pool:,} nodes in {time.perf_counter() - t:.0f}s")

    r_full, gap, tau_estimand, disagreements = run_estimand(
        regret, names, truth, truth_per_decision, true_order, suite_name
    )

    # Sampling: estimated order against truth for each rule and budget.
    sampling: dict[str, dict] = {}
    for rule in cfg["rules"]:
        sampling[rule] = {}
        for k in cfg["budgets"]:
            sampling[rule][k] = sample_budget(
                rule, k, regret, action, len(names), true_order, r_full, args.seeds, cfg
            )
    margin = selection_margin(sampling)
    print_margin(margin)

    return {
        "n_pool": int(n_pool),
        "r_suite_full": r_full.tolist(),
        "gap": gap.tolist(),
        "tau_estimand": float(tau_estimand),
        "estimand_discordant_pairs": disagreements,
        "sampling": {r: {str(k): v for k, v in d.items()} for r, d in sampling.items()},
        "selection_margin": margin,
    }


def main() -> None:
    """Run the estimand and sampling analyses on both suites and write the results."""
    cfg = settings(EXPERIMENT)
    args = build_parser(cfg).parse_args()
    if args.preset:
        if args.agents:
            raise SystemExit("give --agents or --preset, not both")
        args.agents = list(cfg["presets"][args.preset])
        args.tag = args.tag or args.preset
    if args.out_dir is None:
        args.out_dir = output_dir(EXPERIMENT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vstar = np.load(args.vstar) if args.vstar and args.vstar.exists() else solve_optimal()
    panel = build_panel(vstar, seed=cfg["panel_seed"])
    if args.agents:
        missing = [a for a in args.agents if a not in panel]
        if missing:
            raise SystemExit(f"unknown agent(s) {missing}; panel is {list(panel)}")
        panel = {a: panel[a] for a in args.agents}
        print(f"panel restricted to {len(panel)} agents: {', '.join(panel)}")
    names = list(panel)

    truth = load_truth(args, panel, names, vstar)
    truth_per_decision = truth / DECISIONS_PER_GAME
    true_order = order_of(truth, names, "truth")

    results = {
        "agents": names,
        "restricted": bool(args.agents),
        "truth_delta": truth.tolist(),
        "truth_per_decision": truth_per_decision.tolist(),
        "budgets": list(cfg["budgets"]),
        "seeds": args.seeds,
        "games": args.games,
        "suites": {},
    }
    for suite_name, mixture in (("optimal", False), ("mixture", True)):
        results["suites"][suite_name] = run_suite(
            args, cfg, vstar, panel, names, truth, true_order, suite_name, mixture
        )

    margins = [suite["selection_margin"] for suite in results["suites"].values()]
    ratios = [m[key] for m in margins for key in ("label_ratio_min", "label_ratio_max")]
    ratios = [r for r in ratios if r is not None]
    if ratios:
        print(f"\nlabel ratio over both suites: {min(ratios):.2f}x to {max(ratios):.2f}x")

    stem = f"{EXPERIMENT}_{args.tag}" if args.tag else EXPERIMENT
    out = args.out_dir / f"{stem}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    report = args.out_dir / f"{stem}.md"
    write_report(results, report, stem)
    print(f"\nwrote {out}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
