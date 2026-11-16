r"""Sensitivity of the agent ranking to the oracle that labels the positions.

Scores the agent panel on one shared position set under an oracle ladder (GNU Backgammon at 0-ply
and 2-ply, and the weak pubeval control) and compares the resulting Performance Rating (PR)
rankings with Kendall's tau, the best-move disagreement between the GNU Backgammon depths and the
sign agreement of every pairwise PR difference. The position suite is seeded, so the same
``--seed`` and ``--sampler`` give the same positions under every oracle; only the frozen equities
differ. The eXtreme Gammon comparison is
``python -m active_oracle_evaluation.experiments.xg_oracle``.
Settings are in the [oracle_depth] table of config/experiments.toml.

    python -m active_oracle_evaluation.experiments.oracle_depth \
        --agents random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt \
        az:models/az75.pt distilled:models/distilled.pt --n 500 --seed 20260628

It writes outputs/oracle_depth/oracle_depth.md (the report), oracle_depth.json (the same numbers)
and oracle_depth.csv (rank, PR and error per move of every agent under every oracle).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.suite import build_position_suite, score_agent_on_suite
from active_oracle_evaluation.eval.validity import kendall_tau
from active_oracle_evaluation.experiments.oracle_depth_report import render
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "oracle_depth"
VERDICT_NOTE = "verdict from available oracles only; full rule also needs rollout + XG"


def build_parser(config) -> argparse.ArgumentParser:
    """Return the command-line parser with defaults from the settings table."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--agents", nargs="+", required=True, metavar="SPEC", help="agents to rank (>=2)"
    )
    p.add_argument("--n", type=int, default=config["n"], help="number of shared positions")
    p.add_argument("--seed", type=int, default=config["seed"])
    p.add_argument(
        "--sampler", default=config["sampler"], help="agent whose self-play generates the positions"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="markdown report path (default outputs/oracle_depth/oracle_depth.md)",
    )
    return p


def build_ladder(gnubg_plies):
    """Return the oracle ladder as (name, oracle) pairs: GNU Backgammon depths, then pubeval."""
    from active_oracle_evaluation.eval.oracle import (  # lazy: needs gnubg-nn
        GnubgOracle,
        OracleConfig,
    )
    from active_oracle_evaluation.eval.pubeval_oracle import PubevalOracle  # lazy: needs gnubg-nn

    ladder = [(f"{plies}-ply", GnubgOracle(OracleConfig(plies=plies))) for plies in gnubg_plies]
    ladder.append(("pubeval", PubevalOracle()))
    return ladder


def best_action_diff(suite_a, suite_b) -> tuple[int, int]:
    """Return ``(n_differ, n_total)``: positions whose best afterstate differs between suites.

    The suites share positions (same seed and sampler), so they align position by position; only
    the frozen equities differ.
    """
    differ = 0
    for pa, pb in zip(suite_a.positions, suite_b.positions):
        ba = max(range(len(pa.equities)), key=lambda i: pa.equities[i])
        bb = max(range(len(pb.equities)), key=lambda i: pb.equities[i])
        differ += int(ba != bb)
    return differ, len(suite_a.positions)


def sign_agreement(
    pr_a: dict[str, float], pr_b: dict[str, float], names: list[str]
) -> tuple[int, int]:
    """Return ``(n_preserved, n_pairs)``: agent pairs whose PR-difference sign two oracles share.

    A tie counts as a mismatch only if the other oracle separates the pair.
    """
    preserved = 0
    pairs = list(combinations(names, 2))
    for i, j in pairs:
        sa = (pr_a[i] > pr_a[j]) - (pr_a[i] < pr_a[j])
        sb = (pr_b[i] > pr_b[j]) - (pr_b[i] < pr_b[j])
        preserved += int(sa == sb)
    return preserved, len(pairs)


def build_suites(ladder, n, seed, sampler) -> dict[str, object]:
    """Return one suite per oracle: the same positions with different frozen equities."""
    suites: dict[str, object] = {}
    for oname, oracle in ladder:
        print(f"building suite under oracle {oname!r} ({n} positions) ...")
        t0 = time.perf_counter()
        suites[oname] = build_position_suite(n, oracle=oracle, seed=seed, sampler=sampler)
        print(f"  -> built in {time.perf_counter() - t0:.0f}s")
    return suites


def score_suites(suites, agents):
    """Score every agent on every suite; return rankings (by PR, lower = better), PR and EPM."""
    pr_by_oracle: dict[str, dict[str, float]] = {}
    epm_by_oracle: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    for oname, suite in suites.items():
        scores = [score_agent_on_suite(agent, suite, label=name) for name, agent in agents]
        scores.sort(key=lambda s: s.pr)
        rankings[oname] = [s.agent for s in scores]
        pr_by_oracle[oname] = {s.agent: s.pr for s in scores}
        epm_by_oracle[oname] = {s.agent: s.epm for s in scores}
    return rankings, pr_by_oracle, epm_by_oracle


def tau_matrix(onames, rankings) -> dict[str, dict[str, float]]:
    """Return Kendall's tau between the rankings of every pair of oracles."""
    tau: dict[str, dict[str, float]] = {a: {} for a in onames}
    for a in onames:
        for b in onames:
            tau[a][b] = 1.0 if a == b else kendall_tau(rankings[a], rankings[b])
    return tau


def build_verdict(tau, pr_by_oracle, agent_names, strong_sorted, pubeval_margin) -> dict:
    """Return the robustness checks of the oracles in the ladder.

    Robust iff the two strongest oracles preserve every pairwise PR-difference sign and pubeval's
    tau is more than ``pubeval_margin`` below the strong-strong tau. The ladder has no rollout
    oracle, so both rollout terms are None; the eXtreme Gammon comparison is the xg_oracle
    experiment.
    """
    checks: dict[str, object] = {}

    checks["tau_rollout_2ply"] = None
    checks["tau_rollout_xg"] = None

    a, b = strong_sorted[0], strong_sorted[1]
    pres, total = sign_agreement(pr_by_oracle[a], pr_by_oracle[b], agent_names)
    sign_pres = {"pair": [a, b], "preserved": pres, "pairs": total, "all": pres == total}
    checks["sign_preservation"] = sign_pres

    strong_tau = tau[a][b]
    pe_tau = tau["pubeval"][a]
    pubeval_clearly_worse = {
        "pubeval_vs_strongest": pe_tau,
        "strong_strong_tau": strong_tau,
        "clearly_lower": pe_tau < strong_tau - pubeval_margin,
    }
    checks["pubeval_control"] = pubeval_clearly_worse

    checks["robust"] = all([sign_pres["all"], pubeval_clearly_worse["clearly_lower"]])
    checks["note"] = VERDICT_NOTE
    return checks


def summary(args, agent_names, onames, ranked, tau, diffs, verdict, suites) -> dict:
    """Return the JSON summary of the experiment."""
    rankings, pr_by_oracle, epm_by_oracle = ranked
    return {
        "n": args.n,
        "seed": args.seed,
        "sampler": args.sampler,
        "agents": agent_names,
        "oracles": onames,
        "rankings": rankings,
        "pr": pr_by_oracle,
        "epm": epm_by_oracle,
        "kendall_tau": tau,
        "best_action_differs": {f"{a}|{b}": list(v) for (a, b), v in diffs.items()},
        "verdict": verdict,
        "oracle_versions": {o: getattr(s, "oracle_version", "") for o, s in suites.items()},
    }


def write_csv(path, onames, rankings, pr_by_oracle, epm_by_oracle) -> None:
    """Write the rank, PR and error per move of every agent under every oracle."""
    rows = ["oracle,agent,rank,pr,epm_mpts"]
    for oname in onames:
        for rank, agent in enumerate(rankings[oname], 1):
            pr = pr_by_oracle[oname][agent]
            epm = epm_by_oracle[oname][agent]
            rows.append(f"{oname},{agent},{rank},{pr:.3f},{epm:.3f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Build one suite per oracle, rank the agents on each, and write the comparison report."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    encoder = BinarySlotEncoder()
    agents = [resolve_agent(s, encoder) for s in args.agents]
    if len(agents) < 2:
        raise SystemExit("need >= 2 agents to rank")
    agent_names = [n for n, _ in agents]
    _, sampler = resolve_agent(args.sampler, encoder)

    ladder = build_ladder(config["gnubg_plies"])
    suites = build_suites(ladder, args.n, args.seed, sampler)
    ranked = score_suites(suites, agents)
    rankings, pr_by_oracle, epm_by_oracle = ranked

    onames = list(suites.keys())
    tau = tau_matrix(onames, rankings)

    # Best-action disagreement between every pair of GNU Backgammon oracles, in ladder order.
    gnubg_names = [f"{plies}-ply" for plies in config["gnubg_plies"]]
    diffs: dict[tuple[str, str], tuple[int, int]] = {}
    for a, b in combinations(gnubg_names, 2):
        diffs[(a, b)] = best_action_diff(suites[a], suites[b])

    # The GNU Backgammon oracles ordered strongest (deepest) first.
    strong_sorted = sorted(gnubg_names, key=lambda o: -int(o.split("-")[0]))
    verdict = build_verdict(tau, pr_by_oracle, agent_names, strong_sorted, config["pubeval_margin"])

    out = args.out or output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    tau_robust = config["tau_robust"]
    report = render(args, onames, rankings, pr_by_oracle, tau, diffs, verdict, suites, tau_robust)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    sidecar = out.with_suffix(".json")
    numbers = summary(args, agent_names, onames, ranked, tau, diffs, verdict, suites)
    sidecar.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    csv_path = output_dir(EXPERIMENT) / f"{EXPERIMENT}.csv"
    write_csv(csv_path, onames, rankings, pr_by_oracle, epm_by_oracle)

    print(f"\nwrote {out}\nwrote {sidecar}\nwrote {csv_path}\n")
    print(report)


if __name__ == "__main__":
    main()
