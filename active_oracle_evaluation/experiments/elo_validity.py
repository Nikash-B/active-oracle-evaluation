"""Anchored Elo against oracle Performance Rating (rating validity).

Plays a head-to-head round-robin and fits anchored Bradley-Terry Elo as an independent strength
measure. Computes each agent's oracle Performance Rating (PR) from duplicate-dice games scored by
gnubg-nn. Reports the Kendall tau between the PR and Elo orders, and a regression of win
probability on the PR gap. Writes a Markdown report to ``--out`` (default
outputs/elo_validity/elo_validity.md) and the data to a ``.json`` file beside it.

The defaults in the [elo_validity] table of config/experiments.toml give the paper's run:

    python -m active_oracle_evaluation.experiments.elo_validity \
        --agents random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt \
            az:models/az75.pt distilled:models/distilled.pt \
        --games 60 --pairs 60 --plies 0 --seed 20260628 --anchor pubeval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import DiceBank, play_duplicate_pair, score_decisions
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.elo import bradley_terry_elo, round_robin
from active_oracle_evaluation.eval.validity import kendall_tau, pr_gap_winprob_fit
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "elo_validity"


def build_parser(cfg) -> argparse.ArgumentParser:
    """Return the command-line parser with defaults from the settings table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", default=list(cfg["agents"]), metavar="SPEC")
    p.add_argument("--games", type=int, default=cfg["games"], help="round-robin games per pair")
    p.add_argument(
        "--pairs", type=int, default=cfg["pairs"], help="duplicate-dice pairs per pairing (PR)"
    )
    p.add_argument("--plies", type=int, default=cfg["plies"], help="oracle depth for PR")
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument(
        "--anchor", default=cfg["anchor"], help="agent pinned to Elo 0 (the scale anchor)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report path (default outputs/elo_validity/elo_validity.md)",
    )
    return p


def compute_pr(agents, pairs, seed, plies):
    """Round-robin duplicate-dice + gnubg-oracle scoring -> {agent: PR}."""
    bank = DiceBank(seed)
    all_logs = []
    for (na, a), (nb, b) in combinations(agents.items(), 2):
        _, logs = play_duplicate_pair(a, b, bank, pairs, log=True)
        for log in logs:
            all_logs.append(replace(log, agent=(na if log.agent == "a" else nb)))
    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig  # needs gnubg-nn

    summary = score_decisions(all_logs, GnubgOracle(OracleConfig(plies=plies)))
    return {n: summary.by_agent[n].pr for n in agents}


def agreement_words(tau, slope, cfg):
    """Return the verdict phrase and the agreement level for a Kendall tau and slope."""
    direction = "**validates**" if (tau >= cfg["tau_strong"] and slope > 0) else "is **mixed for**"
    if tau == 1:
        agreement = "identical"
    elif tau >= cfg["tau_strong"]:
        agreement = "strong"
    elif tau >= cfg["tau_moderate"]:
        agreement = "moderate"
    else:
        agreement = "weak"
    return direction, agreement


def render(args, cfg, names, pr, elo, tau, slope, r2, n_pairs, t_elo, t_pr) -> str:
    """Return the Markdown report: setup, ratings table and validity statistics."""
    pr_order = sorted(names, key=lambda n: pr[n])  # best (lowest PR) first
    elo_order = sorted(names, key=lambda n: -elo[n])  # best (highest Elo) first
    pr_rank = {n: i + 1 for i, n in enumerate(pr_order)}
    elo_rank = {n: i + 1 for i, n in enumerate(elo_order)}
    lines = [
        "# Oracle-PR validity vs head-to-head Elo",
        "",
        "Does the oracle-PR rating rank agents the same way independent head-to-head games do? "
        "Kendall τ of the PR vs Elo orders, plus a PR-gap → win-probability calibration. "
        "τ near 1 and a positive slope validate PR as a self-contained strength meter.",
        "",
        "## Setup",
        "",
        f"- Agents ({len(names)}): {', '.join(names)}.",
        f"- Elo: round-robin {args.games} games/pair (anchor `{args.anchor}` = 0). "
        f"PR: {args.pairs} duplicate-dice pairs/pairing, gnubg-nn {args.plies}-ply. "
        f"Seed {args.seed}.",
        f"- Wall: {t_elo:.0f}s Elo + {t_pr:.0f}s PR.",
        "",
        "## Ratings (sorted by PR)",
        "",
        "| agent | PR | Elo | PR rank | Elo rank |",
        "|---|---|---|---|---|",
    ]
    for n in pr_order:
        lines.append(f"| {n} | {pr[n]:.1f} | {elo[n]:+.0f} | {pr_rank[n]} | {elo_rank[n]} |")
    direction, agreement = agreement_words(tau, slope, cfg)
    lines += [
        "",
        "## Validity",
        "",
        f"- **Kendall τ (PR order vs Elo order): {tau:+.3f}** ({agreement} agreement).",
        f"- **PR-gap → win-prob:** log-odds slope **{slope:+.4f}** per PR point, "
        f"R² **{r2:.3f}** over {n_pairs} ordered pairs (a larger PR advantage predicts a "
        "higher win rate when slope > 0).",
        "",
        f"This {direction} the oracle-PR benchmark as a self-contained strength meter: a rating "
        f"derived purely from oracle equities ranks agents like real games do.",
        "",
    ]
    return "\n".join(lines)


def write_json(path, names, pr, elo, tau, slope, r2, rr, seed):
    """Write the ratings, statistics and round-robin win and game matrices as JSON."""
    path.write_text(
        json.dumps(
            {
                "names": names,
                "pr": pr,
                "elo": elo,
                "tau": tau,
                "slope": slope,
                "r2": r2,
                "wins": rr.wins.tolist(),
                "games": rr.games.tolist(),
                "seed": seed,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Play the round-robin, compute PR, and write the report and its JSON data."""
    cfg = settings(EXPERIMENT)
    args = build_parser(cfg).parse_args()
    if args.out is None:
        args.out = output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    encoder = BinarySlotEncoder()
    agents = dict(resolve_agent(s, encoder) for s in args.agents)
    names = list(agents)

    print(f"round-robin Elo ({args.games} games/pair) ...")
    t0 = time.perf_counter()
    rr = round_robin(agents, args.games, seed=args.seed)
    elo = bradley_terry_elo(rr, anchor_name=args.anchor, anchor_elo=0.0)
    t_elo = time.perf_counter() - t0

    print(f"oracle PR ({args.pairs} pairs/pairing, {args.plies}-ply) ...")
    t1 = time.perf_counter()
    pr = compute_pr(agents, args.pairs, args.seed, args.plies)
    t_pr = time.perf_counter() - t1

    pr_order = sorted(names, key=lambda n: pr[n])
    elo_order = sorted(names, key=lambda n: -elo[n])
    tau = kendall_tau(pr_order, elo_order)
    slope, r2, n_pairs = pr_gap_winprob_fit(pr, rr.wins, rr.games, names)

    report = render(args, cfg, names, pr, elo, tau, slope, r2, n_pairs, t_elo, t_pr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    write_json(args.out.with_suffix(".json"), names, pr, elo, tau, slope, r2, rr, args.seed)
    print(f"\nwrote {args.out} (+ .json)\n")
    print(report)


if __name__ == "__main__":
    main()
