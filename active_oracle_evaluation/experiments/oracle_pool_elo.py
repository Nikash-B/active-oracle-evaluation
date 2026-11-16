"""Anchored Elo round-robin with the oracle in the pool, and oracle-PR on the same games.

The pool holds gnubg at 0, 1 and 2 ply, two search agents over the strongest network, an
external exact-Bellman TD network and three panel agents. One play pass gives both ratings:

- Elo: a Bradley-Terry fit to the wins of a full round-robin of duplicate-dice pairs, with
  the anchor pinned to 0 and trial-level bootstrap intervals.
- PR: every unforced decision of the same games, scored by the gnubg-nn 0-ply oracle.

The report gives the Kendall tau between the PR and Elo orders on the full pool and on the
agents at or above the oracle's own strength, where PR is 0 for the oracle by construction.
It writes outputs/oracle_pool_elo/oracle_pool_elo.md and oracle_pool_elo.json. The paper's
settings (the [oracle_pool_elo] table of config/experiments.toml) are the defaults:

    python -m active_oracle_evaluation.experiments.oracle_pool_elo --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.oracle_pool_elo_report import render_report, summary
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "oracle_pool_elo"
# Slowest specs first so the long jobs do not straggle at the end of the pool.
SLOW_AGENTS = ("sel3", "search2", "gnubg2")
# Agents built so far in this worker process, keyed by spec.
_AGENT_CACHE: dict[str, object] = {}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, with the settings file as defaults."""
    cfg = settings(EXPERIMENT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", default=cfg["agents"], metavar="SPEC")
    p.add_argument("--pairs-per-shard", type=int, default=cfg["pairs_per_shard"])
    p.add_argument(
        "--shards", type=int, default=cfg["shards"], help="independent dice-bank blocks per pairing"
    )
    p.add_argument("--workers", type=int, default=cfg["workers"])
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument(
        "--strong-tier", default=cfg["strong_tier"], help="comma list for the subset tau"
    )
    p.add_argument(
        "--boots", type=int, default=cfg["boots"], help="bootstrap resamples for Elo CIs"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report path (default outputs/oracle_pool_elo/oracle_pool_elo.md); "
        "the JSON summary goes next to it with suffix .json",
    )
    return p


def spec_label(spec: str) -> str:
    """Return the display label a spec resolves to (mirrors resolve_agent's naming)."""
    label, _, path = spec.partition(":")
    if label == "gnubg":
        return f"gnubg{path or 0}"
    return label


# ----- worker side (spawn) ---------------------------------------------------


def _worker_init():
    """Pin each worker to the CPU with one thread, before any torch import."""
    os.environ["BACKGAMMON_FORCE_CPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _get_agent(spec: str):
    """Build the agent for a spec once per worker process and cache it."""
    if spec not in _AGENT_CACHE:
        import torch

        from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
        from active_oracle_evaluation.eval.agent_specs import resolve_agent

        torch.set_num_threads(1)
        _AGENT_CACHE[spec] = resolve_agent(spec, BinarySlotEncoder())[1]
    return _AGENT_CACHE[spec]


def _play_job(job):
    """Play one block of duplicate-dice pairs between two agents and return games and logs."""
    (spec_a, spec_b, name_a, name_b, bank_seed, n_pairs, agent_seed) = job
    from active_oracle_evaluation.eval import DiceBank, play_duplicate_pair

    t0 = time.perf_counter()
    a, b = _get_agent(spec_a), _get_agent(spec_b)
    res, logs = play_duplicate_pair(
        a, b, DiceBank(bank_seed), n_pairs, agent_seed=agent_seed, log=True
    )
    sp = res.single_points
    return {
        "a": name_a,
        "b": name_b,
        "bank_seed": bank_seed,
        "single": sp.tolist(),  # a's signed points per game, [t-white, t-black, ...]
        "mean_diff": res.mean_diff,
        "logs": [replace(log, agent=(name_a if log.agent == "a" else name_b)) for log in logs],
        "wall": time.perf_counter() - t0,
    }


# ----- parent side -----------------------------------------------------------


def bootstrap_elo_cis(names, trial_wins, anchor, boots, seed, point_elo):
    """Trial-level bootstrap of the Bradley-Terry Elo fit -> {name: (lo, hi)}.

    The intervals are pivotal (basic), 2*point - percentile, because extreme-imbalance
    pairings (a floor agent vs the top tier) skew the anchored fit's resample distribution.
    """
    from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo

    cfg = settings(EXPERIMENT)
    lo_pct, hi_pct = cfg["interval_percentiles"]
    rng = np.random.default_rng(seed)
    n = len(names)
    samples = {name: [] for name in names}
    pair_arrays = {k: np.asarray(v, dtype=float) for k, v in trial_wins.items()}  # (trials,)
    for _ in range(boots):
        wins = np.zeros((n, n))
        games = np.zeros((n, n))
        for (i, j), arr in pair_arrays.items():
            t = arr[rng.integers(0, len(arr), len(arr))]
            wa = t.sum()
            g = 2 * len(t)
            wins[i, j], wins[j, i] = wa, g - wa
            games[i, j] = games[j, i] = g
        elo = bradley_terry_elo(
            RoundRobin(names, games, wins),
            anchor_name=anchor,
            iterations=cfg["bootstrap_iterations"],
        )
        for name in names:
            samples[name].append(elo[name])
    return {
        name: (
            float(2 * point_elo[name] - np.percentile(v, hi_pct)),
            float(2 * point_elo[name] - np.percentile(v, lo_pct)),
        )
        for name, v in samples.items()
    }


def build_jobs(args, labels):
    """Return the play jobs, one per pairing and shard, slowest pairings first."""
    stride = settings(EXPERIMENT)["bank_seed_stride"]
    jobs = []
    for pi, ((sa, na), (sb, nb)) in enumerate(
        combinations(zip(args.agents, labels, strict=True), 2)
    ):
        for sh in range(args.shards):
            jobs.append(
                (sa, sb, na, nb, args.seed + stride * pi + sh, args.pairs_per_shard, args.seed + sh)
            )
    jobs.sort(key=lambda j: -sum(x in SLOW_AGENTS for x in (j[2], j[3])))
    return jobs


def play_round_robin(args, labels, jobs):
    """Play every job in a worker pool and return the tallies, trial wins, logs and wall time."""
    import multiprocessing as mp

    n = len(labels)
    idx = {name: i for i, name in enumerate(labels)}
    t0 = time.perf_counter()
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    points_sum = np.zeros((n, n))  # a-points summed over games, for the points matrix
    shard_wins: dict[tuple[int, int], dict] = {}  # (i, j) -> {bank seed: wins (0-2) per trial}
    all_logs = []
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_play_job, jobs):
            i, j = idx[r["a"]], idx[r["b"]]
            if j < i:  # canonical upper-triangle key
                i, j = j, i
                r["single"] = [-p for p in r["single"]]
                r["mean_diff"] = -r["mean_diff"]
                r["a"], r["b"] = r["b"], r["a"]
            sp = np.asarray(r["single"])
            wa = int((sp > 0).sum())
            wins[i, j] += wa
            wins[j, i] += len(sp) - wa
            games[i, j] += len(sp)
            games[j, i] += len(sp)
            points_sum[i, j] += sp.sum()
            points_sum[j, i] -= sp.sum()
            shard_wins.setdefault((i, j), {})[r["bank_seed"]] = [
                int(sp[2 * t] > 0) + int(sp[2 * t + 1] > 0) for t in range(len(sp) // 2)
            ]
            all_logs.extend(r["logs"])
            done += 1
            print(
                f"  [{done}/{len(jobs)}] {r['a']} vs {r['b']}: "
                f"ppg {r['mean_diff']:+.3f} ({r['wall']:.0f}s) | "
                f"elapsed {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    # Sort the pairings and their shards, so the bootstrap does not depend on finishing order.
    trial_wins = {
        key: [w for seed in sorted(shards) for w in shards[seed]]
        for key, shards in sorted(shard_wins.items())
    }
    return wins, games, points_sum, trial_wins, all_logs, time.perf_counter() - t0


def score_pr(all_logs, labels, plies, workers):
    """Score every logged decision with the oracle and return PR, decision counts and wall time."""
    print(f"scoring PR: {len(all_logs)} decisions at {plies}-ply ...", flush=True)
    t1 = time.perf_counter()
    from active_oracle_evaluation.eval.oracle import OracleConfig
    from active_oracle_evaluation.eval.scoring import score_decisions_parallel

    result = score_decisions_parallel(all_logs, OracleConfig(plies=plies), workers=workers)
    pr = {name: result.by_agent[name].pr for name in labels if name in result.by_agent}
    n_dec = {name: result.by_agent[name].n_checker for name in pr}
    return pr, n_dec, time.perf_counter() - t1


def validity(run, strong_tier):
    """Add the PR and Elo orders, the Kendall taus and the PR-gap fit to the run."""
    from active_oracle_evaluation.eval.validity import (
        kendall_tau,
        kendall_tau_pvalue,
        pr_gap_winprob_fit,
    )

    pr, elo = run["pr"], run["elo"]
    pr_order = sorted(pr, key=lambda x: pr[x])
    elo_order = sorted(pr, key=lambda x: -elo[x])
    tau = kendall_tau(pr_order, elo_order)
    tau_p = kendall_tau_pvalue(pr_order, elo_order)
    strong = [s for s in strong_tier.split(",") if s in pr]
    tau_strong = kendall_tau(
        [x for x in pr_order if x in strong], [x for x in elo_order if x in strong]
    )
    tau_strong_p = kendall_tau_pvalue(
        [x for x in pr_order if x in strong], [x for x in elo_order if x in strong]
    )
    slope, r2, n_fit = pr_gap_winprob_fit(pr, run["wins"], run["games"], list(pr))
    run.update(
        pr_order=pr_order,
        elo_order=elo_order,
        tau=tau,
        tau_p=tau_p,
        strong=strong,
        tau_strong=tau_strong,
        tau_strong_p=tau_strong_p,
        slope=slope,
        r2=r2,
        n_fit=n_fit,
    )


def main() -> None:
    """Play the round-robin, fit Elo, score PR and write the Markdown and JSON reports."""
    args = build_parser().parse_args()
    cfg = settings(EXPERIMENT)
    plies, anchor_name = cfg["oracle_plies"], cfg["anchor"]
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    labels = [spec_label(s) for s in args.agents]
    n = len(labels)
    pairs_total = args.pairs_per_shard * args.shards
    print(
        f"pool ({n}): {', '.join(labels)} | {n * (n - 1) // 2} pairings x "
        f"{pairs_total} duplicate pairs ({args.shards} shards) = "
        f"{n * (n - 1) * pairs_total} games",
        flush=True,
    )

    jobs = build_jobs(args, labels)
    wins, games, points_sum, trial_wins, all_logs, t_play = play_round_robin(args, labels, jobs)

    from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo

    points = np.divide(points_sum, games, out=np.zeros_like(points_sum), where=games > 0)
    rr = RoundRobin(labels, games, wins)
    anchor = anchor_name if anchor_name in labels else None
    elo = bradley_terry_elo(rr, anchor_name=anchor)
    print("bootstrapping Elo CIs ...", flush=True)
    cis = bootstrap_elo_cis(labels, trial_wins, anchor, args.boots, args.seed, elo)

    pr, n_dec, t_pr = score_pr(all_logs, labels, plies, args.workers)
    run = {
        "labels": labels,
        "anchor_name": anchor_name,
        "plies": plies,
        "pairs_total": pairs_total,
        "shards": args.shards,
        "seed": args.seed,
        "workers": args.workers,
        "wins": rr.wins,
        "games": rr.games,
        "points": points,
        "trial_wins": trial_wins,
        "elo": elo,
        "cis": cis,
        "pr": pr,
        "n_dec": n_dec,
        "t_play": t_play,
        "t_pr": t_pr,
    }
    validity(run, args.strong_tier)
    report = render_report(run)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary(run), indent=1), encoding="utf-8")
    print(f"\nwrote {out} (+ .json)\n")
    print(report)


if __name__ == "__main__":
    main()
