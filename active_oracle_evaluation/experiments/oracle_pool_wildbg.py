"""wildbg as a tenth agent in the nine-agent anchored-Elo round-robin.

The experiment plays wildbg's nine pairings against the agents of the nine-agent pool, on
seeded duplicate-dice banks with pairing indices 36..44 (the nine-agent pool uses 0..35). It
merges these games into the pool's win, game and points matrices, refits the anchored
Bradley-Terry Elo with pivotal 95% intervals from a game-level binomial bootstrap, and
recomputes the validity statistics for the ten-agent pool: the Kendall tau between the PR and
Elo orders, the slope and R2 of the PR-gap to log-odds fit, and wildbg's mean residual.

wildbg's PR is scored on its own unforced decisions with the gnubg-nn 0-ply oracle. The other
nine agents keep their games, PR and decision counts from the nine-agent pool.

It needs a wildbg server at ``http://localhost:8080``. With the paper's settings:

    python -m active_oracle_evaluation.experiments.oracle_pool_wildbg --workers 14
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.oracle_pool_elo import (
    _play_job,
    _worker_init,
    spec_label,
)
from active_oracle_evaluation.experiments.oracle_pool_wildbg_report import (
    report_lines,
    summary_json,
)
from active_oracle_evaluation.paths import DATA, output_dir

EXPERIMENT = "oracle_pool_wildbg"
WILDBG_SPEC = "wildbg"
BASE_POOL = DATA / "round_robins" / "oracle_pool_elo.json"


def build_parser(cfg) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE_POOL)
    p.add_argument("--pairs-per-shard", type=int, default=cfg["pairs_per_shard"])
    p.add_argument("--shards", type=int, default=cfg["shards"])
    p.add_argument("--workers", type=int, default=cfg["workers"])
    p.add_argument("--boots", type=int, default=cfg["boots"])
    p.add_argument("--out", type=Path, default=None, help="report path (.json written beside)")
    return p


def binomial_bootstrap_cis(names, wins, games, anchor, boots, seed, point_elo, iterations):
    """Game-level binomial bootstrap of the anchored BT fit, returning pivotal 95% CIs."""
    from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo

    rng = np.random.default_rng(seed)
    n = len(names)
    samples = {name: [] for name in names}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if games[i][j] > 0]
    for _ in range(boots):
        w = np.zeros((n, n))
        g = np.zeros((n, n))
        for i, j in pairs:
            gij = int(games[i][j])
            wa = rng.binomial(gij, wins[i][j] / games[i][j])
            w[i, j], w[j, i] = wa, gij - wa
            g[i, j] = g[j, i] = gij
        elo = bradley_terry_elo(RoundRobin(names, g, w), anchor_name=anchor, iterations=iterations)
        for name in names:
            samples[name].append(elo[name])
    return {
        name: (
            float(2 * point_elo[name] - np.percentile(v, 97.5)),
            float(2 * point_elo[name] - np.percentile(v, 2.5)),
        )
        for name, v in samples.items()
    }


def clipped_log_odds(wins_ij, games_ij, clip):
    """Log-odds of a head-to-head win rate clipped to [clip, 1 - clip]."""
    p = min(max(wins_ij / games_ij, clip), 1 - clip)
    return math.log(p / (1 - p))


def build_jobs(cfg, args, labels9, seed):
    """Return one play job per (pairing, shard), slow agents first."""
    jobs = []
    for k, (spec, name) in enumerate(zip(cfg["agents"], labels9, strict=True)):
        pi = cfg["first_pairing_index"] + k
        for sh in range(args.shards):
            bank_seed = seed + cfg["bank_seed_stride"] * pi + sh
            jobs.append(
                (WILDBG_SPEC, spec, "wildbg", name, bank_seed, args.pairs_per_shard, seed + sh)
            )
    slow = cfg["slow_agents"]
    jobs.sort(key=lambda j: -sum(x in slow for x in (j[2], j[3])))
    return jobs


def play_wildbg(run, jobs, workers):
    """Play the jobs in a worker pool and merge wildbg's games into the run's matrices."""
    labels, base, n, wi = run["labels"], run["base"], run["n"], run["wi"]
    t0 = time.perf_counter()
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    points_mean = np.zeros((n, n))
    wins[:-1, :-1] = np.asarray(base["wins"])
    games[:-1, :-1] = np.asarray(base["games"])
    points_mean[:-1, :-1] = np.asarray(base["points"])
    points_sum_w = np.zeros(n)  # wildbg's summed points against each opponent
    wildbg_logs = []
    pair_ppg: dict[str, list] = {}
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_play_job, jobs):
            j = labels.index(r["b"])  # wildbg is always 'a' in these jobs
            sp = np.asarray(r["single"])
            wa = int((sp > 0).sum())
            wins[wi, j] += wa
            wins[j, wi] += len(sp) - wa
            games[wi, j] += len(sp)
            games[j, wi] += len(sp)
            points_sum_w[j] += sp.sum()
            wildbg_logs.extend(log for log in r["logs"] if log.agent == "wildbg")
            pair_ppg.setdefault(r["b"], []).append(r["mean_diff"])
            done += 1
            print(
                f"  [{done}/{len(jobs)}] wildbg vs {r['b']}: ppg {r['mean_diff']:+.3f} "
                f"({r['wall']:.0f}s) | elapsed {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    run["t_play"] = time.perf_counter() - t0
    for j in range(n - 1):
        if games[wi, j] > 0:
            points_mean[wi, j] = points_sum_w[j] / games[wi, j]
            points_mean[j, wi] = -points_mean[wi, j]
    run.update(wins=wins, games=games, points_mean=points_mean, pair_ppg=pair_ppg)
    return wildbg_logs


def fit_and_score(cfg, args, run, wildbg_logs):
    """Fit the anchored Elo with intervals, score wildbg's PR and compute the validity stats."""
    from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo
    from active_oracle_evaluation.eval.validity import (
        kendall_tau,
        kendall_tau_pvalue,
        pr_gap_winprob_fit,
    )

    labels, base, wins, games = run["labels"], run["base"], run["wins"], run["games"]
    anchor = cfg["anchor"]
    rr = RoundRobin(labels, games, wins)
    elo = bradley_terry_elo(rr, anchor_name=anchor)
    print("bootstrapping Elo CIs (game-level binomial) ...", flush=True)
    cis = binomial_bootstrap_cis(
        labels,
        games=games,
        wins=wins,
        anchor=anchor,
        boots=args.boots,
        seed=run["seed"],
        point_elo=elo,
        iterations=cfg["bootstrap_iterations"],
    )

    print(f"scoring wildbg PR: {len(wildbg_logs)} decisions at 0-ply ...", flush=True)
    t1 = time.perf_counter()
    from active_oracle_evaluation.eval.oracle import OracleConfig
    from active_oracle_evaluation.eval.scoring import score_decisions_parallel

    summary = score_decisions_parallel(
        wildbg_logs, OracleConfig(plies=cfg["plies"]), workers=args.workers
    )
    pr = dict(base["pr"])
    n_dec = dict(base["n_dec"])
    pr["wildbg"] = summary.by_agent["wildbg"].pr
    n_dec["wildbg"] = summary.by_agent["wildbg"].n_checker
    t_pr = time.perf_counter() - t1

    pr_order = sorted(pr, key=lambda x: pr[x])
    elo_order = sorted(pr, key=lambda x: -elo[x])
    tau = kendall_tau(pr_order, elo_order)
    tau_p = kendall_tau_pvalue(pr_order, elo_order)
    clip = cfg["winrate_clip"]
    slope, r2, n_fit = pr_gap_winprob_fit(pr, rr.wins, rr.games, labels, clip=clip)
    wildbg_resid = wildbg_residual(run, pr, slope, clip)
    return dict(
        elo=elo,
        cis=cis,
        pr=pr,
        n_dec=n_dec,
        t_pr=t_pr,
        pr_order=pr_order,
        elo_order=elo_order,
        tau=tau,
        tau_p=tau_p,
        slope=slope,
        r2=r2,
        n_fit=n_fit,
        wildbg_resid=wildbg_resid,
    )


def wildbg_residual(run, pr, slope, clip):
    """wildbg's mean residual against the pooled PR-gap to log-odds fit, over its pairings."""
    labels, labels9, wins, games = run["labels"], run["labels9"], run["wins"], run["games"]
    n, wi = run["n"], run["wi"]
    resids = [
        clipped_log_odds(wins[wi, j], games[wi, j], clip) - slope * (pr[name] - pr["wildbg"])
        for j, name in enumerate(labels9)
    ]
    xs, ys = [], []
    for i in range(n):
        for j in range(n):
            if i != j and games[i][j] > 0:
                xs.append(pr[labels[j]] - pr[labels[i]])
                ys.append(clipped_log_odds(wins[i][j], games[i][j], clip))
    _, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return float(np.mean(resids) - intercept)


def main() -> None:
    """Play wildbg's pairings, merge them into the nine-agent pool and write the report."""
    cfg = settings(EXPERIMENT)
    args = build_parser(cfg).parse_args()
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    base = json.loads(args.base.read_text())
    labels9 = base["labels"]
    seed = int(base["seed"])
    assert labels9 == [spec_label(s) for s in cfg["agents"]], "base pool != agents order"

    labels = labels9 + ["wildbg"]
    n = len(labels)
    run = dict(
        base=base,
        base_stem=args.base.stem,
        labels9=labels9,
        labels=labels,
        n=n,
        wi=n - 1,
        seed=seed,
        pairs_total=args.pairs_per_shard * args.shards,
        base_games=int(np.sum(base["games"])) // 2,
        first_pairing_index=cfg["first_pairing_index"],
    )

    jobs = build_jobs(cfg, args, labels9, seed)
    print(
        f"grafting wildbg: {len(labels9)} pairings x {run['pairs_total']} duplicate pairs = "
        f"{len(labels9) * 2 * run['pairs_total']} games "
        f"(base: {run['base_games']} untouched)",
        flush=True,
    )
    wildbg_logs = play_wildbg(run, jobs, args.workers)
    stats = fit_and_score(cfg, args, run, wildbg_logs)

    report = "\n".join(report_lines(run, stats))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(summary_json(run, stats), encoding="utf-8")
    print(f"\nwrote {out} (+ .json)\n")
    print(report)


if __name__ == "__main__":
    main()
