"""Bootstrap intervals for the anchored-Elo and oracle-PR columns of the validity table."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def boot_p(diff) -> float:
    """Two-sided bootstrap p-value for a zero difference: twice the smaller tail, capped at 1."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    below, above = float((d <= 0).mean()), float((d >= 0).mean())
    return float(min(1.0, 2.0 * min(below, above)))


def _difference_summary(diff, percentiles, mean_fn, percentile_fn) -> dict:
    """Summarise one bootstrap difference: mean, interval, resolved flag and p-value."""
    dlo, dhi = (float(v) for v in percentile_fn(diff, percentiles))
    return {
        "mean": float(mean_fn(diff)),
        "lo": dlo,
        "hi": dhi,
        "resolved": bool(dlo > 0 or dhi < 0),
        "p": boot_p(diff),
    }


def _elo_replicates(names, wins, games, n_boot, seed, refit) -> np.ndarray:
    """Refit Elo on win counts redrawn per pairing; return one row of ratings per replicate."""
    p = np.divide(wins, games, out=np.zeros_like(wins), where=games > 0)
    rng = np.random.default_rng(seed)
    n = len(names)
    reps = np.empty((n_boot, n))
    for b in range(n_boot):
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if games[i, j] <= 0:
                    continue
                # Resampling a pairing's binary outcomes with replacement is the same as drawing
                # its win count from Binomial(n_ij, p_ij).
                wij = rng.binomial(int(games[i, j]), p[i, j])
                w[i, j], w[j, i] = wij, games[i, j] - wij
        e = refit(names, w, games)
        reps[b] = [e[x] for x in names]
    return reps


def elo_intervals(
    validity_json: Path, cluster, n_boot: int, seed: int, percentiles, tolerance: float
) -> dict:
    """Bootstrap the anchored Elo, resampling game outcomes clustered within pairing."""
    from active_oracle_evaluation.experiments.cluster_validity import _refit_elo

    d = json.loads(validity_json.read_text())
    names = d["names"]
    wins = np.asarray(d["wins"], dtype=float)
    games = np.asarray(d["games"], dtype=float)

    fitted = _refit_elo(names, wins, games)
    worst = max(abs(fitted[n] - d["elo"][n]) for n in names)
    assert worst < tolerance, f"refit does not reproduce the stored Elo (max diff {worst:g})"

    reps = _elo_replicates(names, wins, games, n_boot, seed, _refit_elo)
    lo, hi = np.percentile(reps, percentiles, axis=0)
    idx = {x: k for k, x in enumerate(names)}
    out = {
        "per_agent": {
            x: {
                "elo": d["elo"][x],
                "lo": float(lo[k]),
                "hi": float(hi[k]),
                "width": float(hi[k] - lo[k]),
            }
            for x, k in idx.items()
        },
        "pairwise_differences": {},
        "correlation": {},
    }
    for a, b_ in itertools.combinations(cluster, 2):
        diff = reps[:, idx[a]] - reps[:, idx[b_]]
        out["pairwise_differences"][f"{a} - {b_}"] = _difference_summary(
            diff, percentiles, np.mean, np.percentile
        )
    # Correlation among the agents that actually vary (the anchor has zero variance).
    varying = [x for x in names if reps[:, idx[x]].std() > 0]
    corr = np.corrcoef(np.stack([reps[:, idx[x]] for x in varying]))
    out["correlation"] = {
        "agents": varying,
        "matrix": corr.tolist(),
        "min_offdiag": float(np.min(corr[~np.eye(len(varying), dtype=bool)])),
        "max_offdiag": float(np.max(corr[~np.eye(len(varying), dtype=bool)])),
    }
    return out


def _replay_and_score(agent_specs, dice_seed: int, pairs: int, plies: int, workers: int):
    """Replay the duplicate-dice round-robin and score every decision with the oracle."""
    from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
    from active_oracle_evaluation.eval import DiceBank, play_duplicate_pair
    from active_oracle_evaluation.eval.agent_specs import resolve_agent
    from active_oracle_evaluation.eval.oracle import OracleConfig
    from active_oracle_evaluation.eval.scoring import decision_losses_parallel

    encoder = BinarySlotEncoder()
    agents = dict(resolve_agent(s, encoder) for s in agent_specs)

    t0 = time.perf_counter()
    bank = DiceBank(dice_seed)
    logs, pairing_of = [], []
    for (na, a), (nb, b) in itertools.combinations(agents.items(), 2):
        _res, pair_logs = play_duplicate_pair(a, b, bank, pairs, log=True)
        for log in pair_logs:
            logs.append(replace(log, agent=(na if log.agent == "a" else nb)))
            pairing_of.append(f"{na}|{nb}")
    print(f"  replayed {len(logs)} decisions in {time.perf_counter() - t0:.0f}s; scoring ...")

    t1 = time.perf_counter()
    detail = decision_losses_parallel(logs, OracleConfig(plies=plies), workers=workers)
    print(f"  scored in {time.perf_counter() - t1:.0f}s")

    # Keep only scored rows.
    rows = [
        (log.agent, f"{pairing_of[i]}#{log.trial}", float(detail[i]["loss"]))
        for i, log in enumerate(logs)
        if detail[i] is not None
    ]
    return list(agents), rows


def _save_losses(losses_out: Path, names, clusters, cid, rows, pr_per_equity) -> None:
    """Save one row per scored decision: agent index, cluster (pairing#trial) index, loss."""
    np.savez_compressed(
        losses_out,
        agent_names=np.array(names),
        cluster_labels=np.array(clusters),
        agent=np.array([names.index(a) for a, _c, _l in rows], dtype=np.uint8),
        cluster=np.array([cid[c] for _a, c, _l in rows], dtype=np.uint16),
        loss=np.array([loss for _a, _c, loss in rows], dtype=np.float32),
        pr_per_equity=np.float64(pr_per_equity),
    )
    print(f"  persisted {len(rows)} per-decision losses -> {losses_out}")


def _pr_replicates(names, by_agent, n_clusters, n_boot, seed, pr_per_equity) -> dict:
    """Resample (pairing, trial) clusters with replacement; return PR replicates per agent."""
    rng = np.random.default_rng(seed)
    reps = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, n_clusters)
        for n in names:
            vals = [x for k in draw for x in by_agent[n][k]]
            reps[n][b] = np.mean(vals) * pr_per_equity
    return reps


def pr_intervals(
    validity_json: Path, run, n_boot: int, seed: int, workers: int, losses_out: Path | None
) -> dict:
    """Replay the duplicate-dice round-robin, keep per-decision losses, bootstrap on trial.

    ``run`` holds agent_specs, dice_seed, pairs, plies, cluster and percentiles.
    """
    from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY

    percentiles = run["percentiles"]
    names, rows = _replay_and_score(
        run["agent_specs"], run["dice_seed"], run["pairs"], run["plies"], workers
    )
    published = json.loads(validity_json.read_text())["pr"]
    pr_now = {
        n: float(np.mean([loss for a, _c, loss in rows if a == n]) * PR_PER_EQUITY) for n in names
    }
    worst = max(abs(pr_now[n] - published[n]) for n in names)
    print(f"  replay vs published PR: max |diff| = {worst:.4f}")

    # The resampling unit is one duplicate-dice trial of one pairing, because the two seats of a
    # trial share dice and are not independent.
    clusters = sorted({c for _a, c, _l in rows})
    cid = {c: k for k, c in enumerate(clusters)}
    by_agent = {n: [[] for _ in clusters] for n in names}
    for a, c, loss in rows:
        by_agent[a][cid[c]].append(loss)

    if losses_out is not None:
        _save_losses(losses_out, names, clusters, cid, rows, PR_PER_EQUITY)

    reps = _pr_replicates(names, by_agent, len(clusters), n_boot, seed, PR_PER_EQUITY)

    # Pairwise differences on the same replicates, as for the Elo column.
    pairwise = {
        f"{a} - {b_}": _difference_summary(
            reps[a] - reps[b_], percentiles, np.nanmean, np.nanpercentile
        )
        for a, b_ in itertools.combinations(run["cluster"], 2)
    }

    return {
        "pairwise_differences": pairwise,
        "replay_max_abs_diff_vs_published": worst,
        "n_decisions_scored": len(rows),
        "n_clusters": len(clusters),
        "per_agent": {
            n: {
                "pr_published": published[n],
                "pr_replay": pr_now[n],
                "lo": float(np.nanpercentile(reps[n], percentiles[0])),
                "hi": float(np.nanpercentile(reps[n], percentiles[1])),
            }
            for n in names
        },
    }


def adjusted_counts(pairwise: dict, alpha: float) -> dict:
    """Count the pairs that resolve marginally, under Holm, and under Benjamini-Hochberg."""
    items = [(k, v["p"]) for k, v in pairwise.items() if v.get("p") == v.get("p")]
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    holm, running = set(), 0.0
    for i, (k, pv) in enumerate(order):
        running = max(running, pv * (m - i))
        if running <= alpha:
            holm.add(k)
        else:
            break
    kmax = 0
    for i, (k, pv) in enumerate(order, start=1):
        if pv <= alpha * i / m:
            kmax = i
    bh = {k for k, _ in order[:kmax]}
    return {
        "n_pairs": m,
        "marginal": sum(bool(v["resolved"]) for v in pairwise.values()),
        "holm": len(holm),
        "bh": len(bh),
        "bh_resolved": sorted(bh),
        "holm_resolved": sorted(holm),
    }
