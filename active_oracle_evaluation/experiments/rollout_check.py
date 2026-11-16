"""Agreement of the agent ranking under a Monte Carlo playout oracle with the 0-ply ranking.

Samples positions from the frozen decision suite and scores each agent's cached move choice
against a Monte Carlo playout of the oracle's own network (``eval.rollout_oracle``). Per position
it plays out only the candidate afterstates (each agent's choice plus the 0-ply best move), so an
agent's regret is measured against the best of those candidates. It then compares the Performance
Rating (PR) ranking under the playout with the 0-ply and 2-ply rankings of the same positions,
read from the shipped score caches, by Kendall's tau. It needs gnubg-nn but no checkpoints.
Settings are in the [rollout_check] table of config/experiments.toml.

    python -m active_oracle_evaluation.experiments.rollout_check

It writes outputs/rollout_check/rollout_check.md (PR per agent under each oracle and both taus)
and rollout_check.json (the same numbers).
"""

from __future__ import annotations

import random
import time

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.decision_scoring import eligible_records
from active_oracle_evaluation.eval.decision_suite import afterstates_of, read_jsonl
from active_oracle_evaluation.eval.validity import kendall_tau
from active_oracle_evaluation.paths import DATA, repo_relative
from active_oracle_evaluation.results import write_results

EXPERIMENT = "rollout_check"
SUITES = DATA / "decision_suites"
SUITE = SUITES / "backgammon_oracle_suite_0ply_v1.jsonl"
CACHES = {
    "0-ply": SUITES / "full_suite_scores_0ply.npz",
    "2-ply": SUITES / "full_suite_scores_2ply.npz",
}


def load_inputs():
    """Return the eligible suite records and the 0-ply and 2-ply score caches, checked aligned."""
    records = eligible_records(read_jsonl(SUITE))
    caches = {name: np.load(path, allow_pickle=True) for name, path in CACHES.items()}
    ids = [r.position_id for r in records]
    for name, cache in caches.items():
        if [str(p) for p in cache["position_ids"]] != ids:
            raise SystemExit(f"suite and {name} cache list different positions")
        if list(cache["names"]) != list(caches["0-ply"]["names"]):
            raise SystemExit(f"{name} cache lists different agents")
    return records, caches


def playout_regrets(records, choices, sample, oracle):
    """Return each agent's playout regret per sampled position and the number of playouts run."""
    regret = np.zeros((choices.shape[0], len(sample)))
    n_candidates = 0
    for col, i in enumerate(sample):
        record = records[i]
        afters = afterstates_of(record)
        candidates = {int(c) for c in choices[:, i]}
        if record.best_action is not None:
            candidates.add(int(record.best_action))
        # An afterstate is opponent-to-act, so the mover's equity is minus its playout equity.
        mover_equity = {j: -oracle.equity(afters[j]) for j in sorted(candidates)}
        n_candidates += len(candidates)
        best = max(mover_equity.values())
        for a, c in enumerate(choices[:, i]):
            regret[a, col] = max(0.0, best - mover_equity[int(c)])
    return regret, n_candidates


def ranking(pr):
    """Return the agent names ordered from lowest (best) to highest PR."""
    return sorted(pr, key=pr.get)


def main() -> None:
    """Score the sampled positions with the playout oracle and compare the agent rankings."""
    from active_oracle_evaluation.eval.rollout_oracle import RolloutOracle  # lazy: gnubg-nn

    config = settings(EXPERIMENT)
    records, caches = load_inputs()
    names = [str(n) for n in caches["0-ply"]["names"]]
    rng = random.Random(config["sample_seed"])
    sample = sorted(rng.sample(range(len(records)), min(config["positions"], len(records))))

    oracle = RolloutOracle(playouts=config["playouts"], seed=config["seed"])
    start = time.perf_counter()
    regret, n_candidates = playout_regrets(records, caches["0-ply"]["choices"], sample, oracle)
    seconds = time.perf_counter() - start

    scale = config["pr_scale"]
    pr = {"playout": dict(zip(names, (regret.mean(axis=1) * scale).tolist()))}
    for name, cache in caches.items():
        pr[name] = dict(zip(names, (cache["losses"][:, sample].mean(axis=1) * scale).tolist()))
    orders = {oracle_name: ranking(values) for oracle_name, values in pr.items()}
    tau = {name: kendall_tau(orders["playout"], orders[name]) for name in CACHES}

    numbers = {
        "suite": repo_relative(SUITE),
        "positions_eligible": len(records),
        "positions_sampled": len(sample),
        "sample_seed": config["sample_seed"],
        "playouts_per_candidate": config["playouts"],
        "playout_seed": config["seed"],
        "candidates_played_out": n_candidates,
        "pr": pr,
        "ranking": orders,
        "kendall_tau_playout_vs": tau,
        "seconds": round(seconds, 1),
    }
    pr_rows = [[n] + [f"{pr[o][n]:.1f}" for o in pr] for n in orders["playout"]]
    tau_rows = [[f"playout vs {name}", f"{value:.3f}"] for name, value in tau.items()]
    tables = [
        ("PR on the sampled positions (lower is better)", ["agent", *pr], pr_rows),
        ("Ranking agreement", ["comparison", "Kendall's tau"], tau_rows),
    ]
    md_path, json_path = write_results(EXPERIMENT, tables, numbers)
    print(f"{len(sample)} of {len(records)} positions, {config['playouts']} playouts per candidate")
    print(f"{n_candidates} candidates played out in {seconds:.0f} s")
    for name, value in tau.items():
        print(f"Kendall's tau, playout vs {name}: {value:.3f}")
    print(f"wrote {md_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
