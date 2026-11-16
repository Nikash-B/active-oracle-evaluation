"""Compare the scalar Performance Rating (PR) with Nash averaging on a round-robin panel.

Nash averaging (Balduzzi et al., 2018) reads the antisymmetric advantage matrix
``A_ij = 2 p_ij - 1`` as a symmetric zero-sum game, takes its maximum-entropy Nash equilibrium
``p*``, and rates each agent by its expected advantage against that mixture, ``(A p*)_i``. On a
transitive field the equilibrium is pure, so each rating reduces to the advantage against the
dominant agent. The experiment also fits Bradley-Terry to the same games and reports both
orderings against PR, computed from the full-suite score cache. It then duplicates each agent in
turn and records how far the other agents' Nash ratings and Elo move (the clone cells).

It reads the seven-agent round-robin data/round_robins/intransitivity_precision.json (15,000
games per pairing) and plays no games. Settings are read from the [nash_averaging] table of the
config. Run from the repository root:

    python -m active_oracle_evaluation.experiments.nash_averaging

It writes outputs/nash_averaging/nash_averaging.md (the report) and nash_averaging.json (the
same numbers plus the clone cells).
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo
from active_oracle_evaluation.experiments.nash_averaging_clones import clone_cells
from active_oracle_evaluation.experiments.nash_averaging_report import report_lines
from active_oracle_evaluation.paths import DATA, output_dir

EXPERIMENT = "nash_averaging"
SIDECAR = DATA / "round_robins" / "intransitivity_precision.json"
SCORE_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply_large.npz"


@functools.lru_cache(maxsize=1)
def experiment_settings():
    """Return the [nash_averaging] settings, read once."""
    return settings(EXPERIMENT)


def load_pr(cache: Path, cfg) -> dict[str, float]:
    """Full-suite oracle PR (lower is better) per agent, rounded to ``pr_decimals``."""
    scores = np.load(cache, allow_pickle=True)
    return {
        str(name): round(float(losses.mean()) * cfg["pr_scale"], cfg["pr_decimals"])
        for name, losses in zip(scores["names"], scores["losses"])
    }


def nash_equilibrium(A: np.ndarray, cfg=None) -> np.ndarray:
    """Maximum-entropy Nash equilibrium of the symmetric zero-sum game with antisymmetric payoff A.

    1. For antisymmetric A the game value is 0, so p is an equilibrium iff ``p >= 0``,
       ``sum p = 1`` and ``A p <= 0`` (no pure strategy gains against p). An LP finds one point
       of this set.
    2. Entropy is concave, so multiplicative (mirror) ascent that stays inside the set moves to the
       maximum-entropy equilibrium; on a unique equilibrium it returns that point.
    """
    cfg = experiment_settings() if cfg is None else cfg
    n = A.shape[0]
    # Stage 1: feasibility LP (zero objective).
    res = linprog(
        c=np.zeros(n),
        A_ub=A,
        b_ub=np.zeros(n),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"no equilibrium found: {res.message}")
    p = np.clip(res.x, 0.0, None)
    p /= p.sum()

    # Stage 2: entropy ascent; mirror updates keep p on the simplex, and ascent stops at the first
    # step that leaves the equilibrium set.
    for _ in range(cfg["ascent_iterations"]):
        grad = -(np.log(p + cfg["log_epsilon"]) + 1.0)  # d/dp of -sum p log p
        cand = p * np.exp(cfg["ascent_step"] * grad)
        cand /= cand.sum()
        if np.all(A @ cand <= cfg["feasibility_tol"]):  # still an equilibrium
            if np.max(np.abs(cand - p)) < cfg["ascent_convergence"]:
                break
            p = cand
        else:
            break
    return p


def kendall_tau(order_a: list[str], order_b: list[str]) -> tuple[float, int]:
    """Kendall tau between two orderings of the same items, and the number of discordant pairs."""
    rank_in_b = {x: i for i, x in enumerate(order_b)}
    seq = [rank_in_b[x] for x in order_a]
    discordant = sum(1 for i, j in itertools.combinations(range(len(seq)), 2) if seq[i] > seq[j])
    n_pairs = len(seq) * (len(seq) - 1) // 2
    return (n_pairs - 2 * discordant) / n_pairs, discordant


def advantage_matrix(sidecar, idx, n, cfg):
    """Return the win-probability matrix and the antisymmetric advantage matrix 2 P - 1."""
    win_prob = np.full((n, n), 0.5)
    for pr in sidecar["pairings"]:
        i, j = idx[pr["a"]], idx[pr["b"]]
        win_prob[i, j] = pr["a_rate"]
        win_prob[j, i] = 1.0 - pr["a_rate"]
    np.fill_diagonal(win_prob, 0.5)
    A = 2.0 * win_prob - 1.0
    assert np.allclose(A, -A.T, atol=cfg["antisymmetry_atol"]), (
        "advantage matrix must be antisymmetric"
    )
    return win_prob, A


def is_transitive(win_prob, names, idx):
    """Return True when no three agents form a cycle of head-to-head wins."""
    return all(
        not (
            win_prob[idx[x], idx[y]] > 0.5
            and win_prob[idx[y], idx[z]] > 0.5
            and win_prob[idx[z], idx[x]] > 0.5
        )
        for x, y, z in itertools.permutations(names, 3)
    )


def bradley_terry_order(sidecar, names, idx, n):
    """Fit Bradley-Terry to the sidecar's games and return the agents from best to worst."""
    games = np.zeros((n, n))
    wins = np.zeros((n, n))
    for pr in sidecar["pairings"]:
        i, j = idx[pr["a"]], idx[pr["b"]]
        games[i, j] = games[j, i] = pr["games"]
        wins[i, j] = pr["a_wins"]
        wins[j, i] = pr["games"] - pr["a_wins"]
    bt_raw = bradley_terry_elo(RoundRobin(names, games, wins))
    bt = np.asarray([bt_raw[a] for a in names])
    return sorted(names, key=lambda a: -bt[idx[a]]), {a: float(bt_raw[a]) for a in names}


def results_json(r, sidecar_name):
    """Return the JSON record of the report's numbers."""
    return {
        "source_matrix": sidecar_name,
        "agents": r["names"],
        "games_per_pair": r["games_per_pair"],
        "total_games": r["total_games"],
        "oracle_pr": r["oracle_pr"],
        "nash_rating": r["nash_rating"],
        "nash_mixture": r["nash_mass"],
        "nash_support": dict(r["support"]),
        "bt_elo": r["bt_elo"],
        "orders": {"nash": r["nash_order"], "bt": r["bt_order"], "pr": r["pr_order"]},
        "kendall_tau_pr_nash": r["tau"],
        "discordant_pr_nash": r["inv"],
        "kendall_tau_pr_bt": r["tau_bt"],
        "discordant_pr_bt": r["inv_bt"],
        "pairs": r["n"] * (r["n"] - 1) // 2,
        "transitive": r["transitive"],
        "clone_cells": r["clone_cells"],
    }


def main() -> None:
    """Rate the sidecar panel by Nash averaging and Bradley-Terry and write the report."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sidecar", type=Path, default=SIDECAR, help="round-robin JSON")
    ap.add_argument("--out", type=Path, default=None, help="Markdown report (JSON goes beside it)")
    ap.add_argument("--cache", type=Path, default=SCORE_CACHE, help="full-suite score cache")
    args = ap.parse_args()
    if args.out is None:
        args.out = output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    cfg = experiment_settings()

    oracle_pr = load_pr(args.cache, cfg)
    sidecar = json.loads(args.sidecar.read_text())
    names: list[str] = list(sidecar["agents"])
    idx = {a: i for i, a in enumerate(names)}
    n = len(names)

    win_prob, A = advantage_matrix(sidecar, idx, n, cfg)
    p = nash_equilibrium(A, cfg)
    nash_rating = A @ p

    nash_order = sorted(names, key=lambda a: -nash_rating[idx[a]])
    pr_order = sorted(names, key=lambda a: oracle_pr[a])
    tau, inv = kendall_tau(pr_order, nash_order)

    support = [(a, float(p[idx[a]])) for a in names if p[idx[a]] > cfg["support_threshold"]]
    transitive = is_transitive(win_prob, names, idx)
    bt_order, bt_elo = bradley_terry_order(sidecar, names, idx, n)
    tau_bt, inv_bt = kendall_tau(pr_order, bt_order)

    r = {
        "names": names,
        "n": n,
        "games_per_pair": sidecar["games_per_pair"],
        "total_games": len(sidecar["pairings"]) * sidecar["games_per_pair"],
        "oracle_pr": {a: oracle_pr[a] for a in names},
        "nash_rating": {a: float(nash_rating[idx[a]]) for a in names},
        "nash_mass": {a: float(p[idx[a]]) for a in names},
        "nash_order": nash_order,
        "pr_order": pr_order,
        "bt_order": bt_order,
        "bt_elo": bt_elo,
        "tau": tau,
        "inv": inv,
        "tau_bt": tau_bt,
        "inv_bt": inv_bt,
        "support": support,
        "transitive": transitive,
        "clone_cells": clone_cells(sidecar, cfg),
    }
    lines = report_lines(r, args.sidecar.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(results_json(r, args.sidecar.name), indent=1) + "\n")
    print(
        f"tau(PR, Nash) = {tau:+.3f}, {inv} discordant; "
        f"support {len(support)}/{n}; wrote {args.out}"
    )
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
