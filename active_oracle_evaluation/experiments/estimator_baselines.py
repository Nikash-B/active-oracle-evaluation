"""Estimator baselines at a matched label budget.

The estimand is the full-pool 2-ply Performance Rating (PR) of each agent. The 0-ply regret, cached
for every pool item, serves as the free predictor for the prediction-powered estimators and as the
input to the active design.

Estimators compared at each budget K:

* ``classical``        -- sample mean on a uniform draw.
* ``ht_active``        -- Horvitz-Thompson under the ``active_rank`` design built from 0-ply losses.
* ``hajek_active``     -- the ratio (Hajek) form of the same design.
* ``lure_active``      -- LURE under a sequential proposal with the same importance.
* ``ppi``              -- prediction-powered inference, lambda = 1, uniform draw.
* ``ppi_plus``         -- prediction-powered inference with variance-optimal lambda, uniform draw.
* ``ht_oracle_design`` -- Horvitz-Thompson under a design built from the 2-ply labels (skyline).

Each estimator runs under the sampling scheme its own derivation assumes. Fixed-budget sequential
halving (``bai_halving``) is reported for best-agent identification only.

Writes estimator_baselines.csv, estimator_baselines.md and estimator_baselines.json to
outputs/estimator_baselines/ for all eight agents, and to outputs/estimator_baselines/six_agent/
for the competitive cluster:

    python -m active_oracle_evaluation.experiments.estimator_baselines \\
        --ks 50 100 250 500 1000 --seeds 200
    python -m active_oracle_evaluation.experiments.estimator_baselines --seeds 200 \\
        --agents pubeval v1 v2 ppo az distilled \\
        --out-dir outputs/estimator_baselines/six_agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import eval_select as sel
from active_oracle_evaluation.eval.estimators import (
    classical_mean,
    hajek,
    horvitz_thompson,
    lure,
    ppi,
    ppi_plus,
    ppi_plus_lambda,
    sequential_halving_bai,
    sequential_without_replacement,
)
from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY, ScoreCache
from active_oracle_evaluation.experiments.estimator_baselines_report import write_outputs
from active_oracle_evaluation.experiments.estimator_baselines_stats import (
    ESTIMATORS,
    _se_ht,
    _se_lure,
    _se_srs,
    _tau,
    predictor_correlation_range,
)
from active_oracle_evaluation.paths import DATA, output_dir

EXPERIMENT = "estimator_baselines"
CHEAP = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
EXPENSIVE = DATA / "decision_suites" / "full_suite_scores_2ply.npz"
Z_95 = 1.96


def build_parser() -> argparse.ArgumentParser:
    """Command-line options for the estimator sweep."""
    cfg = settings(EXPERIMENT)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cheap", type=Path, default=CHEAP, help="predictor cache (0-ply)")
    parser.add_argument(
        "--expensive", type=Path, default=EXPENSIVE, help="label cache (2-ply), the estimand"
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(cfg["ks"]))
    parser.add_argument(
        "--seeds", type=int, default=cfg["seeds"], help="replicate draws per (estimator, K)"
    )
    parser.add_argument(
        "--eps", type=float, default=cfg["eps"], help="uniform floor for the design"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: outputs/estimator_baselines"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="restrict the panel to these agents (default: every agent in both caches). The "
        "active design reads the panel's own losses, so a subset changes the design too.",
    )
    return parser


def _fill_agent(s, a, draw, design, f_all, y_all, est, se, n):
    """Store one agent's PR estimates and SEs of every estimator for draw ``s``."""
    unif, pps, pps_o, lure_idx, lure_q = draw
    pi, pi_o = design["pi"], design["pi_o"]
    y, f = y_all[a], f_all[a]
    est["classical"][s, a] = classical_mean(y[unif]) * PR_PER_EQUITY
    est["ht_active"][s, a] = horvitz_thompson(y[pps], pi[pps], n) * PR_PER_EQUITY
    est["hajek_active"][s, a] = hajek(y[pps], pi[pps]) * PR_PER_EQUITY
    est["lure_active"][s, a] = lure(y[lure_idx], lure_q, n) * PR_PER_EQUITY
    est["ppi"][s, a] = ppi(y[unif], f[unif], f) * PR_PER_EQUITY
    est["ppi_plus"][s, a] = ppi_plus(y[unif], f[unif], f) * PR_PER_EQUITY
    est["ht_oracle_design"][s, a] = horvitz_thompson(y[pps_o], pi_o[pps_o], n) * PR_PER_EQUITY

    lam = ppi_plus_lambda(y[unif], f[unif])
    se["classical"][s, a] = _se_srs(y[unif], n) * PR_PER_EQUITY
    se["ppi"][s, a] = _se_srs(y[unif] - f[unif], n) * PR_PER_EQUITY
    se["ppi_plus"][s, a] = _se_srs(y[unif] - lam * f[unif], n) * PR_PER_EQUITY
    se["ht_active"][s, a] = _se_ht(y[pps], pi[pps], n) * PR_PER_EQUITY
    se["hajek_active"][s, a] = (
        _se_ht(y[pps] - np.average(y[pps], weights=1.0 / pi[pps]), pi[pps], n) * PR_PER_EQUITY
    )
    se["lure_active"][s, a] = _se_lure(y[lure_idx], np.asarray(lure_q), n) * PR_PER_EQUITY
    se["ht_oracle_design"][s, a] = _se_ht(y[pps_o], pi_o[pps_o], n) * PR_PER_EQUITY


def _one_draw(s, k, design, f_all, y_all, est, se, best_true, cfg) -> int:
    """Fill draw ``s`` of every estimator at budget ``k``; return 1 if halving finds the best."""
    n = design["n"]
    draw_seed = cfg["draw_seed_base"] + s
    rng = np.random.default_rng(draw_seed)
    unif = rng.choice(n, size=k, replace=False)
    pps = np.asarray(sel._pps_sample(design["pi"], seed=draw_seed), dtype=int)
    pps_o = np.asarray(sel._pps_sample(design["pi_o"], seed=draw_seed), dtype=int)
    lure_idx, lure_q = sequential_without_replacement(design["prop"], k, rng)
    lure_idx = np.asarray(lure_idx, dtype=int)
    draw = (unif, pps, pps_o, lure_idx, lure_q)
    for a in range(f_all.shape[0]):
        _fill_agent(s, a, draw, design, f_all, y_all, est, se, n)

    # Fixed-budget best-agent identification, steered by the 0-ply signal only.
    floor = cfg["importance_floor"]
    winner, _ = sequential_halving_bai(
        y_all,
        lambda surv: np.clip(sel.pairwise_regret_importance(f_all[surv]), floor, None),
        k,
        np.random.default_rng(cfg["halving_seed_base"] + s),
    )
    return int(winner == best_true)


def _score_rows(est, se, truth, k, n_seeds, best_true, bai_hits) -> list[dict]:
    """Bias, SD, RMSE, width, coverage, tau and top-1 rows of every estimator at budget ``k``."""
    rows = []
    for e in ESTIMATORS:
        err = est[e] - truth[None, :]
        half = Z_95 * se[e]
        covered = np.abs(err) <= half
        rows.append(
            {
                "estimator": e,
                "k": k,
                "bias": float(err.mean()),
                "sd": float(est[e].std(axis=0, ddof=1).mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "ci_width": float(np.nanmean(2.0 * half)),
                "coverage": float(np.nanmean(covered[np.isfinite(half)])),
                "tau": float(np.mean([_tau(est[e][s], truth) for s in range(n_seeds)])),
                "top1": float(np.mean([np.argmin(est[e][s]) == best_true for s in range(n_seeds)])),
            }
        )
    nan = float("nan")
    rows.append(
        {  # identification only: no PR estimate to score
            "estimator": "bai_halving",
            "k": k,
            "bias": nan,
            "sd": nan,
            "rmse": nan,
            "ci_width": nan,
            "coverage": nan,
            "tau": nan,
            "top1": bai_hits / n_seeds,
        }
    )
    return rows


def run(
    cheap: ScoreCache,
    exp: ScoreCache,
    ks,
    n_seeds: int,
    eps: float,
    names: list[str] | None = None,
    raw: dict | None = None,
) -> list[dict]:
    """Bias, SD, RMSE, interval width, coverage, tau and top-1 rows per (estimator, K).

    Pass ``raw={}`` to also receive, per K, the per-draw estimates and analytic SEs
    (``{k: {"est", "se", "truth", "names"}}``).
    """
    cfg = settings(EXPERIMENT)
    names = list(names) if names is not None else cheap.names
    n = cheap.n_pos
    f_all = np.stack([cheap.losses[a] for a in names])  # predictor: 0-ply
    y_all = np.stack([exp.losses[a] for a in names])  # label: 2-ply
    truth = y_all.mean(axis=1) * PR_PER_EQUITY

    # Active design from the free 0-ply losses; skyline design from the 2-ply labels.
    imp = sel.pairwise_regret_importance(f_all)
    imp_oracle = sel.pairwise_regret_importance(y_all)
    pi_full = np.asarray(sel.floor_mix(imp, eps=eps), dtype=float)
    pi_full_oracle = np.asarray(sel.floor_mix(imp_oracle, eps=eps), dtype=float)
    rows: list[dict] = []

    for k in ks:
        if k > n:
            continue
        design = {
            "n": n,
            "pi": np.asarray(sel.inclusion_probs(pi_full, k), dtype=float),
            "pi_o": np.asarray(sel.inclusion_probs(pi_full_oracle, k), dtype=float),
            # Strictly positive LURE proposal.
            "prop": np.clip(imp, cfg["importance_floor"], None) + imp.mean() * eps,
        }
        # est[name]: (n_seeds, n_agents) PR estimates; se[name]: the matching SEs.
        est = {e: np.zeros((n_seeds, len(names))) for e in ESTIMATORS}
        se = {e: np.full((n_seeds, len(names)), np.nan) for e in ESTIMATORS}
        bai_hits = 0
        best_true = int(np.argmin(truth))
        for s in range(n_seeds):
            bai_hits += _one_draw(s, k, design, f_all, y_all, est, se, best_true, cfg)

        if raw is not None:
            raw[k] = {
                "est": {e: v.copy() for e, v in est.items()},
                "se": {e: v.copy() for e, v in se.items()},
                "truth": truth.copy(),
                "names": list(names),
            }
        rows += _score_rows(est, se, truth, k, n_seeds, best_true, bai_hits)
        print(
            f"  K={k:5d}  "
            + "  ".join(
                f"{e}={[r for r in rows if r['estimator'] == e and r['k'] == k][0]['rmse']:.2f}"
                for e in ESTIMATORS
            )
        )
    return rows


def main() -> None:
    """Run the sweep, then write the CSV, the Markdown tables and the JSON summary."""
    args = build_parser().parse_args()
    cfg = settings(EXPERIMENT)
    cheap = ScoreCache(args.cheap)
    exp = ScoreCache(args.expensive)
    if cheap.names != exp.names or cheap.n_pos != exp.n_pos:
        raise SystemExit(
            "the two caches must cover the same agents and positions in the same order"
        )
    names = list(cheap.names)
    if args.agents:
        missing = [a for a in args.agents if a not in names]
        if missing:
            raise SystemExit(f"agents not in both caches: {missing}; available: {names}")
        names = [a for a in names if a in args.agents]  # keep the caches' order
    print(
        f"pool {cheap.n_pos:,} decisions x {len(names)} agents ({', '.join(names)}); "
        f"{args.seeds} draws per (estimator, K)"
    )
    rows = run(cheap, exp, args.ks, args.seeds, args.eps, names=names)
    out_dir = args.out_dir if args.out_dir is not None else output_dir(EXPERIMENT)
    ks_used = [k for k in args.ks if k <= cheap.n_pos]
    write_outputs(
        rows,
        out_dir,
        n_seeds=args.seeds,
        n_pool=cheap.n_pos,
        names=names,
        ks=ks_used,
        corr_range=predictor_correlation_range(cheap, exp, names, cfg["untrained_agents"]),
        baseline=cfg["saving_baseline"],
    )
    print("done.")


if __name__ == "__main__":
    main()
