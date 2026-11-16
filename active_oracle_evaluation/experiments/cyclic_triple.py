"""Ratings on a deliberately cyclic triple of agents.

Builds a rock-paper-scissors triple from board-only agents that share one network (see
cyclic_triple_agent). The head-to-head interaction is cyclic by construction while
unconditional play is near-identical. The experiment certifies the cycle with a duplicate-dice
round robin, scores the triple and two reference agents (always strong, always weak) by suite
Performance Rating, and rates the field with Bradley-Terry Elo (fit and misfit), maxent Nash and
alpha-rank. It also fits alpha-rank to the transitive seven-agent precision panel.

Run from the repo root, in order (settings in the [cyclic_triple] table of config/experiments.toml):

    python -m active_oracle_evaluation.experiments.cyclic_triple rr --pairs 400 --workers 15
    python -m active_oracle_evaluation.experiments.cyclic_triple pr
    python -m active_oracle_evaluation.experiments.cyclic_triple analyze

rr writes cyclic_triple_rr.json and pr writes cyclic_triple_pr.json; analyze reads both and
writes cyclic_triple.json and cyclic_triple.md, all in outputs/cyclic_triple/.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.cyclic_triple_agent import (
    build_agent,
    play_shard,
    signature_points,
)
from active_oracle_evaluation.experiments.cyclic_triple_ratings import (
    alpha_rank,
    bt_theta,
    maxent_nash_3,
)
from active_oracle_evaluation.experiments.cyclic_triple_report import AGENTS, write_markdown
from active_oracle_evaluation.paths import DATA, REPO, output_dir

# Torch inference runs on the CPU so the agent choices are reproducible.
os.environ.setdefault("BACKGAMMON_FORCE_CPU", "1")

EXPERIMENT = "cyclic_triple"
PRECISION_JSON = DATA / "round_robins" / "intransitivity_precision.json"
RR_NAME = "cyclic_triple_rr.json"
PR_NAME = "cyclic_triple_pr.json"
RUN = "python -m active_oracle_evaluation.experiments.cyclic_triple"
# Each pairing lists the predator first.
PAIRINGS = [(0, 1), (1, 2), (2, 0)]
# Kept under this name for callers that import it from this module.
_alpha_rank = alpha_rank


def _sampler_spec(spec):
    """Return an agent spec with its checkpoint path made absolute against the repo root."""
    label, sep, path = spec.partition(":")
    return f"{label}:{REPO / path}" if sep else spec


def _pool_pairing(rs):
    """Pool the shards of one pairing by inverse-variance weighting and sum the counters."""
    diffs_mean = np.array([r["mean_diff"] for r in rs])
    ses = np.array([r["paired_stderr"] for r in rs])
    w = 1.0 / ses**2
    mean = float(np.sum(w * diffs_mean) / np.sum(w))
    se = float(1.0 / math.sqrt(np.sum(w)))
    wins = sum(r["a_wins"] for r in rs)
    games = sum(r["games"] for r in rs)
    counters = {}
    for key in rs[0]["counters_a"]:
        counters[key] = sum(r["counters_a"][key] for r in rs) + sum(
            r["counters_b"][key] for r in rs
        )
    return {
        "ppg": mean,
        "se": se,
        "z": mean / se,
        "wins": wins,
        "games": games,
        "winrate": wins / games,
        "counters_pooled": counters,
    }


def cmd_rr(args, cfg):
    """Run the three-pairing round robin in parallel shards and pool the shard results."""
    import multiprocessing as mp

    t0 = time.time()
    shards = []
    n_shards = cfg["n_shards"]
    per = args.pairs // n_shards
    for pi, (ia, ib) in enumerate(PAIRINGS):
        for s in range(n_shards):
            dice_seed = cfg["dice_seed_base"] * (pi + 1) + s
            agent_seed = cfg["agent_seed_base"] * (pi + 1) + s
            shards.append((ia, ib, per, dice_seed, agent_seed, cfg))
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(play_shard, shards):
            results.append(r)
            done = len(results)
            print(
                f"[{done}/{len(shards)}] pair {r['pair']} ppg {r['mean_diff']:+.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    combined = {}
    for ia, ib in PAIRINGS:
        rs = [r for r in results if r["pair"] == [ia, ib]]
        combined[f"cyc{ia}_vs_cyc{ib}"] = _pool_pairing(rs)
    payload = {
        "pairings": combined,
        "pairs_per_pairing": per * n_shards,
        "delta": cfg["delta"],
        "eps": cfg["eps"],
        "sig": signature_points(cfg),
        "wall_s": time.time() - t0,
    }
    (output_dir(EXPERIMENT) / RR_NAME).write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


def _build_suite(cfg, t0):
    """Build the Performance Rating suite from the sampler agents' self-play."""
    from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
    from active_oracle_evaluation.eval.agent_specs import resolve_agent
    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig
    from active_oracle_evaluation.eval.suite import PositionSuite, build_position_suite

    enc = BinarySlotEncoder()
    oracle = GnubgOracle(OracleConfig(plies=cfg["oracle_plies"]))
    samplers = [resolve_agent(_sampler_spec(s), enc) for s in cfg["samplers"]]
    suite_seed = cfg["suite_seed"]
    positions = []
    for i, (_n, agent) in enumerate(samplers):
        sub = build_position_suite(
            cfg["positions_per_sampler"],
            oracle=oracle,
            seed=suite_seed + cfg["sampler_seed_step"] * (i + 1),
            sampler=agent,
        )
        positions.extend(sub.positions)
    suite = PositionSuite(tuple(positions), seed=suite_seed, oracle_version=oracle.cfg.version)
    print(f"suite: {len(suite.positions)} positions, build {time.time() - t0:.0f}s", flush=True)
    return suite


def cmd_pr(cfg):
    """Score the triple and the two reference agents by Performance Rating on a fresh suite."""
    from active_oracle_evaluation.backgammon.core.game import Game
    from active_oracle_evaluation.eval.suite import SuiteScore, paired_contrast, pr_ci

    t0 = time.time()
    suite = _build_suite(cfg, t0)
    eval_seed = cfg["eval_seed"]
    specs = [("cyc0", 0), ("cyc1", 1), ("cyc2", 2), ("strong_ref", "strong"), ("weak_ref", "weak")]
    agents = {name: build_agent(spec, eval_seed, cfg) for name, spec in specs}
    scores = {}
    for name, agent in agents.items():
        t1 = time.time()
        losses = []
        for pos in suite.positions:
            game = Game(board=pos.board)
            game.set_dice(pos.dice)
            losses.append(pos.loss_of(agent.choose_move(game).afterstate))
        sc = SuiteScore(name, tuple(losses))
        scores[name] = sc
        lo, hi = pr_ci(sc, seed=eval_seed)
        print(f"{name}: PR {sc.pr:.2f} [{lo:.2f},{hi:.2f}] ({time.time() - t1:.0f}s)", flush=True)

    contrasts = {}
    for a, b in [("cyc0", "cyc1"), ("cyc1", "cyc2"), ("cyc2", "cyc0")]:
        c = paired_contrast(scores[a], scores[b], seed=eval_seed)
        contrasts[f"{a}-{b}"] = {
            "delta_pr": c.delta_pr,
            "ci95": [c.ci_low, c.ci_high],
            "distinguishable": c.distinguishable,
        }
        print(f"{a}-{b}: dPR {c.delta_pr:+.3f} [{c.ci_low:+.3f},{c.ci_high:+.3f}]", flush=True)

    payload = {
        "suite": {"seed": cfg["suite_seed"], "n": len(suite.positions)},
        "pr": {k: {"pr": v.pr, "ci": list(pr_ci(v, seed=eval_seed))} for k, v in scores.items()},
        "cyc_contrasts": contrasts,
        "mode_counters": {k: agents[k].counters for k in agents},
        "wall_s": time.time() - t0,
    }
    pr_json = output_dir(EXPERIMENT) / PR_NAME
    pr_json.write_text(json.dumps(payload, indent=1))
    print(f"done {time.time() - t0:.0f}s -> {pr_json}", flush=True)


def _win_matrix(rr):
    """Return the 3x3 win-rate matrix (predator rows) and the round-robin edges."""
    P = np.full((3, 3), 0.5)
    edges = {}
    for key, cell in rr["pairings"].items():
        ia = int(key[3])
        ib = int(key[-1])
        P[ia, ib] = cell["winrate"]
        P[ib, ia] = 1.0 - cell["winrate"]
        edges[key] = cell
    return P, edges


def _bt_misfit(P, edges, games, s):
    """Return each edge's observed win rate, Bradley-Terry prediction and excess in sigmas."""
    misfit = {}
    for key in edges:
        ia, ib = int(key[3]), int(key[-1])
        pred = s[ia] / (s[ia] + s[ib])
        se_bin = math.sqrt(P[ia, ib] * (1 - P[ia, ib]) / games)
        misfit[key] = {
            "observed": P[ia, ib],
            "bt_predicted": pred,
            "excess_sigmas": (P[ia, ib] - pred) / se_bin,
        }
    return misfit


def _transitive_panel_alpha_rank(alphas):
    """Fit alpha-rank to the seven-agent precision panel's round-robin win rates."""
    prec = json.loads(PRECISION_JSON.read_text())
    agents7 = prec["agents"]
    P7 = np.full((7, 7), 0.5)
    for pr_ in prec["pairings"]:
        i, j = agents7.index(pr_["a"]), agents7.index(pr_["b"])
        P7[i, j] = pr_["a_rate"]
        P7[j, i] = 1.0 - pr_["a_rate"]
    return {"agents": agents7, "stationary": alpha_rank(P7, alphas)}


def _read_inputs(out):
    """Read the round-robin result (required) and the suite PR result (optional)."""
    rr_json = out / RR_NAME
    pr_json = out / PR_NAME
    if not rr_json.exists():
        sys.exit(f"missing {rr_json}; run: {RUN} rr")
    rr = json.loads(rr_json.read_text())
    pr = json.loads(pr_json.read_text()) if pr_json.exists() else None
    return rr, pr


def cmd_analyze(cfg):
    """Fit BT-Elo, maxent Nash and alpha-rank to the round robin and write the results."""
    out = output_dir(EXPERIMENT)
    rr, pr = _read_inputs(out)
    P, edges = _win_matrix(rr)
    A = (P - P.T) / 2.0

    # Bradley-Terry Elo forced onto the cyclic matrix, with per-edge lack of fit.
    games = rr["pairings"]["cyc0_vs_cyc1"]["games"]
    theta = bt_theta(P, games, cfg["bt_iterations"], cfg["bt_tolerance"])
    elo = theta * 400.0 / math.log(10.0)
    misfit = _bt_misfit(P, edges, games, np.exp(theta))

    p_star = maxent_nash_3(A, cfg["nash_tolerance"])
    nash_skill = (A @ p_star) * 2.0
    viol = float(np.max(A @ p_star))

    alphas = cfg["alphas"]
    alpha_cyclic = alpha_rank(P, alphas)
    alpha_transitive = _transitive_panel_alpha_rank(alphas)

    result = {
        "construction": {
            "sig": rr["sig"],
            "delta": rr["delta"],
            "eps": rr["eps"],
            "pairs_per_pairing": rr["pairs_per_pairing"],
        },
        "cycle_edges": {
            k: {kk: v[kk] for kk in ("ppg", "se", "z", "winrate", "wins", "games")}
            for k, v in edges.items()
        },
        "cycle_certified": all(v["z"] > cfg["cycle_min_z"] for v in edges.values()),
        "bt_elo": {AGENTS[i]: float(elo[i]) for i in range(3)},
        "bt_misfit": misfit,
        "maxent_nash_mixture": {AGENTS[i]: float(p_star[i]) for i in range(3)},
        "nash_feasibility_violation": viol,
        "nash_skill_winrate_margin": {AGENTS[i]: float(nash_skill[i]) for i in range(3)},
        "alpha_rank_cyclic": {"agents": AGENTS, "stationary_by_alpha": alpha_cyclic},
        "alpha_rank_transitive_panel": alpha_transitive,
        "suite_pr": pr,
    }
    out_json = out / f"{EXPERIMENT}.json"
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(EXPERIMENT, result, out / f"{EXPERIMENT}.md")
    print(json.dumps({k: v for k, v in result.items() if k != "suite_pr"}, indent=1))
    print(f"wrote {out_json}")


def main():
    """Parse the command line and dispatch to one command."""
    cfg = settings(EXPERIMENT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["rr", "pr", "analyze"])
    p.add_argument("--pairs", type=int, default=cfg["pairs"])
    p.add_argument("--workers", type=int, default=cfg["workers"])
    args = p.parse_args()
    if args.cmd == "rr":
        cmd_rr(args, cfg)
    elif args.cmd == "pr":
        cmd_pr(cfg)
    else:
        cmd_analyze(cfg)


if __name__ == "__main__":
    main()
