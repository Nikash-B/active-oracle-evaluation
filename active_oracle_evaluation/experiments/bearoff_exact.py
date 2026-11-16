"""Exact-value check of the oracle-regret rating on the solved bear-off slice.

Samples uniform-random decisions where both sides are fully home with at least one checker borne
off, so gammons are impossible. Each afterstate is labelled three ways: exactly, from a one-sided
bear-off table (cubeless equity = 1 - 2 * P(opponent on roll wins), by convolving the two
roll-count distributions), by the 0-ply oracle that labels the suite, and by a 2-ply oracle. A
seven-agent panel is scored against all three references, and the experiment reports the
oracle's equity error, the recovered ranking and the paired PR contrasts with bootstrap
intervals. Settings come from the [bearoff_exact] table of config/experiments.toml.
With the paper's settings:

    python -m active_oracle_evaluation.experiments.bearoff_exact --n 1500 --seed 227
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.bearoff_exact_report import write_markdown
from active_oracle_evaluation.paths import DATA, REPO, output_dir

# Torch inference runs on the CPU so the agent choices are reproducible.
os.environ.setdefault("BACKGAMMON_FORCE_CPU", "1")

EXPERIMENT = "bearoff_exact"
DB_PATH = DATA / "bearoff" / "bearoff_os_ar0183.npz"
# Roll-count support of the one-sided bear-off table.
MAX_ROLLS = 32
REFS = ("exact", "gnubg0", "gnubg2")


def survival_from_dist(dist: np.ndarray) -> np.ndarray:
    """Return P(more than r rolls needed) for r = 0..MAX_ROLLS, per state."""
    cdf = np.cumsum(dist, axis=1)
    surv = np.ones((dist.shape[0], dist.shape[1] + 1), dtype=np.float64)
    surv[:, 1:] = 1.0 - cdf
    np.clip(surv, 0.0, 1.0, out=surv)
    return surv


def pwin_onroll(m_idx: int, o_idx: int, dist: np.ndarray, surv: np.ndarray) -> float:
    """Return the probability that the side on roll (state m_idx) bears off first."""
    return float(dist[m_idx] @ surv[o_idx, :MAX_ROLLS])


def random_state(rng, max_on: int) -> tuple[int, ...]:
    """Return a random home-board state with 1 to max_on checkers still on."""
    n_checkers = int(rng.integers(1, max_on + 1))
    counts = [0] * 6
    for _ in range(n_checkers):
        counts[int(rng.integers(0, 6))] += 1
    return tuple(counts)


def home_counts_white(board) -> tuple[int, ...]:
    """Return White's checker counts on points 1 to 6."""
    return tuple(max(board.point(p), 0) for p in range(1, 7))


def load_table():
    """Return the state index, roll-count distributions and survival functions of the table."""
    db = np.load(DB_PATH)
    states = [tuple(int(c) for c in row) for row in db["states"]]
    index = {s: i for i, s in enumerate(states)}
    dist = db["dist"].astype(np.float64)
    return index, dist, survival_from_dist(dist)


def load_panel(panel):
    """Return {name: agent} for the panel entries of the settings."""
    from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
    from active_oracle_evaluation.eval.agent_specs import resolve_agent

    encoder = BinarySlotEncoder()
    agents = {}
    for entry in panel:
        spec = entry["spec"]
        if "checkpoint" in entry:
            spec = f"{spec}:{REPO / entry['checkpoint']}"
        agents[entry["name"]] = resolve_agent(spec, encoder)[1]
    return agents


def label_afterstates(afters, opp, table):
    """Return the exact, 0-ply and 2-ply equities of each afterstate for the mover."""
    import gnubg_nn as g

    from active_oracle_evaluation.eval._gnubg_board import to_gnubg_board
    from active_oracle_evaluation.eval.oracle import equity_from_probs

    index, dist, surv = table
    o_idx = index[opp]
    eq_exact, eq_g0, eq_g2 = [], [], []
    for after in afters:
        mover_after = home_counts_white(after)
        p_opp = pwin_onroll(o_idx, index[mover_after], dist, surv)
        eq_exact.append(1.0 - 2.0 * p_opp)
        if sum(mover_after) == 0:  # mover has borne off everything: equity is exactly +1
            eq_g0.append(1.0)
            eq_g2.append(1.0)
        else:
            gb = to_gnubg_board(after)  # afterstate: opponent to act
            eq_g0.append(-equity_from_probs(g.probabilities(gb, 0)))
            eq_g2.append(-equity_from_probs(g.probabilities(gb, 2)))
    return {"exact": eq_exact, "gnubg0": eq_g0, "gnubg2": eq_g2}


def sample_decisions(n, rng, table, cfg, t0):
    """Sample n unforced decisions on the slice; return them and the forced-move count."""
    from active_oracle_evaluation.backgammon.core.board import Board
    from active_oracle_evaluation.backgammon.core.moves import legal_moves
    from active_oracle_evaluation.backgammon.core.types import Player

    decisions = []
    n_forced = 0
    while len(decisions) < n:
        mover = random_state(rng, cfg["max_checkers_on"])
        opp = random_state(rng, cfg["max_checkers_on"])
        points = [0] * 24
        for i, c in enumerate(mover):
            points[i] = c
        for i, c in enumerate(opp):
            points[23 - i] = -c
        board = Board(tuple(points), 0, 0, 15 - sum(mover), 15 - sum(opp), Player.WHITE)
        roll = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
        moves = legal_moves(board, roll)
        if len(moves) <= 1:
            n_forced += 1
            continue
        afters = [mv.afterstate for mv in moves]
        eq = label_afterstates(afters, opp, table)
        decisions.append({"board": board, "dice": roll, "afters": afters, "eq": eq})
        if len(decisions) % cfg["progress_every"] == 0:
            print(
                f"  [label] {len(decisions)}/{n} decisions ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    return decisions, n_forced


def oracle_error(decisions, quantile):
    """Return the oracle equity error against the exact value, per afterstate."""
    err = {}
    for ref in ("gnubg0", "gnubg2"):
        diffs = np.array(
            [e - x for dec in decisions for e, x in zip(dec["eq"][ref], dec["eq"]["exact"])]
        )
        err[ref] = {
            "n": int(diffs.size),
            "mae": float(np.abs(diffs).mean()),
            "bias": float(diffs.mean()),
            "p95_abs": float(np.quantile(np.abs(diffs), quantile)),
            "max_abs": float(np.abs(diffs).max()),
        }
    return err


def best_action_agreement(decisions):
    """Return the oracle's best-action agreement with the exact reference (ties: lowest index)."""
    agree = {}
    for ref in ("gnubg0", "gnubg2"):
        same = [
            int(np.argmax(dec["eq"][ref]) == np.argmax(dec["eq"]["exact"])) for dec in decisions
        ]
        agree[ref] = float(np.mean(same))
    return agree


def agent_choices(decisions, agents, progress_every, seed):
    """Return each agent's chosen afterstate index per decision; choices ignore the reference."""
    from active_oracle_evaluation.backgammon.core.game import Game

    choices = {name: [] for name in agents}
    t1 = time.perf_counter()
    for k, dec in enumerate(decisions):
        for name, agent in agents.items():
            # Seed each decision's game, so the random agent's choices reproduce.
            game = Game(board=dec["board"], rng=random.Random(f"{seed}:{k}"))
            game.set_dice(dec["dice"])
            mv = agent.choose_move(game)
            choices[name].append(dec["afters"].index(mv.afterstate))
        if (k + 1) % progress_every == 0:
            print(
                f"  [choose] {k + 1}/{len(decisions)} ({time.perf_counter() - t1:.0f}s)",
                flush=True,
            )
    return choices


def losses_of(decisions, choices, name: str, ref: str) -> np.ndarray:
    """Return the agent's per-decision equity loss under one reference."""
    out = np.empty(len(decisions))
    for k, dec in enumerate(decisions):
        eq = dec["eq"][ref]
        out[k] = max(eq) - eq[choices[name][k]]
    return out


def paired_contrasts(names, losses, boot_idx, cfg):
    """Return {ref: {"a-b": [dPR, lo, hi]}} for every agent pair, with bootstrap intervals."""
    pr_scale = cfg["pr_scale"]
    contrasts = {ref: {} for ref in REFS}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for ref in REFS:
                delta = losses[ref][a] - losses[ref][b]
                point = float(delta.mean() * pr_scale)
                boot = delta[boot_idx].mean(axis=1) * pr_scale
                lo, hi = (float(np.quantile(boot, q)) for q in cfg["interval_quantiles"])
                contrasts[ref][f"{a}-{b}"] = [point, lo, hi]
    return contrasts


def gap_distortion(contrasts):
    """Return the exact-separated pairs, the 0-ply sign flips on them and the gap distortions."""
    sep_pairs = [k for k, (pt, lo, hi) in contrasts["exact"].items() if lo > 0 or hi < 0]
    sign_flips, distortions = [], {}
    for k in sep_pairs:
        ex = contrasts["exact"][k][0]
        g0 = contrasts["gnubg0"][k][0]
        if ex * g0 < 0:
            sign_flips.append(k)
        distortions[k] = [ex, g0, float(abs(g0 - ex) / abs(ex))]
    return sep_pairs, sign_flips, distortions


def score_panel(decisions, choices, names, seed, cfg):
    """Return the PR table, rankings, paired contrasts and 0-ply distortion of the panel."""
    losses = {
        ref: {name: losses_of(decisions, choices, name, ref) for name in names} for ref in REFS
    }
    pr = {
        ref: {name: float(losses[ref][name].mean() * cfg["pr_scale"]) for name in names}
        for ref in REFS
    }
    boot_rng = np.random.default_rng(seed + 1)
    n = len(decisions)
    boot_idx = boot_rng.integers(0, n, size=(cfg["n_bootstrap"], n))
    contrasts = paired_contrasts(names, losses, boot_idx, cfg)
    ranking = {ref: sorted(names, key=lambda nm: pr[ref][nm]) for ref in REFS}
    sep_pairs, sign_flips, distortions = gap_distortion(contrasts)
    return {
        "pr": pr,
        "ranking": ranking,
        "paired_contrasts_dPRx500": contrasts,
        "exact_separated_pairs": sep_pairs,
        "sign_flips_gnubg0_on_separated": sign_flips,
        "gap_distortion_gnubg0": distortions,
    }


def parse_args(cfg):
    """Parse the command line, with defaults from the settings."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=cfg["n_decisions"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--out", default=str(output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"))
    return ap.parse_args()


def main() -> int:
    """Label the slice, score the panel under each reference and write the results."""
    cfg = settings(EXPERIMENT)
    args = parse_args(cfg)

    table = load_table()
    rng = np.random.default_rng(args.seed)
    agents = load_panel(cfg["panel"])

    t0 = time.perf_counter()
    decisions, n_forced = sample_decisions(args.n, rng, table, cfg, t0)
    err = oracle_error(decisions, cfg["error_quantile"])
    agree = best_action_agreement(decisions)
    choices = agent_choices(decisions, agents, cfg["progress_every"], args.seed)
    scores = score_panel(decisions, choices, list(agents), args.seed, cfg)

    result = {
        "n_decisions": len(decisions),
        "n_forced_skipped": n_forced,
        "seed": args.seed,
        "instrument": "1-ply (k=1 greedy) agent choices; uniform-random in-domain positions",
        "oracle_equity_error_vs_exact": err,
        "best_action_agreement_vs_exact": agree,
        "pr": scores["pr"],
        "ranking": scores["ranking"],
        "paired_contrasts_dPRx500": scores["paired_contrasts_dPRx500"],
        "exact_separated_pairs": scores["exact_separated_pairs"],
        "sign_flips_gnubg0_on_separated": scores["sign_flips_gnubg0_on_separated"],
        "gap_distortion_gnubg0": scores["gap_distortion_gnubg0"],
        "wall_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    md_path = write_markdown(EXPERIMENT, result, out.with_suffix(".md"))
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("paired_contrasts_dPRx500", "gap_distortion_gnubg0")
    }
    print(json.dumps(summary, indent=1), flush=True)
    print(f"full result -> {args.out}", flush=True)
    print(f"tables -> {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
