"""Score an agent panel once on a decision suite and cache the per-agent regrets.

Every agent is scored once on every eligible (checker, oracle-labelled, unforced) decision, and
each selection rule reads a subset of this one score matrix, so selection, efficiency and
tail-risk analyses never re-score an agent. The output is an ``.npz`` with the per-agent regret
and chosen-action matrices aligned to the eligible records, a JSON sidecar beside it with names,
PRs and metadata, and a summary (score_panel_cache.md and score_panel_cache.json) in
outputs/score_panel_cache/.

The eight-agent panel cache and the 18-agent pool cache of the paper, whose shipped copies are
data/decision_suites/full_suite_scores_0ply.npz and full_suite_scores_0ply_large.npz:

    python -m active_oracle_evaluation.experiments.score_panel_cache
    python -m active_oracle_evaluation.experiments.score_panel_cache --pool large

They write outputs/score_panel_cache/full_suite_scores_0ply.npz and
full_suite_scores_0ply_large.npz.

With --plies, the eligible records are first re-labelled by the oracle at that depth, so the
agents are scored against the new labels on the same decisions. The 2-ply pool cache, shipped as
data/decision_suites/full_suite_scores_2ply_large.npz:

    python -m active_oracle_evaluation.experiments.score_panel_cache --pool large --plies 2 \
        --workers 4

The pool cache on the panel suite, shipped as
data/decision_suites/panel_suite_scores_0ply_large.npz:

    python -m active_oracle_evaluation.experiments.score_panel_cache --pool large \
        --suite data/decision_suites/panel_suite_0ply_v1.jsonl \
        --out outputs/score_panel_cache/panel_suite_scores_0ply_large.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import decision_scoring as ds
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.decision_suite import afterstates_of, read_jsonl
from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY
from active_oracle_evaluation.paths import DATA, output_dir, repo_relative
from active_oracle_evaluation.results import add_run

EXPERIMENT = "score_panel_cache"
DEFAULT_SUITE = DATA / "decision_suites" / "backgammon_oracle_suite_0ply_v1.jsonl"
POOL_SUFFIXES = {"panel": "", "large": "_large"}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="decision-suite JSONL")
    p.add_argument(
        "--pool",
        choices=["panel", "large"],
        default="panel",
        help="agent list from the settings: the eight-agent panel or the 18-agent pool",
    )
    p.add_argument(
        "--agents", nargs="+", default=None, metavar="SPEC", help="agent specs; overrides --pool"
    )
    p.add_argument(
        "--plies",
        type=int,
        default=None,
        help="re-label the eligible records with the oracle at this depth before scoring "
        "(default: keep the suite's own labels)",
    )
    p.add_argument("--workers", type=int, default=1, help="oracle processes for --plies")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output .npz, sidecar .json beside it (default outputs/{EXPERIMENT}/"
        "full_suite_scores_<plies>ply[_large].npz, with 0 ply when --plies is not given)",
    )
    return p


def _label_chunk(packed):
    """Return the oracle's mover equity of every legal action of each record in one chunk."""
    from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig  # needs gnubg-nn

    records, plies = packed
    oracle = GnubgOracle(OracleConfig(plies=plies))
    # The afterstate is the opponent's turn, so the mover's equity is its negation, as in
    # build_position_suite.
    return [[-oracle.equity(b) for b in afterstates_of(r)] for r in records]


def relabel(records, plies, workers):
    """Return copies of the records labelled by the oracle at ``plies``, in input order."""
    chunks = [records[k::workers] for k in range(workers)]  # round-robin keeps chunks balanced
    with mp.get_context("spawn").Pool(workers) as pool:
        labels = pool.map(_label_chunk, [(chunk, plies) for chunk in chunks])
    out = [None] * len(records)
    for k, chunk_labels in enumerate(labels):
        for j, equities in enumerate(chunk_labels):
            i = k + j * workers
            best = max(range(len(equities)), key=equities.__getitem__)
            out[i] = replace(
                records[i], oracle_equities=equities, best_action=best, best_equity=equities[best]
            )
    return out


def cache_arrays(names, losses, choices, elig):
    """Return the arrays of the score cache, keyed by their name in the .npz."""
    # Source game of each decision. The game counter restarts at 0 in each per-agent pool, so a
    # game is the pair (agent_source, game_id); ScoreCache builds the cluster key from both.
    # -1 and "" mark a record with no recorded provenance.
    return {
        "names": np.array(names),
        "position_ids": np.array([r.position_id for r in elig]),
        "phases": np.array([r.phase for r in elig]),
        "game_ids": np.array([int(r.metadata.get("game_id", -1)) for r in elig], dtype=np.int64),
        "agent_sources": np.array([str(r.agent_source or "") for r in elig]),
        "seeds": np.array([int(r.metadata.get("seed", -1)) for r in elig], dtype=np.int64),
        "losses": np.array([losses[n] for n in names], dtype=np.float64),  # (n_agents, n_pos)
        "choices": np.array([choices[n] for n in names], dtype=np.int32),
        "best_equity": np.array([r.best_equity for r in elig], dtype=np.float64),
        "n_actions": np.array([len(r.legal_actions) for r in elig], dtype=np.int32),
    }


def build_sidecar(suite, plies, names, elig, arrays, pr_decimals, seconds):
    """Return the sidecar dict and the per-agent PR of a scored cache."""
    loss_arr = arrays["losses"]
    agent_sources = arrays["agent_sources"]
    clusters = {(a, int(g)) for a, g in zip(agent_sources, arrays["game_ids"]) if g >= 0}
    pr = {
        n: round(float(loss_arr[i].mean() * PR_PER_EQUITY), pr_decimals)
        for i, n in enumerate(names)
    }
    sidecar = {
        "suite": repo_relative(suite),
        "oracle_plies": plies,  # None: the suite's own labels
        "agents": names,
        "n_eligible": len(elig),
        "pr": pr,
        "pr_order": sorted(names, key=lambda n: pr[n]),
        "phases": sorted({r.phase for r in elig}),
        "n_games": len(clusters),  # distinct (agent_source, game_id) pairs, not raw game ids
        "agent_sources": sorted({str(a) for a in agent_sources}),
        "seeds": sorted({int(x) for x in arrays["seeds"] if x >= 0}),
        "score_seconds": seconds,
    }
    return sidecar, pr


def summary_rows(numbers):
    """Return one table row per agent of one cache, best PR first."""
    return [
        [numbers["cache"], numbers["suite"], numbers["n_eligible"], name, value]
        for name, value in numbers["pr"].items()
    ]


def write_summary(cache_name, suite_name, n_records, n_eligible, pr):
    """Add this cache's PRs to score_panel_cache.md and score_panel_cache.json."""
    numbers = {
        "cache": cache_name,
        "suite": suite_name,
        "n_records": n_records,
        "n_eligible": n_eligible,
        "pr": dict(sorted(pr.items(), key=lambda kv: kv[1])),
    }
    header = ["cache", "suite", "eligible", "agent", "PR (lower is better)"]
    return add_run(EXPERIMENT, cache_name, numbers, summary_rows, "PR by cache", header)


def main() -> None:
    """Score every agent on the eligible suite records and write the cache and its sidecar."""
    args = build_parser().parse_args()
    config = settings(EXPERIMENT)
    specs = args.agents if args.agents is not None else config[f"{args.pool}_agents"]
    plies_name = 0 if args.plies is None else args.plies
    default_name = f"full_suite_scores_{plies_name}ply{POOL_SUFFIXES[args.pool]}.npz"
    out = args.out if args.out is not None else output_dir(EXPERIMENT) / default_name
    encoder = BinarySlotEncoder()
    agents = dict(resolve_agent(s, encoder) for s in specs)
    names = list(agents)

    recs = read_jsonl(args.suite)
    elig = ds.eligible_records(recs)
    print(
        f"suite {args.suite.name}: {len(recs)} records, "
        f"{len(elig)} eligible (checker/labeled/unforced)"
    )

    if args.plies is not None:
        t0 = time.perf_counter()
        elig = relabel(elig, args.plies, args.workers)
        print(f"re-labelled at {args.plies} ply in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    # One pass per record, so each loss and chosen action describe the same move.
    losses, choices = ds.score_matrices(agents, elig, seed=config["tie_break_seed"])
    dt = time.perf_counter() - t0

    arrays = cache_arrays(names, losses, choices, elig)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    sidecar, pr = build_sidecar(
        args.suite, args.plies, names, elig, arrays, config["pr_decimals"], round(dt, 1)
    )
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    print(f"PR (lower=better): {dict(sorted(pr.items(), key=lambda kv: kv[1]))}")
    print(f"wrote {out} (+ .json) in {dt:.1f}s")
    md_path, json_path = write_summary(out.name, args.suite.name, len(recs), len(elig), pr)
    print(f"wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
