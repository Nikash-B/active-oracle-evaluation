"""Select budget-K evaluation subsets from the frozen 0-ply oracle decision suite.

For each selection rule (``random``, ``phase_balanced``, ``disagreement``, ``active_rank``) and
each budget ``K``, the experiment draws one subset of the suite by Madow systematic PPS with
``active_oracle_evaluation.eval.eval_select.select`` and writes two files:

* ``{out_dir}/{strategy}_K{K}_seed{seed}.jsonl``: the selected decision records, in selection
  order.
* ``{out_dir}/{strategy}_K{K}_seed{seed}.meta.json``: the strategy, ``K``, seed, source suite and
  its size, position ids, inclusion probabilities and a one-line design note. The inclusion
  probabilities feed the Horvitz-Thompson PR estimator.

It also writes ``select_positions.md`` and ``select_positions.json`` with the phase mix of every
subset and the phase mix by rule at one budget. Per-agent losses, choices and phases come from
the precomputed score cache, aligned to the eligible-records order; no agent is re-scored.
Settings live in the ``[select_positions]`` table of ``config/experiments.toml``. The paper's
settings:

    python -m active_oracle_evaluation.experiments.select_positions \\
        --suite data/decision_suites/backgammon_oracle_suite_0ply_v1.jsonl \\
        --cache data/decision_suites/full_suite_scores_0ply.npz \\
        --seed 20260628 --ks 50 100 250 500 1000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.decision_scoring import eligible_records
from active_oracle_evaluation.eval.decision_suite import read_jsonl, write_jsonl
from active_oracle_evaluation.eval.eval_select import STRATEGIES, select
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "select_positions"
DEFAULT_SUITE = DATA / "decision_suites" / "backgammon_oracle_suite_0ply_v1.jsonl"
DEFAULT_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"

NOTES = {
    "random": "uniform without replacement (Madow PPS, constant weights); inclusion = K/N",
    "phase_balanced": "phase-stratified water-fill, uniform within phase; inclusion = K_s/N_s",
    "disagreement": "PPS by panel chosen-action Simpson diversity + uniform eps-floor",
    "active_rank": (
        "phase-balanced; within phase PPS by summed pairwise |regret_i-regret_j| + eps-floor"
    ),
}


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=DEFAULT_SUITE, help="full DecisionRecord JSONL"
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE, help="precomputed score cache (.npz)"
    )
    parser.add_argument("--seed", type=int, default=int(config["seed"]))
    parser.add_argument("--ks", nargs="+", type=int, default=[int(k) for k in config["ks"]])
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="output directory (default outputs/<name>)"
    )
    return parser


def load_cache(cache_path: Path, records: list) -> dict:
    """Load the score cache and check that it aligns with the eligible records."""
    cache = np.load(cache_path, allow_pickle=True)
    cache_ids = [str(x) for x in cache["position_ids"]]
    rec_ids = [r.position_id for r in records]
    if cache_ids != rec_ids:
        raise SystemExit(
            f"cache/suite misalignment: {len(cache_ids)} cache ids vs "
            f"{len(rec_ids)} eligible records"
        )
    return {
        "names": [str(x) for x in cache["names"]],
        "phases": [str(x) for x in cache["phases"]],
        "choices": cache["choices"],
        "losses": cache["losses"],
    }


def write_subset(args, strategy: str, k: int, records: list, selection) -> list:
    """Check one selection, write its JSONL and meta files, and return its indices."""
    n_pos = len(records)
    indices, incl = selection.indices, selection.inclusion_probs
    expected = min(k, n_pos)
    assert len(indices) == expected, f"{strategy} K={k}: {len(indices)} != {expected}"
    assert len(set(indices)) == len(indices), f"{strategy} K={k}: duplicate indices"
    assert all(0 <= i < n_pos for i in indices) and len(incl) == len(indices)

    sel_records = [records[i] for i in indices]
    stem = f"{strategy}_K{k}_seed{args.seed}"
    write_jsonl(sel_records, args.out_dir / f"{stem}.jsonl")
    meta = {
        "strategy": strategy,
        "K": k,
        "seed": args.seed,
        "source_suite": str(args.suite.name),
        "source_suite_size": n_pos,
        "position_ids": [r.position_id for r in sel_records],
        "inclusion_probs": [float(x) for x in incl],
        "notes": NOTES[strategy],
    }
    (args.out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
    return indices


def phase_mix(selected: dict, phases: list, k_show: int, uniq: list) -> list:
    """Return (row label, per-phase percent list) rows for the pool and each rule at k_show."""
    n_pos = len(phases)
    rows = [("pool%", [100 * phases.count(p) / n_pos for p in uniq])]
    for strategy in STRATEGIES:
        indices = selected[strategy][k_show]
        counts = Counter(phases[i] for i in indices)
        rows.append((strategy, [100 * counts.get(p, 0) / len(indices) for p in uniq]))
    return rows


def print_phase_mix(mix_rows: list, k_show: int, uniq: list) -> None:
    """Print the phase-distribution comparison of the rules against the pool."""
    print(f"\nPhase distribution by strategy @K={k_show} (vs pool):")
    print("  " + "strategy".ljust(15) + "".join(p[:8].rjust(9) for p in uniq))
    for label, percents in mix_rows:
        print("  " + label.ljust(15) + "".join(f"{x:8.1f}%" for x in percents))


def save_report(args, cache: dict, set_rows: list, mix_rows: list, k_show: int, uniq: list):
    """Write select_positions.md and select_positions.json; return their paths."""
    phases = cache["phases"]
    tables = [
        (
            "Phase counts of each selected subset",
            ["strategy", "K", "file"] + uniq,
            [
                [row["strategy"], str(row["K"]), row["file"]]
                + [str(row["phases"].get(p, 0)) for p in uniq]
                for row in set_rows
            ],
        ),
        (
            f"Phase distribution by strategy at K={k_show} (percent)",
            ["strategy"] + uniq,
            [[label] + [f"{x:.1f}" for x in percents] for label, percents in mix_rows],
        ),
    ]
    numbers = {
        "seed": args.seed,
        "suite_size": len(phases),
        "agents": cache["names"],
        "pool_phases": dict(Counter(phases)),
        "sets": set_rows,
        "phase_mix_k": k_show,
        "phase_mix_percent": {
            label: {p: float(f"{x:.1f}") for p, x in zip(uniq, percents)}
            for label, percents in mix_rows
        },
    }
    return write_results(EXPERIMENT, tables, numbers, out_dir=args.out_dir)


def main() -> None:
    """Select and write the evaluation subsets, then print and save their phase mix."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    if args.out_dir is None:
        args.out_dir = output_dir(EXPERIMENT)
    records = eligible_records(read_jsonl(args.suite))
    n_pos = len(records)
    cache = load_cache(args.cache, records)
    phases = cache["phases"]
    print(f"suite: {n_pos} eligible records; agents: {cache['names']}; seed {args.seed}")
    print(f"pool phase distribution: {dict(Counter(phases))}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict[int, list[int]]] = {s: {} for s in STRATEGIES}
    set_rows = []
    for strategy in STRATEGIES:
        for k in args.ks:
            selection = select(
                strategy,
                k,
                losses=cache["losses"],
                choices=cache["choices"],
                phases=phases,
                seed=args.seed,
                importance=config["importance"],
            )
            indices = write_subset(args, strategy, k, records, selection)
            selected[strategy][k] = indices
            stem = f"{strategy}_K{k}_seed{args.seed}"
            sel_phases = dict(Counter(phases[i] for i in indices))
            print(f"  {strategy:15s} K={k:5d}  -> {stem}.jsonl  phases={sel_phases}")
            set_rows.append(
                {"strategy": strategy, "K": k, "file": f"{stem}.jsonl", "phases": sel_phases}
            )

    phase_mix_k = int(config["phase_mix_k"])
    k_show = phase_mix_k if phase_mix_k in args.ks else args.ks[len(args.ks) // 2]
    uniq = sorted(set(phases))
    mix_rows = phase_mix(selected, phases, k_show, uniq)
    print_phase_mix(mix_rows, k_show, uniq)

    print(f"\nwrote {len(STRATEGIES) * len(args.ks)} selection sets (+ meta) to {args.out_dir}")
    md_path, json_path = save_report(args, cache, set_rows, mix_rows, k_show, uniq)
    print(f"wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
