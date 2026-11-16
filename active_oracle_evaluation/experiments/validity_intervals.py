"""95% bootstrap intervals for both columns of the validity table (oracle PR and anchored Elo).

The Elo column resamples game outcomes clustered within pairing from the stored win/game matrices
and refits Bradley-Terry per replicate, with ``pubeval`` pinned at 0. The Elo intervals are
therefore anchor-relative and correlated across agents; the experiment reports that correlation
and the anchor-free pairwise differences.

The oracle-PR column replays the seeded duplicate-dice round-robin, scores every decision with the
oracle, and resamples (pairing, duplicate-dice trial) clusters. The per-decision losses are saved
to an ``.npz`` file. Pairwise counts are reported marginally and under Holm and Benjamini-Hochberg.

Results go to outputs/validity_intervals/validity_intervals.{md,json}. With the paper's settings
(the [validity_intervals] table of config/experiments.toml):

    python -m active_oracle_evaluation.experiments.validity_intervals --elo-only
    python -m active_oracle_evaluation.experiments.validity_intervals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.validity_intervals_stats import (
    adjusted_counts,
    elo_intervals,
    pr_intervals,
)
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "validity_intervals"
VALIDITY_JSON = DATA / "round_robins" / "elo_validity.json"
LOSSES_NAME = "validity_pr_decision_losses.npz"


def _parse_args(cfg: dict) -> argparse.Namespace:
    """Parse the command line, with defaults from the experiment settings."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elo-only", action="store_true")
    ap.add_argument("--boot", type=int, default=cfg["boot"])
    ap.add_argument("--seed", type=int, default=cfg["bootstrap_seed"])
    ap.add_argument("--workers", type=int, default=cfg["workers"])
    ap.add_argument(
        "--pairs",
        type=int,
        default=cfg["pairs"],
        help="duplicate-dice trials per pairing in the PR replay",
    )
    ap.add_argument("--json", type=Path, help="also write the results dict here (sorted keys)")
    ap.add_argument(
        "--losses-npz",
        type=Path,
        default=None,
        help=f"where to save the per-decision losses (default outputs/{EXPERIMENT}/{LOSSES_NAME})",
    )
    return ap.parse_args()


def _print_elo(elo: dict) -> None:
    """Print the per-agent Elo intervals, their correlation and the pairwise differences."""
    print(f"\n{'agent':<12}{'Elo':>7}{'95% CI':>20}{'width':>8}")
    for n, v in elo["per_agent"].items():
        ci = f"[{v['lo']:+.0f}, {v['hi']:+.0f}]"
        print(f"{n:<12}{v['elo']:>+7.0f}{ci:>20}{v['width']:>8.0f}")
    c = elo["correlation"]
    print(
        f"\ncross-agent correlation among the {len(c['agents'])} unpinned agents: "
        f"{c['min_offdiag']:+.2f} to {c['max_offdiag']:+.2f}  "
        "(so these are NOT independent per-agent intervals)"
    )
    print("\npairwise differences (anchor-free):")
    for k, v in elo["pairwise_differences"].items():
        print(
            f"  {k:<22}{v['mean']:>+7.0f}  [{v['lo']:+.0f}, {v['hi']:+.0f}]  "
            f"{'resolved' if v['resolved'] else '-'}"
        )


def _print_pr(out: dict, pairs: int) -> None:
    """Print the PR intervals, the multiplicity counts and the resolved cluster pairs."""
    print(f"\n{'agent':<12}{'PR':>8}{'95% CI':>20}")
    for n, v in out["pr"]["per_agent"].items():
        ci = f"[{v['lo']:.1f}, {v['hi']:.1f}]"
        print(f"{n:<12}{v['pr_replay']:>8.1f}{ci:>20}")
    adj_pr = out["multiplicity"]["oracle_pr"]
    adj_elo = out["multiplicity"]["anchored_elo"]
    print(
        f"\nmultiplicity (alpha {out['multiplicity']['alpha']}): oracle-PR marginal "
        f"{adj_pr['marginal']} / Holm {adj_pr['holm']} / BH {adj_pr['bh']}   |   "
        f"Elo marginal {adj_elo['marginal']} / Holm {adj_elo['holm']} / BH {adj_elo['bh']}"
    )
    pw = out["pr"]["pairwise_differences"]
    elo_pw = out["elo"]["pairwise_differences"]
    n_pr = sum(v["resolved"] for v in pw.values())
    n_elo = sum(v["resolved"] for v in elo_pw.values())
    print(
        f"\ncluster pairs resolved:  oracle-PR {n_pr}/{len(pw)}   "
        f"anchored Elo {n_elo}/{len(elo_pw)}   "
        f"(same {pairs}-per-pairing budget)"
    )
    for k, v in pw.items():
        print(
            f"  {k:<22}{v['mean']:>+8.2f}  [{v['lo']:+.2f}, {v['hi']:+.2f}]  "
            f"{'resolved' if v['resolved'] else '-'}"
        )


def _pairwise_rows(pairwise: dict, mean_fmt: str, ci_fmt: str) -> list:
    """Format pairwise differences as table rows, with the printed precision."""
    return [
        [
            k,
            format(v["mean"], mean_fmt),
            f"[{format(v['lo'], ci_fmt)}, {format(v['hi'], ci_fmt)}]",
            "resolved" if v["resolved"] else "-",
        ]
        for k, v in pairwise.items()
    ]


def _elo_tables(elo: dict) -> list:
    """Build the Markdown tables of the anchored-Elo column."""
    per_agent = [
        [n, f"{v['elo']:+.0f}", f"[{v['lo']:+.0f}, {v['hi']:+.0f}]", f"{v['width']:.0f}"]
        for n, v in elo["per_agent"].items()
    ]
    c = elo["correlation"]
    corr = [[str(len(c["agents"])), f"{c['min_offdiag']:+.2f}", f"{c['max_offdiag']:+.2f}"]]
    diff_header = ["pair", "mean", "95% CI", "resolved"]
    return [
        ("Anchored Elo per agent", ["agent", "Elo", "95% CI", "width"], per_agent),
        (
            "Cross-agent correlation of the Elo replicates",
            ["unpinned agents", "min off-diagonal", "max off-diagonal"],
            corr,
        ),
        (
            "Anchored Elo pairwise differences",
            diff_header,
            _pairwise_rows(elo["pairwise_differences"], "+.0f", "+.0f"),
        ),
    ]


def _pr_tables(out: dict) -> list:
    """Build the Markdown tables of the oracle-PR column and the resolved-pair counts."""
    pr = out["pr"]
    per_agent = [
        [n, f"{v['pr_replay']:.1f}", f"[{v['lo']:.1f}, {v['hi']:.1f}]"]
        for n, v in pr["per_agent"].items()
    ]
    replay = [
        [
            str(pr["n_decisions_scored"]),
            str(pr["n_clusters"]),
            f"{pr['replay_max_abs_diff_vs_published']:.4f}",
        ]
    ]
    counts = []
    for label, key, pairwise in (
        ("oracle PR", "oracle_pr", pr["pairwise_differences"]),
        ("anchored Elo", "anchored_elo", out["elo"]["pairwise_differences"]),
    ):
        adj = out["multiplicity"][key]
        resolved = sum(v["resolved"] for v in pairwise.values())
        row = [label, f"{resolved}/{len(pairwise)}", adj["holm"], adj["bh"]]
        counts.append([str(cell) for cell in row])
    alpha = out["multiplicity"]["alpha"]
    return [
        ("Oracle PR per agent", ["agent", "PR", "95% CI"], per_agent),
        (
            "Oracle PR replay",
            ["decisions scored", "clusters", "max abs diff vs stored PR"],
            replay,
        ),
        (
            "Oracle PR pairwise differences",
            ["pair", "mean", "95% CI", "resolved"],
            _pairwise_rows(pr["pairwise_differences"], "+.2f", "+.2f"),
        ),
        (
            f"Resolved cluster pairs (alpha {alpha})",
            ["rating", "marginal", "Holm", "BH"],
            counts,
        ),
    ]


def main() -> None:
    """Parse arguments, compute the Elo and (optionally) PR intervals, print and save them."""
    cfg = settings(EXPERIMENT)
    args = _parse_args(cfg)
    percentiles = cfg["interval_percentiles"]

    out: dict = {}
    print("anchored Elo — clustered bootstrap within pairing, pubeval pinned ...")
    elo = elo_intervals(
        VALIDITY_JSON,
        tuple(cfg["cluster"]),
        args.boot,
        args.seed,
        percentiles,
        cfg["refit_tolerance"],
    )
    out["elo"] = elo
    _print_elo(elo)

    tables = _elo_tables(elo)
    if not args.elo_only:
        losses_out = args.losses_npz
        if losses_out is None:
            losses_out = output_dir(EXPERIMENT) / LOSSES_NAME
        run = {
            "agent_specs": tuple(cfg["agent_specs"]),
            "dice_seed": cfg["dice_seed"],
            "pairs": args.pairs,
            "plies": cfg["plies"],
            "cluster": tuple(cfg["cluster"]),
            "percentiles": percentiles,
        }
        print("\noracle-PR — replaying the round-robin (persisting per-decision losses) ...")
        out["pr"] = pr_intervals(VALIDITY_JSON, run, args.boot, args.seed, args.workers, losses_out)
        alpha = cfg["alpha"]
        adj_pr = adjusted_counts(out["pr"]["pairwise_differences"], alpha)
        adj_elo = adjusted_counts(elo["pairwise_differences"], alpha)
        out["multiplicity"] = {"oracle_pr": adj_pr, "anchored_elo": adj_elo, "alpha": alpha}
        _print_pr(out, args.pairs)
        tables += _pr_tables(out)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    md_path, json_path = write_results(EXPERIMENT, tables, out)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
