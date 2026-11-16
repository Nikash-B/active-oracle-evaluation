"""Cluster pairs resolved by oracle-PR and anchored Elo at an equal number of games per pairing.

Anchored Elo comes from a round-robin of 60 games per pairing. Oracle-PR comes from 60
duplicate-dice trials per pairing, each two games with seats swapped, so 120 games. This
experiment recounts oracle-PR on 30 of the 60 trials in each pairing (60 games), from the
per-decision losses that the validity_intervals experiment stores, with the bootstrap of its
`pr_intervals`: one draw over the kept trials shared by every agent, PR = mean loss x
`pr_per_equity`, and a pair resolves when the 95% interval of its difference excludes zero.

The full-data count must first reproduce the intervals and resolved set stored in
data/round_robins/validity_column_intervals.json; the experiment exits 1 otherwise. Halving the
trials keeps the duplicate-dice pairing inside each trial, which Elo's round-robin games do not
have. For comparison with the head-to-head games themselves, it also counts the cluster pairs
whose 60-game win rate in the shipped round-robin (data/round_robins/elo_validity.json) has a
Wilson 95% interval that excludes one half. The defaults in the [equal_games] table of
config/experiments.toml are the paper's settings:

    python -m active_oracle_evaluation.experiments.equal_games

It writes outputs/equal_games/equal_games.json and equal_games.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.validity_intervals_stats import adjusted_counts, boot_p
from active_oracle_evaluation.paths import DATA, REPO, output_dir, repo_relative
from active_oracle_evaluation.results import markdown_table

EXPERIMENT = "equal_games"
LOSSES = DATA / "round_robins" / "validity_pr_decision_losses.npz"
INTERVALS = DATA / "round_robins" / "validity_column_intervals.json"
ROUND_ROBIN = DATA / "round_robins" / "elo_validity.json"
COUNT_KEYS = ("marginal", "holm", "bh")
INTERVALS_COMMAND = "python -m active_oracle_evaluation.experiments.validity_intervals"


def load_interval_tools():
    """Return the validity-interval experiment module."""
    from active_oracle_evaluation.experiments import validity_intervals

    return validity_intervals


def adjusted_at_alpha(pairwise):
    """Multiplicity-adjusted counts at the validity_intervals alpha."""
    return adjusted_counts(pairwise, settings("validity_intervals")["alpha"])


def pairwise_intervals(keep, trials, cluster, n_boot, seed, percentiles, vci):
    """Bootstrap PR over the kept trials and return the interval of every cluster pair."""
    loss_sum, decision_count, names, pr_per_equity = trials
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    reps = np.full((len(names), n_boot), np.nan)
    kept_sum, kept_count = loss_sum[:, idx], decision_count[:, idx]
    for rep in range(n_boot):
        draw = rng.integers(0, len(idx), len(idx))
        s, c = kept_sum[:, draw].sum(1), kept_count[:, draw].sum(1)
        reps[:, rep] = np.where(c > 0, s / np.maximum(c, 1) * pr_per_equity, np.nan)
    out = {}
    for a, b in itertools.combinations([x for x in cluster if x in names], 2):
        diff = reps[names.index(a)] - reps[names.index(b)]
        lo, hi = np.nanpercentile(diff, percentiles)
        out[f"{a} - {b}"] = {
            "mean": float(np.nanmean(diff)),
            "lo": float(lo),
            "hi": float(hi),
            "resolved": bool(lo > 0 or hi < 0),
            "p": boot_p(diff),
        }
    return out


def wilson_interval(wins, games, z):
    """Wilson score interval of a binomial proportion of ``wins`` in ``games``."""
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return centre - half, centre + half


def wilson_pairs(round_robin, cluster, z):
    """Head-to-head win rate of each cluster pair, its Wilson interval and if it excludes 0.5."""
    names = round_robin["names"]
    out = {}
    for a, b in itertools.combinations([x for x in cluster if x in names], 2):
        i, j = names.index(a), names.index(b)
        wins, games = int(round_robin["wins"][i][j]), int(round_robin["games"][i][j])
        lo, hi = wilson_interval(wins, games, z)
        out[f"{a} - {b}"] = {
            "wins": wins,
            "games": games,
            "lo": lo,
            "hi": hi,
            "resolved": bool(lo > 0.5 or hi < 0.5),
        }
    return out


def summarize(halves, key):
    """Sorted values, median, min and max of one count over the random half-samples."""
    values = sorted(h[key] for h in halves)
    return {
        "values": values,
        "median": float(np.median(values)),
        "min": values[0],
        "max": values[-1],
    }


def parse_args(cfg):
    """Parse the command line, with defaults from the [equal_games] settings."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --boot and --seed must match the validity_intervals run, or the full-data check fails.
    ap.add_argument("--boot", type=int, default=cfg["n_boot"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--halves", type=int, default=cfg["halves"])
    ap.add_argument("--halves-seed", type=int, default=cfg["halves_seed"])
    ap.add_argument(
        "--intervals-json",
        type=Path,
        default=INTERVALS,
        help="JSON output of the validity_intervals experiment (full run, with multiplicity)",
    )
    ap.add_argument(
        "--json",
        type=Path,
        help="JSON summary path (default outputs/equal_games/equal_games.json); "
        "the Markdown report goes next to it with suffix .md",
    )
    return ap.parse_args()


def load_trials():
    """Per-agent loss sums and decision counts per trial, with the trial labels."""
    data = np.load(LOSSES, allow_pickle=True)
    names = [str(n) for n in data["agent_names"]]
    labels = [str(c) for c in data["cluster_labels"]]
    n_trials, n_agents = len(labels), len(names)
    loss_sum = np.zeros((n_agents, n_trials))
    decision_count = np.zeros((n_agents, n_trials))
    np.add.at(loss_sum, (data["agent"], data["cluster"]), data["loss"].astype(np.float64))
    np.add.at(decision_count, (data["agent"], data["cluster"]), 1.0)
    pr_per_equity = float(data["pr_per_equity"])
    trial = np.array([int(lab.split("#")[1]) for lab in labels])
    pairing = np.array([lab.split("#")[0] for lab in labels])
    return (loss_sum, decision_count, names, pr_per_equity), trial, pairing


def random_halves(count, pairing, half, n_halves, halves_seed, vci):
    """Counts of resolved pairs on random half-samples of the trials in every pairing."""
    rng = np.random.default_rng(halves_seed)
    halves = []
    for _ in range(n_halves):
        keep = np.zeros(len(pairing), bool)
        for p in np.unique(pairing):
            in_pairing = np.flatnonzero(pairing == p)
            keep[rng.choice(in_pairing, half, replace=False)] = True
        adj = adjusted_at_alpha(count(keep))
        halves.append({k: adj[k] for k in COUNT_KEYS})
    return halves


def build_summary(args, cfg, per_pairing, control, first, halves, elo_counts, wilson):
    """Assemble the JSON summary of the control, the equal-games recounts and Elo."""
    half = per_pairing // 2
    worst, same, full = control
    elo_games = cfg["elo_games_per_pairing"]
    return {
        "tool": f"python -m active_oracle_evaluation.experiments.{EXPERIMENT}",
        "question": "oracle-PR vs anchored Elo pair counts at equal games per pairing",
        "games_per_pairing": {
            "elo": elo_games,
            "pr_full_data": 2 * per_pairing,
            "pr_this_recount": 2 * half,
        },
        "why": (
            f"The elo_validity experiment plays Elo as {elo_games} round-robin games per "
            f"pairing and PR as {per_pairing} duplicate-dice trials ({2 * per_pairing} games) "
            "per pairing."
        ),
        "bootstrap": {
            "n_boot": args.boot,
            "seed": args.seed,
            "method": "validity_intervals.pr_intervals",
        },
        "control_full_data": {
            "max_abs_diff_vs_stored": worst,
            "same_resolved_set": same,
            **{k: adjusted_at_alpha(full)[k] for k in COUNT_KEYS},
        },
        "equal_games_first_half": {
            **{k: adjusted_at_alpha(first)[k] for k in COUNT_KEYS},
            "unresolved": {k: v for k, v in first.items() if not v["resolved"]},
        },
        "equal_games_random_halves": {
            "n": args.halves,
            "seed": args.halves_seed,
            **{k: summarize(halves, k) for k in COUNT_KEYS},
        },
        "elo_at_60_games": {
            **{k: elo_counts[k] for k in COUNT_KEYS},
            "source": INTERVALS_COMMAND,
        },
        "wilson_head_to_head": {
            "z": cfg["wilson_z"],
            "games_per_pairing": min(v["games"] for v in wilson.values()),
            "resolved": sum(v["resolved"] for v in wilson.values()),
            "pairs": wilson,
            "source": repo_relative(ROUND_ROBIN),
        },
        "not_equalised": (
            "halving trials keeps duplicate-dice pairing within each trial; Elo's games "
            "are not duplicate-dice. That reduction is part of the PR instrument, so equal "
            "games is the fair budget comparison but not an identical design."
        ),
    }


def report_tables(out, n_pairs):
    """Result tables of the summary, with numbers formatted as the printed report."""
    games = out["games_per_pairing"]
    ctrl, fh = out["control_full_data"], out["equal_games_first_half"]
    rh, elo = out["equal_games_random_halves"], out["elo_at_60_games"]
    wilson = out["wilson_head_to_head"]

    def spread(key):
        """Median and range of one count over the random halves."""
        return f"{rh[key]['median']:.0f} [{rh[key]['min']}-{rh[key]['max']}]"

    rows = [
        ["PR, full data", games["pr_full_data"], *(ctrl[k] for k in COUNT_KEYS)],
        ["PR, first half", games["pr_this_recount"], *(fh[k] for k in COUNT_KEYS)],
        [
            f"PR, {rh['n']} random halves, median [min-max]",
            games["pr_this_recount"],
            *(spread(k) for k in COUNT_KEYS),
        ],
        ["Anchored Elo", games["elo"], *(elo[k] for k in COUNT_KEYS)],
        [
            "Head-to-head win rate, Wilson 95%",
            wilson["games_per_pairing"],
            wilson["resolved"],
            "–",
            "–",
        ],
    ]
    wilson_rows = [
        [pair, f"{v['wins']}/{v['games']}", f"[{v['lo']:.3f}, {v['hi']:.3f}]", v["resolved"]]
        for pair, v in wilson["pairs"].items()
    ]
    control_row = [f"{ctrl['max_abs_diff_vs_stored']:.1e}", ctrl["same_resolved_set"], n_pairs]
    return [
        (
            f"Resolved pairs of the {n_pairs} cluster pairs",
            ["Rating", "Games per pairing", "Marginal", "Holm", "BH"],
            rows,
        ),
        (
            "Full-data check against the stored intervals",
            ["Max abs difference", "Same resolved set", "Pairs"],
            [control_row],
        ),
        (
            "Head-to-head win rate of each cluster pair, Wilson 95% interval",
            ["Pair", "Wins/games", "Interval", "Excludes 0.5"],
            wilson_rows,
        ),
    ]


def write_report(out, n_pairs, json_path):
    """Write the JSON summary and the Markdown report next to it."""
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    parts = [f"# {EXPERIMENT}"]
    for title, header, rows in report_tables(out, n_pairs):
        parts += [f"## {title}", markdown_table(header, rows)]
    md_path = json_path.with_suffix(".md")
    md_path.write_text("\n\n".join(parts) + "\n")
    return json_path, md_path


def print_report(out, n_pairs, half, elo_games):
    """Print the control and the resolved-pair counts."""
    fh, rh = out["equal_games_first_half"], out["equal_games_random_halves"]
    elo_counts = out["elo_at_60_games"]
    worst = out["control_full_data"]["max_abs_diff_vs_stored"]
    print(
        f"control: full data reproduces the stored intervals to {worst:.1e}, resolved "
        f"{out['control_full_data']['marginal']}/{n_pairs}"
    )
    print(
        f"PR at {2 * half} games/pairing, first half:  marginal {fh['marginal']}  "
        f"holm {fh['holm']}  bh {fh['bh']}"
    )
    print(
        f"PR at {2 * half} games/pairing, {rh['n']} random halves: marginal median "
        f"{rh['marginal']['median']:.0f} [{rh['marginal']['min']}-{rh['marginal']['max']}], "
        f"holm median {rh['holm']['median']:.0f} [{rh['holm']['min']}-{rh['holm']['max']}]"
    )
    print(
        f"Elo at {elo_games} games/pairing: marginal {elo_counts['marginal']}  "
        f"holm {elo_counts['holm']}  bh {elo_counts['bh']}"
    )
    wilson = out["wilson_head_to_head"]
    print(
        f"head-to-head at {wilson['games_per_pairing']} games/pairing, Wilson 95% excludes 0.5: "
        f"{wilson['resolved']}/{n_pairs}"
    )


def show_path(path):
    """A path relative to the repo when it lies inside it, else as given."""
    written = path.resolve()
    return written.relative_to(REPO) if written.is_relative_to(REPO) else path


def main() -> int:
    """Run the full-data control, then recount PR on the first half and on random halves."""
    cfg = settings(EXPERIMENT)
    args = parse_args(cfg)
    percentiles = list(cfg["interval_percentiles"])

    vci = load_interval_tools()
    cluster = list(settings(vci.EXPERIMENT)["cluster"])
    trials, trial, pairing = load_trials()
    n_trials = len(trial)
    per_pairing = int(np.bincount(np.unique(pairing, return_inverse=True)[1]).min())

    def count(keep):
        """Pairwise intervals for one subset of trials."""
        return pairwise_intervals(keep, trials, cluster, args.boot, args.seed, percentiles, vci)

    intervals = json.loads(args.intervals_json.read_text())
    stored = intervals["pr"]["pairwise_differences"]
    elo_counts = intervals["multiplicity"]["anchored_elo"]
    full = count(np.ones(n_trials, bool))
    worst = max(max(abs(full[k][f] - stored[k][f]) for f in ("mean", "lo", "hi")) for k in stored)
    same = {k for k, v in full.items() if v["resolved"]} == {
        k for k, v in stored.items() if v["resolved"]
    }
    if worst > cfg["control_tolerance"] or not same:
        print(f"control failed: max |diff| {worst:.2e}, same resolved set {same}")
        return 1

    half = per_pairing // 2
    first = count(trial < half)
    halves = random_halves(count, pairing, half, args.halves, args.halves_seed, vci)
    control = (worst, same, full)
    wilson = wilson_pairs(json.loads(ROUND_ROBIN.read_text()), cluster, cfg["wilson_z"])
    out = build_summary(args, cfg, per_pairing, control, first, halves, elo_counts, wilson)
    json_path = args.json if args.json else output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    paths = write_report(out, len(full), json_path)

    print_report(out, len(full), half, cfg["elo_games_per_pairing"])
    for path in paths:
        print(f"wrote {show_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
