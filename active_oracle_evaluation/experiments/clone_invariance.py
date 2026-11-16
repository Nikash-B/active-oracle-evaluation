"""Clone invariance of the Performance Rating, of head-to-head Elo and of the selection designs.

A redundant entrant should not move the rating of the other agents. This experiment checks three
things without playing new games:

1. PR. An agent's PR is its own mean regret on a fixed suite, so it has no panel term. We compute
   the base agents' PR on the 8-agent and 18-agent panels and report the largest shift.
2. Elo. Bradley-Terry is fit jointly over the panel. We inject an exact duplicate into the measured
   round-robin: the clone inherits the original's record against every other agent and splits evenly
   against the original. We refit and report the shift of the other agents.
3. Selection. ``disagreement`` reads every agent's chosen move and ``active_rank`` every agent's
   regret, so clones can change which decisions are labeled. We select with and without the clone
   families, score the same base panel both ways and report the paired difference in Kendall tau.

The results go to outputs/clone_invariance/clone_invariance.md and clone_invariance.json. With the
paper's settings (the [clone_invariance] table of config/experiments.toml):

    python -m active_oracle_evaluation.experiments.clone_invariance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import eval_select as sel
from active_oracle_evaluation.eval import rank_recovery as rr
from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo
from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY, ScoreCache
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import markdown_table

EXPERIMENT = "clone_invariance"
LARGE = DATA / "decision_suites" / "full_suite_scores_0ply_large.npz"
ROUND_ROBIN_JSON = DATA / "round_robins" / "elo_validity.json"

#: Elo shifts are reported with this agent pinned; the fallback is pinned when it is duplicated.
ANCHOR = settings(EXPERIMENT)["anchor"]
ANCHOR_FALLBACK = settings(EXPERIMENT)["anchor_fallback"]


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the command-line parser, with defaults from the settings."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, default=LARGE)
    p.add_argument(
        "--round-robin",
        type=Path,
        default=ROUND_ROBIN_JSON,
        help="8-agent round-robin, for the Elo comparison",
    )
    p.add_argument("--ks", type=int, nargs="+", default=list(config["ks"]))
    p.add_argument("--seeds", type=int, default=config["seeds"])
    p.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Markdown report path; the JSON summary is written next to it "
        f"(default: outputs/{EXPERIMENT}/{EXPERIMENT}.md)",
    )
    return p


# --- 1 + 2: rating clone-invariance ------------------------------------------ #


def pr_invariance(cache: ScoreCache, base: list[str]) -> dict:
    """Compute the PR of each base agent on the base panel and on the full 18-agent panel."""
    base_pr = {a: float(cache.losses[a].mean() * PR_PER_EQUITY) for a in base}
    # PR has no panel term, so the wide-panel value uses the same per-agent expression.
    wide_pr = {a: float(cache.losses[a].mean() * PR_PER_EQUITY) for a in base}
    return {
        "base": base_pr,
        "wide": wide_pr,
        "max_abs_shift": max(abs(base_pr[a] - wide_pr[a]) for a in base),
    }


def _fit_elo(names, wins, games) -> dict:
    """Fit Bradley-Terry Elo to a wins/games matrix."""
    return bradley_terry_elo(RoundRobin(names=list(names), games=games, wins=wins))


def elo_with_clone(round_robin: dict, clone_of: str) -> dict:
    """Refit Bradley-Terry Elo after injecting an exact duplicate of ``clone_of``.

    The clone inherits the original's record against every other agent and splits an equal-sized
    match evenly against the original.

    Elo has a free additive constant, which the fit fixes by centring the panel at zero, so the
    clone shifts every centred rating by a similar amount. ``shifts`` and ``max_abs_shift`` are
    therefore over the agents other than the duplicated one, with ``ANCHOR`` pinned
    (``ANCHOR_FALLBACK`` when ``ANCHOR`` is duplicated). ``spread_others`` is the largest change in
    any Elo difference between two other agents, which no anchor can move.
    ``max_abs_shift_centred_all`` is the largest centred shift over all agents.
    """
    names = list(round_robin["names"])
    wins = np.asarray(round_robin["wins"], dtype=float)
    games = np.asarray(round_robin["games"], dtype=float)
    i = names.index(clone_of)
    n = len(names)

    new_names = [*names, f"{clone_of}_clone"]
    w = np.zeros((n + 1, n + 1))
    g = np.zeros((n + 1, n + 1))
    w[:n, :n], g[:n, :n] = wins, games
    w[n, :n], g[n, :n] = wins[i], games[i]  # clone vs everyone = original's record
    w[:n, n], g[:n, n] = wins[:, i], games[:, i]
    typical = float(np.median(games[games > 0])) if np.any(games > 0) else 2.0
    g[i, n] = g[n, i] = typical  # original vs clone: an even split
    w[i, n] = w[n, i] = typical / 2.0

    before = _fit_elo(names, wins, games)
    after = _fit_elo(new_names, w, g)
    centred = {a: after[a] - before[a] for a in names}
    anchor = ANCHOR_FALLBACK if clone_of == ANCHOR else ANCHOR
    others = [a for a in names if a != clone_of]
    shifts = {a: centred[a] - centred[anchor] for a in others}
    # Ranking of the original agents before and after the injection.
    order_before = sorted(names, key=lambda a: -before[a])
    order_after = sorted(names, key=lambda a: -after[a])
    return {
        "clone_of": clone_of,
        "shifts": shifts,
        "max_abs_shift": max(abs(v) for v in shifts.values()),
        "spread_others": max(shifts.values()) - min(shifts.values()),
        "max_abs_shift_centred_all": max(abs(v) for v in centred.values()),
        "order_changed": order_before != order_after,
    }


# --- 3: is the selection clone-robust? --------------------------------------- #


def _paired_row(k: int, strategy: str, res: dict, seeds: int) -> dict:
    """Summarise the two conditions of one (K, strategy) cell and their paired difference."""
    # Paired difference: both conditions use the same seeds.
    diff = np.array(res["base+clones"][2]) - np.array(res["base"][2])
    return {
        "k": k,
        "strategy": strategy,
        "tau_base": res["base"][0],
        "sem_base": res["base"][1],
        "tau_cloned": res["base+clones"][0],
        "sem_cloned": res["base+clones"][1],
        "diff": float(diff.mean()),
        "diff_sem": float(diff.std(ddof=1) / np.sqrt(seeds)),
    }


def selection_robustness(cache: ScoreCache, ks, seeds: int, config: dict) -> list[dict]:
    """Recover the base ranking when the selection sees the base panel, with or without clones.

    Both conditions score the same eight agents on the same estimand; only the agents the selection
    reads differ.
    """
    base = config["base_panel"]
    names = cache.names
    d = np.load(cache.path, allow_pickle=True)
    choices_all = np.asarray(d["choices"])
    phases = cache.phases
    idx_of = {a: names.index(a) for a in names}

    base_rows = [idx_of[a] for a in base]
    clones = [a for fam in config["clone_families"].values() for a in fam]
    wide_rows = base_rows + [idx_of[a] for a in clones]

    y_base = np.stack([cache.losses[a] for a in base])
    truth = rr.ranking(y_base.mean(axis=1) * PR_PER_EQUITY)

    losses_all = np.stack([cache.losses[a] for a in names])
    out = []
    for k in ks:
        for strat in config["strategies"]:
            res = {}
            for condition, rows in (("base", base_rows), ("base+clones", wide_rows)):
                taus = []
                for s in range(seeds):
                    # active_rank steers on these same losses (skyline importance).
                    sl = sel.select(
                        strat,
                        k,
                        losses=losses_all[rows],
                        choices=choices_all[rows],
                        phases=phases,
                        seed=s,
                        importance=config["importance"],
                    )
                    pr = rr.pr_subset(y_base, list(sl.indices))
                    taus.append(rr.kendall_vs_full(rr.ranking(pr), truth))
                res[condition] = (
                    float(np.mean(taus)),
                    float(np.std(taus, ddof=1) / np.sqrt(seeds)),
                    taus,
                )
            out.append(_paired_row(k, strat, res, seeds))
        r = [x for x in out if x["k"] == k]
        print(f"  K={k:5d}  " + "  ".join(f"{x['strategy'][:9]}:{x['diff']:+.4f}" for x in r))
    return out


# --- report ------------------------------------------------------------------ #


def _flag(row: dict, n_se) -> str:
    """Mark a paired difference beyond ``n_se`` standard errors, or an identical sample."""
    if row["diff"] == 0.0 and row["diff_sem"] == 0.0:
        # The strategy reads no agent data, so both conditions drew the identical sample.
        return "identical sample"
    if abs(row["diff"]) < n_se * row["diff_sem"]:
        return ""
    return f"below {n_se} SE" if row["diff"] < 0 else f"above {n_se} SE"


def _summary_rows(pr_res, elo_res, selrows, n_base, n_agents, n_pool) -> list[list[str]]:
    """Rows of the summary table: the headline number of each check."""
    worst_centred = max(elo_res, key=lambda e: e["max_abs_shift_centred_all"])
    worst = min(selrows, key=lambda x: x["diff"])
    return [
        [
            f"Largest PR shift of a base agent, {n_base} to {n_agents} agents (PR points)",
            f"{pr_res['max_abs_shift']:.10f}",
        ],
        [
            f"Largest centred Elo shift (duplicating `{worst_centred['clone_of']}`)",
            f"{worst_centred['max_abs_shift_centred_all']:.1f}",
        ],
        [
            "Largest change in an Elo difference between two other agents",
            f"{max(e['spread_others'] for e in elo_res):.1f}",
        ],
        [
            f"Largest paired tau change (`{worst['strategy']}`, K={worst['k']})",
            f"{worst['diff']:+.4f} ±{worst['diff_sem']:.4f}",
        ],
        ["Decisions in the pool", f"{n_pool:,}"],
    ]


def report_tables(pr_res, elo_res, selrows, seeds, n_pool, n_agents, config) -> list:
    """Build the (title, header, rows) tables of the three checks."""
    n_base = len(config["base_panel"])
    elo_rows = [
        [
            f"`{e['clone_of']}`",
            f"{e['max_abs_shift']:.1f}",
            f"{e['spread_others']:.1f}",
            "yes" if e["order_changed"] else "no",
        ]
        for e in elo_res
    ]
    n_se = config["flag_standard_errors"]
    sel_rows = [
        [
            str(x["k"]),
            f"`{x['strategy']}`",
            f"{x['tau_base']:.4f} ±{x['sem_base']:.4f}",
            f"{x['tau_cloned']:.4f} ±{x['sem_cloned']:.4f}",
            f"{x['diff']:+.4f} ±{x['diff_sem']:.4f}",
            _flag(x, n_se),
        ]
        for x in selrows
    ]
    elo_title = (
        f"Elo shift after duplicating one agent ({ANCHOR} pinned, {ANCHOR_FALLBACK} when "
        f"{ANCHOR} is duplicated)"
    )
    return [
        (
            "Summary",
            ["quantity", "value"],
            _summary_rows(pr_res, elo_res, selrows, n_base, n_agents, n_pool),
        ),
        (
            elo_title,
            ["duplicated agent", "max shift of the others", "spread of the others", "reordered"],
            elo_rows,
        ),
        (
            f"Kendall tau of the base ranking over {seeds} draws, paired change "
            "(base+clones minus base)",
            ["K", "strategy", "tau (base)", "tau (base+clones)", "paired change", "flag"],
            sel_rows,
        ),
    ]


def write_report(doc: Path, tables: list, numbers: dict) -> None:
    """Write the Markdown tables to ``doc`` and the numbers to the JSON file next to it."""
    parts = [f"# {EXPERIMENT}"]
    for title, header, rows in tables:
        parts += [f"## {title}", markdown_table(header, rows)]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {doc}")
    json_path = doc.with_suffix(".json")
    json_path.write_text(json.dumps(numbers, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")


def main() -> None:
    """Run the three clone-invariance checks and write the report."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    base = config["base_panel"]
    cache = ScoreCache(args.cache)
    missing = [a for a in base if a not in cache.losses]
    if missing:
        raise SystemExit(f"base panel agents missing from {args.cache}: {missing}")
    print(f"pool {cache.n_pos:,} decisions; {len(cache.names)} agents in cache")

    pr_res = pr_invariance(cache, base)
    print(f"PR max shift base->wide: {pr_res['max_abs_shift']:.10f}")

    round_robin = json.loads(args.round_robin.read_text(encoding="utf-8"))
    # Duplicate every agent in turn; rows are sorted by the largest shift.
    elo_res = sorted(
        (elo_with_clone(round_robin, a) for a in round_robin["names"]),
        key=lambda e: -e["max_abs_shift"],
    )
    for e in elo_res:
        print(
            f"Elo: cloning {e['clone_of']:8s} -> max shift {e['max_abs_shift']:6.1f}, "
            f"order changed: {e['order_changed']}"
        )

    print("selection robustness (paired Δτ, base+clones − base):")
    selrows = selection_robustness(cache, args.ks, args.seeds, config)

    n_agents = len(cache.names)
    tables = report_tables(pr_res, elo_res, selrows, args.seeds, cache.n_pos, n_agents, config)
    numbers = {
        "n_decisions": cache.n_pos,
        "n_agents": n_agents,
        "seeds": args.seeds,
        "anchor": ANCHOR,
        "anchor_fallback": ANCHOR_FALLBACK,
        "pr": pr_res,
        "elo": elo_res,
        "selection": selrows,
    }
    write_report(args.doc or output_dir(EXPERIMENT) / f"{EXPERIMENT}.md", tables, numbers)
    print("done.")


if __name__ == "__main__":
    main()
