"""Markdown report and JSON summary of the oracle-pool round-robin."""

from __future__ import annotations


def _setup_lines(run):
    """Return the title, the introduction and the setup section of the report."""
    labels = run["labels"]
    n = len(labels)
    plies = run["plies"]
    return [
        "# Anchored Elo with the oracle in the pool",
        "",
        "One round-robin play pass (duplicate dice) supplies both instruments: Bradley-Terry",
        "Elo on wins (games-based, oracle-free) and oracle-PR on the same games' unforced",
        f"decisions (gnubg-nn {plies}-ply oracle). The strong-tier tau measures how far",
        "PR validity extends into the tier at or above the oracle's own strength.",
        "",
        "## Setup",
        "",
        f"- Pool ({n}): {', '.join(labels)}; anchor `{run['anchor_name']}` = Elo 0.",
        f"- {n * (n - 1) // 2} pairings x {run['pairs_total']} duplicate pairs "
        f"({run['shards']} independent dice banks) = {int(run['games'].sum()) // 2} games; "
        f"seed {run['seed']}.",
        f"- PR: gnubg-nn {plies}-ply oracle over {sum(run['n_dec'].values())} unforced decisions.",
        f"- Wall: {run['t_play']:.0f}s play + {run['t_pr']:.0f}s scoring; {run['workers']} workers.",
        "",
    ]


def _ratings_lines(run):
    """Return the ratings table, sorted by Elo."""
    elo, pr, n_dec, cis = run["elo"], run["pr"], run["n_dec"], run["cis"]
    pr_rank = {x: k + 1 for k, x in enumerate(run["pr_order"])}
    elo_rank = {x: k + 1 for k, x in enumerate(run["elo_order"])}
    lines = [
        "## Ratings (sorted by Elo)",
        "",
        "| agent | Elo | 95% CI | PR | n dec | Elo rank | PR rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in run["elo_order"]:
        lo, hi = cis[x]
        lines.append(
            f"| {x} | {elo[x]:+.0f} | [{lo:+.0f}, {hi:+.0f}] | {pr[x]:.2f} | {n_dec[x]} "
            f"| {elo_rank[x]} | {pr_rank[x]} |"
        )
    return lines


def _validity_lines(run):
    """Return the validity section and the points matrix."""
    labels = run["labels"]
    n = len(labels)
    points = run["points"]
    strong = run["strong"]
    lines = [
        "",
        "## Validity",
        "",
        f"- **Full pool ({len(run['pr'])}): Kendall tau (PR vs Elo) = {run['tau']:+.3f}** "
        f"(perm p = {run['tau_p']:.4f}).",
        f"- **Strong tier ({len(strong)}: {', '.join(strong)}): "
        f"tau = {run['tau_strong']:+.3f}** (perm p = {run['tau_strong_p']:.4f}).",
        f"- PR-gap -> win-prob: log-odds slope {run['slope']:+.4f}/PR point, "
        f"R2 {run['r2']:.3f} over {run['n_fit']} ordered pairs.",
        "",
        "## Points matrix (row's mean ppg vs column)",
        "",
        "| vs | " + " | ".join(labels) + " |",
        "|---|" + "---|" * n,
    ]
    for i, x in enumerate(labels):
        row = [f"{points[i, j]:+.2f}" if j != i else "-" for j in range(n)]
        lines.append(f"| {x} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_report(run):
    """Return the Markdown report of one run."""
    return "\n".join(_setup_lines(run) + _ratings_lines(run) + _validity_lines(run))


def summary(run):
    """Return the JSON summary of one run."""
    return {
        "labels": run["labels"],
        "elo": run["elo"],
        "elo_ci": run["cis"],
        "pr": run["pr"],
        "n_dec": run["n_dec"],
        "tau_full": run["tau"],
        "tau_full_p": run["tau_p"],
        "tau_strong": run["tau_strong"],
        "tau_strong_p": run["tau_strong_p"],
        "strong_tier": run["strong"],
        "slope": run["slope"],
        "r2": run["r2"],
        "wins": run["wins"].tolist(),
        "games": run["games"].tolist(),
        "points": run["points"].tolist(),
        "seed": run["seed"],
        "trial_wins": {f"{i},{j}": [int(w) for w in v] for (i, j), v in run["trial_wins"].items()},
        "pairs_per_pairing": run["pairs_total"],
        "wall_play_s": run["t_play"],
        "wall_pr_s": run["t_pr"],
    }
