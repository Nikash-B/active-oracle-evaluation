"""Markdown report and JSON summary of wildbg in the ten-agent anchored-Elo pool."""

import json


def report_lines(run, stats):
    """Return the lines of the Markdown report."""
    labels9, base, games = run["labels9"], run["base"], run["games"]
    wi, n, seed = run["wi"], run["n"], run["seed"]
    elo, cis, pr, n_dec = stats["elo"], stats["cis"], stats["pr"], stats["n_dec"]
    pr_order, elo_order = stats["pr_order"], stats["elo_order"]
    pr_rank = {x: k + 1 for k, x in enumerate(pr_order)}
    elo_rank = {x: k + 1 for k, x in enumerate(elo_order)}
    first_pi = run["first_pairing_index"]
    last_pi = first_pi + len(labels9) - 1
    lines = [
        "# wildbg added to the nine-agent anchored-Elo pool",
        "",
        f"The {len(labels9)}-agent {run['base_games']}-game base ({run['base_stem']}) is unchanged;",
        f"wildbg adds {len(labels9)} pairings x {run['pairs_total']} duplicate pairs = "
        f"{int(games[wi].sum())} games on fresh banks",
        f"(pairing indices {first_pi}..{last_pi} of seed {seed}).",
        "Elo CIs: game-level binomial bootstrap, pivotal. wildbg's PR is from its own",
        "unforced decisions in these games (gnubg-nn 0-ply oracle); incumbents",
        "keep their base-run PR.",
        "",
        "## Ratings (sorted by Elo)",
        "",
        "| agent | Elo | 95% CI | PR | n dec | Elo rank | PR rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in elo_order:
        lo, hi = cis[x]
        lines.append(
            f"| {x} | {elo[x]:+.0f} | [{lo:+.0f}, {hi:+.0f}] | {pr[x]:.2f} | {n_dec[x]} "
            f"| {elo_rank[x]} | {pr_rank[x]} |"
        )
    points_mean = run["points_mean"]
    lines += [
        "",
        f"## Validity ({n}-agent pool)",
        "",
        f"- **Kendall tau (PR vs Elo) = {stats['tau']:+.3f}** (perm p = {stats['tau_p']:.4f}); "
        f"{len(labels9)}-agent base tau was {base['tau_full']:+.3f}.",
        f"- PR-gap -> win-prob: log-odds slope {stats['slope']:+.4f}/PR point, "
        f"R2 {stats['r2']:.3f} over {stats['n_fit']} ordered pairs"
        f" ({len(labels9)}-agent base: {base['slope']:+.4f}, R2 {base['r2']:.3f}).",
        f"- wildbg mean log-odds residual vs the pooled fit: {stats['wildbg_resid']:+.3f}"
        " (positive = wins more than its PR predicts).",
        "",
        "## wildbg row (mean ppg vs column, fresh banks)",
        "",
        "| vs | " + " | ".join(labels9) + " |",
        "|---|" + "---|" * (n - 1),
        "| wildbg | " + " | ".join(f"{points_mean[wi, j]:+.2f}" for j in range(n - 1)) + " |",
        "",
    ]
    return lines


def summary_json(run, stats):
    """Return the JSON summary text: ratings, validity statistics and the merged matrices."""
    return json.dumps(
        {
            "labels": run["labels"],
            "elo": stats["elo"],
            "elo_ci": stats["cis"],
            "pr": stats["pr"],
            "n_dec": stats["n_dec"],
            "tau_full": stats["tau"],
            "tau_full_p": stats["tau_p"],
            "slope": stats["slope"],
            "r2": stats["r2"],
            "wildbg_resid": stats["wildbg_resid"],
            "wins": run["wins"].tolist(),
            "games": run["games"].tolist(),
            "points": run["points_mean"].tolist(),
            "seed": run["seed"],
            "pair_ppg_shards": run["pair_ppg"],
            "wall_play_s": run["t_play"],
            "wall_pr_s": stats["t_pr"],
        },
        indent=1,
    )
