"""Markdown report, JSON summary and per-agent CSV of the panel_ratings experiment."""

from __future__ import annotations

import csv
from pathlib import Path


def write_panel_csv(path: Path, scores, cis, ranks) -> None:
    """Write one row per agent, best first, at the precision of the report's table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["agent", "n", "pr", "pr_ci_low", "pr_ci_high", "rank_low", "rank_high"])
        for s in scores:
            lo, hi = cis[s.agent]
            rlo, rhi = ranks[s.agent]
            w.writerow([s.agent, s.n, f"{s.pr:.1f}", f"{lo:.1f}", f"{hi:.1f}", rlo, rhi])
    print(f"wrote {path}")


def _one_decimal(x: float) -> float:
    """Return ``x`` at the one-decimal precision of the report."""
    return float(f"{x:.1f}")


def summary_numbers(args, scores, cis, contrasts, ranks, holm, n_positions: int) -> dict:
    """Collect the report's numbers, at the report's precision, as a JSON-serialisable dict."""
    agents = []
    for i, s in enumerate(scores, 1):
        lo, hi = cis[s.agent]
        rlo, rhi = ranks[s.agent]
        agents.append(
            {
                "rank": i,
                "agent": s.agent,
                "n": s.n,
                "epm": _one_decimal(s.epm),
                "pr": _one_decimal(s.pr),
                "pr_ci": [_one_decimal(lo), _one_decimal(hi)],
                "rank_interval": [rlo, rhi],
            }
        )
    adjacent = [
        {
            "worse": c.a,
            "better": c.b,
            "delta_pr": _one_decimal(c.delta_pr),
            "ci": [_one_decimal(c.ci_low), _one_decimal(c.ci_high)],
            "distinguishable": c.distinguishable,
        }
        for c in contrasts
    ]
    holm_rows = [
        {
            "worse": a,
            "better": b,
            "delta_pr": _one_decimal(delta),
            "p_raw": float(f"{p_raw:.3f}"),
            "p_holm": float(f"{p_holm:.3f}"),
            "significant": bool(sig),
        }
        for a, b, delta, p_raw, p_holm, sig in holm
    ]
    return {
        "positions": n_positions,
        "seed": args.seed,
        "plies": args.plies,
        "samplers": list(args.samplers),
        "agents": agents,
        "adjacent_contrasts": adjacent,
        "holm_adjacent_contrasts": holm_rows,
    }


def _header_lines(args, scores, suite, sampler_names, t_build, t_score) -> list[str]:
    """Render the report's title, introduction and setup."""
    return [
        "# Panel suite PR table (paired)",
        "",
        "Every agent scored on the **same** frozen, held-out position suite (the panel suite): "
        "for each position the oracle's mover-equity over all legal moves is the ground truth, "
        "and an agent loses `best − chosen`. Because the positions are shared, between-agent PR "
        "differences are *paired* per-position contrasts in which the oracle's selection bias "
        "and the between-position variance cancel.",
        "",
        "## Setup",
        "",
        f"- Agents ({len(scores)}): {', '.join(s.agent for s in scores)}.",
        f"- Suite: {len(suite):,} positions (1 per game → independent), sampled from self-play of "
        f"{', '.join(sampler_names)}; seed {args.seed}. "
        f"Oracle: gnubg-nn {args.plies}-ply cubeless.",
        f"- Wall: {t_build:.0f}s suite build + {t_score:.0f}s scoring.",
        "",
    ]


def _pr_table_lines(scores, cis, ranks) -> list[str]:
    """Render the PR-by-agent table."""
    lines = [
        "## PR by agent (paired suite; sorted best → worst)",
        "",
        "PR ≈ performance rating (lower better); CI is a 95% bootstrap on the *absolute* PR (still "
        "carries the oracle bias — use the paired contrasts below for between-agent claims).",
        "",
        "| rank | agent | n | EPM (mpts) | PR | PR 95% CI | 95% rank interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scores, 1):
        lo, hi = cis[s.agent]
        rlo, rhi = ranks[s.agent]
        rank_str = f"{rlo}" if rlo == rhi else f"{rlo}–{rhi}"
        lines.append(
            f"| {i} | {s.agent} | {s.n:,} | {s.epm:.1f} | {s.pr:.1f} "
            f"| [{lo:.1f}, {hi:.1f}] | {rank_str} |"
        )
    return lines


def _contrast_lines(contrasts, holm, n_boot: int, n_positions: int) -> list[str]:
    """Render the adjacent paired contrasts and their Holm-corrected p-values."""
    lines = [
        "",
        "## Adjacent paired contrasts (is each rank separable from the next?)",
        "",
        "Δ-PR = (worse agent) − (better agent), paired bootstrap 95% CI over positions; "
        "**distinguishable** when the CI excludes 0.",
        "",
        "| pair (worse − better) | Δ-PR | 95% CI | distinguishable |",
        "|---|---|---|---|",
    ]
    for c in contrasts:
        mark = "**yes**" if c.distinguishable else "no"
        lines.append(
            f"| {c.a} − {c.b} | {c.delta_pr:+.1f} | [{c.ci_low:+.1f}, {c.ci_high:+.1f}] | {mark} |"
        )
    lines += [
        "",
        "## Simultaneous rank intervals and the Holm-corrected adjacent-rank contrasts",
        "",
        "The **rank-interval** column above is the multiplicity-robust ranking summary "
        f"(each agent's 95% rank across {n_boot} position-bootstraps; two agents are "
        "*statistically distinguishable* iff their rank intervals don't overlap). "
        "The adjacent-rank contrasts below form one family, "
        "with two-sided paired-bootstrap p-values **Holm-corrected** at α=0.05 "
        "(family-wise error control).",
        "",
        "| pair (worse − better) | Δ-PR | p (raw) | p (Holm) | significant |",
        "|---|---|---|---|---|",
    ]
    for a, b, delta, p_raw, p_holm, sig in holm:
        lines.append(
            f"| {a} − {b} | {delta:+.1f} | {p_raw:.3f} | {p_holm:.3f} "
            f"| {'**yes**' if sig else 'no'} |"
        )
    lines += [
        "",
        f"(Paired panel suite; oracle bias cancels in the contrasts. Positions: {n_positions:,}.)",
        "",
    ]
    return lines


def render(
    args, scores, cis, contrasts, suite, sampler_names, t_build, t_score, ranks, holm, n_boot
) -> str:
    """Render the markdown report: PR table, adjacent contrasts and the Holm family."""
    lines = _header_lines(args, scores, suite, sampler_names, t_build, t_score)
    lines += _pr_table_lines(scores, cis, ranks)
    lines += _contrast_lines(contrasts, holm, n_boot, len(suite))
    return "\n".join(lines)


def agreement_lines(agreement: dict) -> str:
    """Render the Kendall tau of the PR order against anchored Elo and any compared run."""
    titles = {"anchored_elo": "Anchored Elo", "compared_run": "Compared panel_ratings run"}
    lines = [
        "",
        "## Agreement of the PR order with other orders",
        "",
        "| reference | source | Kendall tau | discordant pairs |",
        "|---|---|---|---|",
    ]
    for key, row in agreement.items():
        pairs = ", ".join(f"{a}–{b}" for a, b in row["discordant"]) or "–"
        lines.append(f"| {titles[key]} | {row['source']} | {row['kendall_tau']:.3f} | {pairs} |")
    return "\n".join(lines) + "\n"
