"""Markdown report of the oracle-depth experiment."""

from __future__ import annotations


def format_tau(value) -> str:
    """Format an optional tau value."""
    return "n/a" if value is None else f"{value:+.3f}"


def header_lines(args, onames, rankings, suites, tau_robust) -> list[str]:
    """Return the title, the introduction and the setup section."""
    return [
        "# Oracle-depth / cross-oracle sensitivity",
        "",
        "Every agent scored on the **same** frozen position set under each oracle in the ladder "
        "(same seed+sampler ⇒ identical positions, only the frozen equities differ). The question: "
        "does the PR ranking depend on which oracle judges it? It is *oracle-robust* if the strong "
        f"oracles agree (Kendall-τ ≥ {tau_robust}) while the weak control (pubeval) agrees clearly "
        "less.",
        "",
        "## Setup",
        "",
        f"- Agents ({len(rankings[onames[0]])}): {', '.join(rankings[onames[0]])} "
        f"(order = PR rank under `{onames[0]}`).",
        f"- Oracle ladder: {', '.join(onames)}.",
        f"- Positions: {len(suites[onames[0]]):,} (1 per game → independent), sampled from "
        f"self-play of `{args.sampler}`; seed {args.seed}.",
        "- **pubeval is uncalibrated → rank-only.** Its PR *magnitudes* are not comparable to "
        "gnubg PR; only its rank agreement (τ) is meaningful.",
        "",
    ]


def tau_lines(onames, tau, tau_robust) -> list[str]:
    """Return the Kendall tau matrix between the oracle rankings."""
    lines = [
        "## Kendall-τ between oracle rankings",
        "",
        f"τ = +1 identical ranking, −1 reversed. Strong-strong τ ≥ {tau_robust} ⇒ robust.",
        "",
        "| oracle | " + " | ".join(onames) + " |",
        "|---|" + "---|" * len(onames),
    ]
    for a in onames:
        lines.append("| " + a + " | " + " | ".join(f"{tau[a][b]:+.3f}" for b in onames) + " |")
    return lines


def pr_lines(onames, rankings, pr_by_oracle) -> list[str]:
    """Return the PR table of every agent under every oracle."""
    lines = [
        "",
        "## PR by agent under each oracle (lower = better)",
        "",
        "| agent | " + " | ".join(f"{o} PR" for o in onames) + " |",
        "|---|" + "---|" * len(onames),
    ]
    for agent in rankings[onames[0]]:
        cells = []
        for o in onames:
            cells.append("rank-only" if o == "pubeval" else f"{pr_by_oracle[o][agent]:.1f}")
        lines.append(f"| {agent} | " + " | ".join(cells) + " |")
    return lines


def diff_lines(diffs) -> list[str]:
    """Return the best-action disagreement table of the strong oracle pairs."""
    lines = [
        "",
        "## Best-action disagreement (strong oracles)",
        "",
        "Positions where the argmax-equity (oracle-best) move differs between the two oracles.",
        "",
        "| oracle pair | positions differing | total | % |",
        "|---|---|---|---|",
    ]
    for (a, b), (d, t) in diffs.items():
        pct = 100.0 * d / t if t else 0.0
        lines.append(f"| {a} vs {b} | {d} | {t} | {pct:.1f}% |")
    return lines


def verdict_lines(verdict, tau_robust) -> list[str]:
    """Return the verdict section."""
    lines = ["", "## Verdict", ""]
    lines.append(
        f"- τ(rollout, 2-ply): {format_tau(verdict['tau_rollout_2ply'])} (need ≥ {tau_robust})"
    )
    lines.append(f"- τ(rollout, XG): {format_tau(verdict['tau_rollout_xg'])} (need ≥ {tau_robust})")
    sp = verdict["sign_preservation"]
    status = "all preserved" if sp["all"] else "NOT all preserved"
    lines.append(
        f"- Δ-PR sign preservation ({sp['pair'][0]} vs {sp['pair'][1]}): "
        f"{sp['preserved']}/{sp['pairs']} pairs → {status}"
    )
    pc = verdict["pubeval_control"]
    status = "clearly lower ✓" if pc["clearly_lower"] else "NOT clearly lower"
    lines.append(
        f"- pubeval control: τ(pubeval, strongest)={pc['pubeval_vs_strongest']:+.3f} "
        f"vs strong-strong τ={pc['strong_strong_tau']:+.3f} → {status}"
    )
    verdict_str = "ORACLE-ROBUST ✓" if verdict["robust"] else "NOT robust ✗"
    lines += ["", f"**Verdict: {verdict_str}**", "", f"_{verdict['note']}._", ""]
    return lines


def render(args, onames, rankings, pr_by_oracle, tau, diffs, verdict, suites, tau_robust) -> str:
    """Return the markdown report: tau matrix, PR table, best-action disagreement and verdict."""
    lines = header_lines(args, onames, rankings, suites, tau_robust)
    lines += tau_lines(onames, tau, tau_robust)
    lines += pr_lines(onames, rankings, pr_by_oracle)
    lines += diff_lines(diffs)
    lines += verdict_lines(verdict, tau_robust)
    return "\n".join(lines)
