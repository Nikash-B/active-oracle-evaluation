"""Markdown report of the Nash-averaging experiment."""


def result_lines(r, sidecar_name):
    """Return the header, method summary and Nash rating table."""
    n, inv = r["n"], r["inv"]
    lines = [
        "# Nash averaging vs Performance Rating",
        "",
        f"Computed from `{sidecar_name}` — the {n}-agent round-robin at "
        f"{r['games_per_pair']:,} games per pairing ({r['total_games']:,} games).",
        "",
        "Method: antisymmetric advantage matrix `A_ij = 2p_ij − 1`, maximum-entropy Nash "
        "equilibrium `p*` of the resulting symmetric zero-sum game, agent rating `(A p*)_i`.",
        "",
        "## Result",
        "",
        f"**Kendall τ(PR, Nash averaging) = {r['tau']:+.3f}** ({inv} discordant pair"
        f"{'' if inv == 1 else 's'} of {n * (n - 1) // 2}).",
        "",
        "| agent | Nash rating | Nash mass | PR |",
        "|---|---|---|---|",
    ]
    for a in r["nash_order"]:
        lines.append(
            f"| {a} | {r['nash_rating'][a]:+.4f} | {r['nash_mass'][a]:.4f} | {r['oracle_pr'][a]} |"
        )
    return lines


def comparison_lines(r):
    """Return the equilibrium support and the Bradley-Terry cross-check."""
    same_order = r["bt_order"] == r["nash_order"]
    return [
        "",
        f"**Equilibrium support: {len(r['support'])} of {r['n']} agents** — "
        + ", ".join(f"`{a}` {m:.3f}" for a, m in r["support"])
        + ".",
        "",
        f"**Cross-check against Bradley–Terry on the same matrix:** τ(PR, BT) = "
        f"{r['tau_bt']:+.3f} ({r['inv_bt']} discordant). The Nash and BT orderings are "
        + ("**identical**" if same_order else "**different**")
        + ". "
        + (
            "So the disagreement with PR is a property of *head-to-head play*, not of Nash "
            "averaging specifically — two different rating methods on the same games place the "
            "same agents in the same order, and both differ from PR on the same pairs."
            if same_order
            else ""
        ),
    ]


def scope_lines(r):
    """Return the scope section: transitivity and what a pure equilibrium implies."""
    lines = [
        "",
        "## Scope",
        "",
        f"The panel is **{'transitive' if r['transitive'] else 'NOT transitive'}** (checked on "
        "the same matrix). ",
    ]
    if r["transitive"] and len(r["support"]) == 1:
        lines += [
            "On a transitive field the equilibrium collapses onto the single dominant agent, so "
            "every other agent's Nash rating reduces to *its advantage against that one agent*. "
            "This panel therefore exercises little of what makes Nash averaging different from a "
            "scalar rating.",
            "",
            "PR and Nash averaging agree *where the field is transitive*. This does not show that "
            "PR would track Nash averaging on a cyclic field, which this panel does not contain.",
        ]
    return lines


def method_lines():
    """Return the method notes."""
    return [
        "",
        "## Method notes",
        "",
        "- The equilibrium set of a symmetric zero-sum game is generally a polytope; the "
        "**maximum-entropy** equilibrium is used so the ratings depend on the data rather than on "
        "the LP's pivoting. Stage 1 finds the set by LP (`A p ≤ 0`, `Σp = 1`, `p ≥ 0`), stage 2 "
        "maximises entropy inside it.",
        "- The constraint `A p ≤ 0` states that no pure strategy gains against `p`.",
        "- Antisymmetry of `A` is asserted, not assumed: a non-antisymmetric matrix would make the "
        "game value non-zero and silently invalidate the construction.",
        "- Win probabilities come from completed head-to-head play, so no model is interposed "
        "between the games and the rating.",
    ]


def report_lines(r, sidecar_name):
    """Return every line of the Markdown report."""
    return result_lines(r, sidecar_name) + comparison_lines(r) + scope_lines(r) + method_lines()
