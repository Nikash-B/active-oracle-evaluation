"""Markdown report of the label-efficiency experiment."""

from active_oracle_evaluation.experiments.label_efficiency_stats import (
    DESIGN_SOURCE,
    STRATEGIES,
    TAU_THRESHOLDS,
    TIGHT_PAIRS,
    labels_to_threshold,
)


def _k_header(first, ks):
    """Return the header and separator lines of a table with one column per K."""
    return [
        f"| {first} | " + " | ".join(f"K={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]


def _tau_table(agg, ks, header):
    """Return markdown lines for a strategy × K table of mean Kendall-τ with seed bands."""
    lines = ["", header, "", *_k_header("strategy", ks)]
    for strat in STRATEGIES:
        cells = []
        for k in ks:
            mean, _, lo, hi = agg[strat][k]["kendall_tau"]
            cells.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
        lines.append(f"| {strat} | " + " | ".join(cells) + " |")
    return lines


def _cell(value):
    """Format an optional K for a markdown cell."""
    return "—" if value is None else str(value)


def _setup_lines(cl_names, ks, full_order, full_pr, names, seeds):
    """Return the title, setup and full-suite PR lines."""
    full_names = [names[i] for i in full_order]
    lines = [
        "# Evaluation efficiency — active vs uniform oracle selection",
        "",
        "Each strategy selects K positions; the agent ranking is induced from the plain "
        "subset-mean oracle regret on those K and compared to the full-pool ranking (Kendall-τ). "
        "The all-agent ranking is easy (outliers trivially separable), so it saturates; the "
        "**near-tied competitive cluster** is the discriminating test. PR-MAE uses the unbiased "
        "Horvitz–Thompson estimate.",
        "",
        "## Setup",
        "",
        f"- Full panel ({len(names)}): {', '.join(names)}; full-pool ranking (best→worst): "
        f"{', '.join(full_names)}.",
        f"- Competitive cluster ({len(cl_names)}): {', '.join(cl_names)}.",
        f"- Tight pairs: {', '.join(f'{a}-{b}' for a, b in TIGHT_PAIRS)}.",
        f"- Budgets K: {ks}; seeds per cell: {seeds}. Ranking on plain subset-mean PR; "
        "selection per panel.",
        "",
        "## Full-suite PR (reference)",
        "",
        "| agent | PR |",
        "|---|---|",
    ]
    for i in full_order:
        lines.append(f"| {names[i]} | {full_pr[i]:.1f} |")
    return lines


def _ht_lines(agg_full, agg_cl, ks):
    """Return the plain-mean versus Horvitz-Thompson Kendall-τ tables of both panels."""
    lines = []
    for agg, label in ((agg_cl, "competitive cluster"), (agg_full, "all 8")):
        lines += [
            "",
            f"## Kendall-τ ranked by plain mean → by Horvitz–Thompson PR ({label}; same draws)",
            "",
            "| strategy | design | " + " | ".join(f"K={k}" for k in ks) + " |",
            "|---|---|" + "---|" * len(ks),
        ]
        for strat in STRATEGIES:
            cells = " | ".join(
                f"{agg[strat][k]['kendall_tau'][0]:.3f} → {agg[strat][k]['kendall_tau_ht'][0]:.3f}"
                for k in ks
            )
            lines.append(f"| {strat} | {DESIGN_SOURCE[strat]} | " + cells + " |")
    return lines


def _cluster_metric_lines(agg_cl, ks):
    """Return the tight-pair accuracy and CI-width tables of the cluster."""
    lines = [
        "",
        "## Tight-pair ordering accuracy vs K",
        "",
        "Fraction of the hardest near-tied contrasts ordered correctly vs the full pool (cluster "
        "run, mean over seeds).",
        "",
        *_k_header("strategy", ks),
    ]
    for strat in STRATEGIES:
        cells = " | ".join(f"{agg_cl[strat][k]['tight_pair_acc'][0]:.3f}" for k in ks)
        lines.append(f"| {strat} | " + cells + " |")
    lines += [
        "",
        "## Mean per-agent PR CI-width vs K (cluster; absolute-PR precision)",
        "",
        "Tighter = more precise absolute PR. active_rank concentrates labels, sharpening ranking "
        "but *widening* absolute-PR CIs; disagreement stays near random.",
        "",
        *_k_header("strategy", ks),
    ]
    for strat in STRATEGIES:
        cells = " | ".join(f"{agg_cl[strat][k]['mean_ci_width'][0]:.1f}" for k in ks)
        lines.append(f"| {strat} | " + cells + " |")
    return lines


def _threshold_lines(agg_cl, ks, saving):
    """Return the labels-to-threshold table of the cluster."""
    tgt = saving["target_cluster_tau"]
    threshold_header = "".join(f" | τ_cluster≥{t:.2f}" for t in TAU_THRESHOLDS)
    lines = [
        "",
        "## Labels-to-threshold: smallest K reaching a cluster-τ level",
        "",
        f"`—` = not reached within budget. **Target column** = random's max-K cluster-τ "
        f"({tgt:.3f}); fewer labels to reach it = more efficient.",
        "",
        f"| strategy | reach random's K={ks[-1]} τ ({tgt:.3f}) | ×fewer labels"
        + threshold_header
        + " |",
        "|---|---|---|" + "---|" * len(TAU_THRESHOLDS),
    ]
    l2t = labels_to_threshold(agg_cl, ks, "kendall_tau", TAU_THRESHOLDS)
    for strat in STRATEGIES:
        s = saving["savings"].get(strat)
        reach = _cell(saving["reach_k"][strat])
        xf = "—" if not s or s["x_fewer"] is None else f"{s['x_fewer']:.1f}×"
        if strat == "random":
            xf = "1.0× (baseline)"
        cells = " | ".join(_cell(l2t[strat][t]) for t in TAU_THRESHOLDS)
        lines.append(f"| {strat} | {reach} | {xf} | " + cells + " |")
    return lines


def _loao_lines(loao, ks):
    """Return the leave-one-agent-out table."""
    return [
        "",
        "## Leave-one-agent-out robustness (disagreement selection, full panel)",
        "",
        "Each agent held out in turn; positions selected using only the *other* agents' "
        "disagreement, then the full panel scored. `frac_exact` = fraction of held-out agents "
        "whose rank is recovered exactly. High ⇒ selection doesn't distort unseen agents' ranks.",
        "",
        *_k_header("metric", ks),
        "| frac_exact | " + " | ".join(f"{loao[k]['frac_exact']:.3f}" for k in ks) + " |",
        "| mean_rank_disp | " + " | ".join(f"{loao[k]['mean_rank_disp']:.2f}" for k in ks) + " |",
    ]


def _saving_phrase(saving):
    """Return the sentence fragment that states the saving of the passing designs."""
    passed_savings = [
        s["x_fewer"]
        for st, s in saving["savings"].items()
        if st != "random" and s.get("passed") and s.get("x_fewer")
    ]
    if not passed_savings:
        return "no strategy reached random's max-K cluster-τ at half the labels random itself needs"
    if max(passed_savings) - min(passed_savings) < 0.05:
        return (
            f"random's max-K cluster-τ at {min(passed_savings):.1f}× fewer labels than random "
            "needs to reach it"
        )
    return (
        f"random's max-K cluster-τ at {min(passed_savings):.1f}–{max(passed_savings):.1f}× "
        "fewer labels than random needs to reach it"
    )


def _saving_lines(saving, seeds):
    """Return the section that compares each design's reach with uniform sampling's."""
    tgt = saving["target_cluster_tau"]
    uniform_k = saving["uniform_reach_k"]
    lines = [
        "",
        "## Saving over uniform sampling",
        "",
        f"A design saves labels when it reaches random's max-K cluster-τ ({tgt:.3f}) at ≤ ½ "
        f"the labels random itself first needs to reach it (random reaches it at K={uniform_k}, "
        f"so ≤ {uniform_k // 2}).",
        "",
    ]
    for strat in STRATEGIES:
        if strat == "random":
            continue
        s = saving["savings"][strat]
        k = s["reach_k"]
        msg = "never" if k is None else f"K={k} ({s['x_fewer']:.1f}× fewer)"
        outcome = "saving" if s["passed"] else "no saving"
        lines.append(f"- {strat}: reaches {tgt:.3f} at {msg} → {outcome}")
    subject = "the passing designs reach " if saving["half_labels"] else ""
    lines += [
        "",
        f"On the competitive cluster, {subject}{_saving_phrase(saving)} "
        f"(measured at {seeds} seeds).",
        "",
        "Ranking uses the plain subset mean. The Horvitz–Thompson absolute-PR estimate (PR-MAE, "
        "CI width) is unbiased but has higher variance under active_rank's concentrated "
        "sampling; see the CI-width table.",
        "",
    ]
    return lines


def _paired_lines(paired, ks):
    """Return per-panel tables of paired margins and draw counts against uniform sampling."""
    lines = []
    for panel, label in (("cluster", "competitive cluster"), ("all8", "all 8")):
        tau = paired[panel]["kendall_tau"]
        lines += [
            "",
            f"## Paired comparison with uniform sampling ({label}; same seed)",
            "",
            "Kendall-τ margin over uniform: mean of the paired difference, then the mean over its "
            "paired standard error.",
            "",
            *_k_header("strategy", ks),
        ]
        for strat, cells in tau.items():
            row = " | ".join(
                f"{cells[k]['mean_diff']:+.4f} ({_se_cell(cells[k]['margin_in_se'])} SE)"
                for k in ks
            )
            lines.append(f"| {strat} | " + row + " |")
        for metric, block in paired[panel].items():
            lines += [
                "",
                f"Draws on which {metric} is at least as good as uniform / strictly better.",
                "",
                *_k_header("strategy", ks),
            ]
            for strat, cells in block.items():
                row = " | ".join(
                    f"{cells[k]['at_least']} / {cells[k]['strictly']} of {cells[k]['n_draws']}"
                    for k in ks
                )
                lines.append(f"| {strat} | " + row + " |")
    return lines


def _se_cell(value):
    """Format an optional margin in standard errors for a markdown cell."""
    return "—" if value is None else f"{value:.2f}"


def render_md(
    cl_names, agg_full, agg_cl, ks, full_order, full_pr, names, saving, *, loao, seeds, paired
) -> str:
    """Render the markdown report of all tables and the saving over uniform sampling."""
    ks = sorted(ks)
    lines = _setup_lines(cl_names, ks, full_order, full_pr, names, seeds)
    lines += _tau_table(agg_full, ks, "## All-8 Kendall-τ vs full ranking (mean [seed band])")
    lines += _tau_table(
        agg_cl, ks, "## Competitive-cluster Kendall-τ (selection + metric on the cluster)"
    )
    lines += _ht_lines(agg_full, agg_cl, ks)
    lines += _paired_lines(paired, ks)
    lines += _cluster_metric_lines(agg_cl, ks)
    lines += _threshold_lines(agg_cl, ks, saving)
    lines += _loao_lines(loao, ks)
    lines += _saving_lines(saving, seeds)
    return "\n".join(lines)
