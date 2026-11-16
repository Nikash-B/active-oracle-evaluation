"""Print and write the tables of the label-efficiency robustness checks."""

from __future__ import annotations

from active_oracle_evaluation.results import markdown_table

SCOPES = ("all_agents", "cluster_agents")


def _recovery_cell(stats):
    """Format one exact-recovery cell as 'frac_exact (mean rank displacement)'."""
    return f"{stats['frac_exact']:.3f} ({stats['mean_rank_disp']:.2f})"


def _paired_cell(cell):
    """Format one paired-difference cell as 'mean (paired SE)'."""
    return f"{cell['mean_diff']:+.3f} ({cell['paired_se']:.3f})"


def _threshold_title(result):
    """Describe the uniform target of one efficiency CSV."""
    return (
        f"target {result['target_uniform_tau_at_k_max']:.4f}, "
        f"uniform first reaches it at K={result['uniform_first_reaches_target_at']}"
    )


def print_summary(out, control, ks):
    """Print the control, the recovery tables, the paired differences and the thresholds."""
    print(f"LOAO control vs label_efficiency CSV: max |diff| {control:.2e}")
    for scope in SCOPES:
        print(f"\n{scope}: frac_exact (mean rank displacement)")
        for design, stats in out["loao"][scope].items():
            print(f"  {design:18s} " + "  ".join(f"K{k}: {_recovery_cell(stats[k])}" for k in ks))
    for scope in SCOPES:
        print(f"\npaired differences in frac_exact, {scope}: mean (paired SE)")
        for pair, cells in out["loao"]["paired_differences"][scope].items():
            print(
                f"  {pair:38s} " + "  ".join(f"K{k}: {_paired_cell(v)}" for k, v in cells.items())
            )
    for name, result in out["threshold_logic"].items():
        print(f"\n{name}: {_threshold_title(result)}")
        for strategy, v in result["strategies"].items():
            print(
                f"  {strategy:15s} reach {v['reach_k']}  "
                f"saving at K_max {v['saving_vs_uniform_budget_at_k_max']}  "
                f"vs uniform's own reach {v['saving_vs_uniform_own_reach']}"
            )


def write_markdown(experiment, out, control, ks, path):
    """Write the printed tables as Markdown to ``path`` and return it."""
    k_header = [f"K{k}" for k in ks]
    parts = [
        f"# {experiment}",
        "## LOAO control",
        markdown_table(
            ["quantity", "value"], [["max abs diff vs label_efficiency CSV", f"{control:.2e}"]]
        ),
    ]
    for scope in SCOPES:
        rows = [
            [design] + [_recovery_cell(stats[k]) for k in ks]
            for design, stats in out["loao"][scope].items()
        ]
        parts += [
            f"## {scope}: frac_exact (mean rank displacement)",
            markdown_table(["design"] + k_header, rows),
        ]
    for scope in SCOPES:
        cells_by_pair = out["loao"]["paired_differences"][scope]
        rows = [
            [pair] + [_paired_cell(v) for v in cells.values()]
            for pair, cells in cells_by_pair.items()
        ]
        parts += [
            f"## paired differences in frac_exact, {scope}: mean (paired SE)",
            markdown_table(["pair"] + k_header, rows),
        ]
    for name, result in out["threshold_logic"].items():
        rows = [
            [
                strategy,
                str(v["reach_k"]),
                str(v["saving_vs_uniform_budget_at_k_max"]),
                str(v["saving_vs_uniform_own_reach"]),
            ]
            for strategy, v in result["strategies"].items()
        ]
        header = ["strategy", "reach K", "saving at K_max", "vs uniform's own reach"]
        parts += [f"## {name}: {_threshold_title(result)}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n")
    return path
