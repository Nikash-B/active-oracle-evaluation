"""Summaries of the alpha_rank_clones cells: the printed report and the Markdown tables."""

from __future__ import annotations

from active_oracle_evaluation.results import markdown_table


def panel_summary(panel, alphas):
    """Return the Elo, Nash averaging and per-alpha alpha-rank maxima of one panel."""
    cells = panel["cells"]
    summary = {
        "n_agents": len(panel["agents"]),
        "per_pairing": panel["games_per_pairing"],
        "n_cells": len(cells),
        "elo_shift": max(c["elo"]["max_shift_others"] for c in cells),
        "elo_spread": max(c["elo"]["spread_others"] for c in cells),
        "elo_centred": max(c["elo"]["max_shift_centred_all"] for c in cells),
        "n_reordered": sum(c["elo"]["order_changed"] for c in cells),
        "nash_shift": max(c["nash_averaging"]["max_rating_shift_others"] for c in cells),
        "nash_mass": max(c["nash_averaging"]["merged_mass_shift"] for c in cells),
        "alpha_rank": [],
    }
    for alpha in alphas:
        rows = [c["alpha_rank"][str(alpha)] for c in cells]
        resolved = [r for r in rows if not r["numerically_unresolved"]]
        entry = {"alpha": alpha, "n_rows": len(rows), "n_resolved": len(resolved)}
        if resolved:
            entry["mass_shift"] = max(r["others_max_shift"] for r in resolved)
            entry["merged_tv"] = max(r["merged_tv"] for r in resolved)
            entry["n_reordered"] = sum(r["order_changed"] for r in resolved)
        summary["alpha_rank"].append(entry)
    return summary


def print_summary(controls, panels, alphas):
    """Print the controls and each panel's Elo, Nash averaging and alpha-rank summary."""
    print("controls:")
    for key, value in controls.items():
        print(f"  {key}: {value}")
    for pname, panel in panels.items():
        s = panel_summary(panel, alphas)
        print(f"\n{pname} ({s['n_agents']} agents, {s['per_pairing']:.0f} games/pairing)")
        print(
            f"  Elo            max anchored shift among others {s['elo_shift']:8.2f}, "
            f"spread {s['elo_spread']:8.2f}, "
            f"centred (old) {s['elo_centred']:8.2f} Elo; "
            f"originals reordered in {s['n_reordered']}/{s['n_cells']} cells"
        )
        print(
            f"  Nash averaging max rating shift {s['nash_shift']:.2e}, "
            f"merged-mass shift {s['nash_mass']:.2e}"
        )
        for entry in s["alpha_rank"]:
            alpha = entry["alpha"]
            if not entry["n_resolved"]:
                print(
                    f"  alpha-rank a={alpha:<6} all {entry['n_rows']} cells numerically unresolved"
                )
                continue
            n_unresolved = entry["n_rows"] - entry["n_resolved"]
            suffix = "" if n_unresolved == 0 else f"  ({n_unresolved} unresolved)"
            print(
                f"  alpha-rank a={alpha:<6} "
                f"others' mass shift max {entry['mass_shift']:.4f}  "
                f"merged TV max {entry['merged_tv']:.4f}  "
                f"reordered {entry['n_reordered']}/{entry['n_resolved']}"
                f"{suffix}"
            )


def markdown_tables(controls, panels, alphas):
    """Return the controls table and one Elo/Nash and one alpha-rank table per panel."""
    tables = [("Controls", ["Control", "Value"], [[k, str(v)] for k, v in controls.items()])]
    for pname, panel in panels.items():
        s = panel_summary(panel, alphas)
        title = f"{pname} ({s['n_agents']} agents, {s['per_pairing']:.0f} games/pairing)"
        header = ["Rating", "Max shift among others", "Spread", "Centred", "Merged-mass shift"]
        header.append("Reordered cells")
        rows = [
            [
                "Elo",
                f"{s['elo_shift']:.2f}",
                f"{s['elo_spread']:.2f}",
                f"{s['elo_centred']:.2f}",
                "–",
                f"{s['n_reordered']}/{s['n_cells']}",
            ],
            ["Nash averaging", f"{s['nash_shift']:.2e}", "–", "–", f"{s['nash_mass']:.2e}", "–"],
        ]
        tables.append((f"{title}: Elo and Nash averaging", header, rows))
        header = ["Alpha", "Others' mass shift max", "Merged TV max", "Reordered", "Unresolved"]
        rows = []
        for entry in s["alpha_rank"]:
            n_unresolved = entry["n_rows"] - entry["n_resolved"]
            if not entry["n_resolved"]:
                rows.append([f"{entry['alpha']}", "–", "–", "–", str(n_unresolved)])
                continue
            rows.append(
                [
                    f"{entry['alpha']}",
                    f"{entry['mass_shift']:.4f}",
                    f"{entry['merged_tv']:.4f}",
                    f"{entry['n_reordered']}/{entry['n_resolved']}",
                    str(n_unresolved),
                ]
            )
        tables.append((f"{title}: alpha-rank", header, rows))
    return tables


def write_markdown(path, experiment, controls, panels, alphas):
    """Write the titled summary tables of the experiment to ``path``."""
    parts = [f"# {experiment}"]
    for title, header, rows in markdown_tables(controls, panels, alphas):
        parts += [f"## {title}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n")
