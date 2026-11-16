"""Markdown report of the cyclic-triple analysis: every number as it appears in the JSON."""

import json

from active_oracle_evaluation.results import markdown_table

AGENTS = ["cyc0", "cyc1", "cyc2"]


def _fmt(value):
    """Format one value as it appears in the JSON output."""
    return json.dumps(value)


def _result_tables(result):
    """Return the (title, header, rows) tables of the analysis result."""
    edges = result["cycle_edges"]
    edge_keys = ("ppg", "se", "z", "winrate", "wins", "games")
    misfit_keys = ("observed", "bt_predicted", "excess_sigmas")
    rating_keys = ("bt_elo", "maxent_nash_mixture", "nash_skill_winrate_margin")
    summary = [
        ["cycle_certified", _fmt(result["cycle_certified"])],
        ["nash_feasibility_violation", _fmt(result["nash_feasibility_violation"])],
    ]
    tables = [
        ("Summary", ["quantity", "value"], summary),
        (
            "Cycle edges",
            ["edge", *edge_keys],
            [[k] + [_fmt(v[x]) for x in edge_keys] for k, v in edges.items()],
        ),
        (
            "Ratings",
            ["agent", *rating_keys],
            [[a] + [_fmt(result[x][a]) for x in rating_keys] for a in AGENTS],
        ),
        (
            "Bradley-Terry misfit",
            ["edge", *misfit_keys],
            [[k] + [_fmt(v[x]) for x in misfit_keys] for k, v in result["bt_misfit"].items()],
        ),
    ]
    for title, block, key in [
        ("Alpha-rank on the cyclic triple", result["alpha_rank_cyclic"], "stationary_by_alpha"),
        ("Alpha-rank on the transitive panel", result["alpha_rank_transitive_panel"], "stationary"),
    ]:
        rows = [[alpha] + [_fmt(x) for x in pi] for alpha, pi in block[key].items()]
        tables.append((title, ["alpha", *block["agents"]], rows))
    pr = result["suite_pr"]
    if pr is not None:
        pr_rows = [
            [k, _fmt(v["pr"]), _fmt(v["ci"][0]), _fmt(v["ci"][1])] for k, v in pr["pr"].items()
        ]
        tables.append(("Suite Performance Rating", ["agent", "pr", "ci_low", "ci_high"], pr_rows))
        contrast_rows = [
            [
                k,
                _fmt(v["delta_pr"]),
                _fmt(v["ci95"][0]),
                _fmt(v["ci95"][1]),
                _fmt(v["distinguishable"]),
            ]
            for k, v in pr["cyc_contrasts"].items()
        ]
        header = ["pair", "delta_pr", "ci_low", "ci_high", "distinguishable"]
        tables.append(("Suite PR contrasts", header, contrast_rows))
    return tables


def write_markdown(experiment, result, path):
    """Write the result tables as a Markdown file."""
    parts = [f"# {experiment}"]
    for title, header, rows in _result_tables(result):
        parts += [f"## {title}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n")
