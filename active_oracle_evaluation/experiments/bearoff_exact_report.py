"""Markdown tables of the bear-off exact-value check, with numbers formatted as in its JSON."""

import json

from active_oracle_evaluation.results import markdown_table

REFS = ("exact", "gnubg0", "gnubg2")


def fmt(value):
    """Format a number exactly as the JSON result prints it."""
    return json.dumps(value)


def oracle_tables(result):
    """Return the oracle equity-error and best-action-agreement tables."""
    err = result["oracle_equity_error_vs_exact"]
    agree = result["best_action_agreement_vs_exact"]
    columns = ("n", "mae", "bias", "p95_abs", "max_abs")
    error_rows = [[ref] + [fmt(err[ref][c]) for c in columns] for ref in err]
    agree_rows = [[ref, fmt(agree[ref])] for ref in agree]
    return [
        ("Oracle equity error against the exact value", ["reference", *columns], error_rows),
        ("Best-action agreement with the exact reference", ["reference", "agreement"], agree_rows),
    ]


def panel_tables(result):
    """Return the PR, ranking, paired-contrast and gap-distortion tables."""
    pr = result["pr"]
    names = list(pr["exact"])
    pr_rows = [[name] + [fmt(pr[ref][name]) for ref in REFS] for name in names]
    ranks = result["ranking"]
    rank_rows = [[str(r + 1)] + [ranks[ref][r] for ref in REFS] for r in range(len(names))]
    contrasts = result["paired_contrasts_dPRx500"]
    separated = set(result["exact_separated_pairs"])
    flips = set(result["sign_flips_gnubg0_on_separated"])
    contrast_rows = []
    for pair in contrasts["exact"]:
        cells = [fmt(v) for ref in REFS for v in contrasts[ref][pair]]
        contrast_rows.append([pair, *cells, str(pair in separated), str(pair in flips)])
    contrast_header = ["pair"]
    for ref in REFS:
        contrast_header += [f"{ref} dPR", f"{ref} lo", f"{ref} hi"]
    contrast_header += ["exact separates", "gnubg0 sign flip"]
    distortion = result["gap_distortion_gnubg0"]
    distortion_rows = [[pair] + [fmt(v) for v in distortion[pair]] for pair in distortion]
    return [
        ("PR by reference", ["agent", *REFS], pr_rows),
        ("Ranking by reference (lowest PR first)", ["rank", *REFS], rank_rows),
        ("Paired PR contrasts with bootstrap intervals", contrast_header, contrast_rows),
        (
            "Gap distortion of the 0-ply oracle on exact-separated pairs",
            ["pair", "exact dPR", "gnubg0 dPR", "relative distortion"],
            distortion_rows,
        ),
    ]


def write_markdown(experiment, result, path):
    """Write the result tables of the experiment to path and return the path."""
    counts = [
        ["n_decisions", fmt(result["n_decisions"])],
        ["n_forced_skipped", fmt(result["n_forced_skipped"])],
        ["seed", fmt(result["seed"])],
        ["exact_separated_pairs", fmt(len(result["exact_separated_pairs"]))],
        ["sign_flips_gnubg0_on_separated", fmt(len(result["sign_flips_gnubg0_on_separated"]))],
    ]
    tables = [("Decisions", ["quantity", "value"], counts)]
    tables += oracle_tables(result) + panel_tables(result)
    parts = [f"# {experiment}"]
    for title, header, rows in tables:
        parts += [f"## {title}", markdown_table(header, rows)]
    path.write_text("\n\n".join(parts) + "\n")
    return path
