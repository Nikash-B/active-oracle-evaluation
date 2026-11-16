"""Result tables of the tail_regret experiment, for tail_regret.md and tail_regret.json."""

from __future__ import annotations

from active_oracle_evaluation.eval.score_cache import ScoreCache
from active_oracle_evaluation.eval.tail_risk import phase_breakdown, tail_stats

TEXT_COLUMNS = ("agent", "phase", "cube_epc")


def similar_mean_pair(cache: ScoreCache, tol: float) -> tuple[str, str] | None:
    """Return the agent pair with mean regret within ``tol`` whose p99 differs most, or None."""
    names = cache.names
    stats = {name: tail_stats(cache.losses[name]) for name in names}
    best = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if abs(stats[a]["mean"] - stats[b]["mean"]) > tol:
                continue
            p99_gap = abs(stats[a]["p99"] - stats[b]["p99"])
            if best is None or p99_gap > best[0]:
                best = (p99_gap, a, b)
    return None if best is None else (best[1], best[2])


def worst_phases(cache: ScoreCache) -> dict[str, str]:
    """Return each agent's phase of highest mean regret (a point estimate)."""
    worst = {}
    for name in cache.names:
        breakdown = phase_breakdown(cache.losses[name], cache.phases)
        worst[name] = max(breakdown, key=lambda ph: breakdown[ph]["mean"]) if breakdown else "-"
    return worst


def numeric_row(row: dict) -> dict:
    """Return a CSV row with its formatted numbers parsed back to floats, without cube_epc."""
    out = {}
    for key, value in row.items():
        if key == "cube_epc":
            continue
        out[key] = value if key in TEXT_COLUMNS or isinstance(value, int) else float(value)
    return out


def pair_table(cache: ScoreCache, tol: float) -> tuple[list[list[str]], dict | None]:
    """Return the table rows and JSON entry of the similar-mean pair with the largest p99 gap."""
    pair = similar_mean_pair(cache, tol)
    if pair is None:
        return [], None
    a, b = pair
    p99_a = tail_stats(cache.losses[a])["p99"]
    p99_b = tail_stats(cache.losses[b])["p99"]
    row = [
        a,
        b,
        f"{cache.pr[a]:.1f}",
        f"{cache.pr[b]:.1f}",
        f"{abs(cache.pr[a] - cache.pr[b]):.1f}",
        f"{p99_a * 1000:.0f}",
        f"{p99_b * 1000:.0f}",
    ]
    entry = {"first": a, "second": b, "pr": [cache.pr[a], cache.pr[b]], "p99": [p99_a, p99_b]}
    return [row], entry


def report_tables(
    cache: ScoreCache, tail_rows: list[dict], phase_rows: list[dict], cat_cols: list, tol: float
) -> tuple[list, dict]:
    """Return the (title, header, rows) tables and the JSON numbers of the experiment."""
    tail_header = ["agent", "PR", "mean", "median", "p90", "p95", "p99", "p95 95% interval"]
    tail_header += [col.replace("cat_gt_", "cat>") for col in cat_cols]
    tail_table = [
        [r["agent"], r["pr"], r["mean"], r["median"], r["p90"], r["p95"], r["p99"]]
        + [f"[{r['p95_ci_lo']}, {r['p95_ci_hi']}]"]
        + [r[col] for col in cat_cols]
        for r in tail_rows
    ]
    phase_header = ["agent", "phase", "n", "games", "mean", "mean 95% interval", "p95"]
    phase_table = [
        [r["agent"], r["phase"], r["n"], r["n_games"], r["mean"]]
        + [f"[{r['mean_ci_lo']}, {r['mean_ci_hi']}]", r["p95"]]
        for r in phase_rows
    ]
    worst = worst_phases(cache)
    pair_rows, pair_entry = pair_table(cache, tol)
    pair_header = ["first", "second", "PR first", "PR second", "PR gap", "p99 first (mpts)"]
    pair_header += ["p99 second (mpts)"]
    tables = [
        ("Tail regret by agent (equity, sorted by PR)", tail_header, tail_table),
        ("Mean and p95 regret by phase (equity)", phase_header, phase_table),
        (
            "Phase of highest mean regret (point estimate)",
            ["agent", "phase"],
            [[name, worst[name]] for name in cache.names],
        ),
        (
            f"Similar-mean pair (mean gap at most {tol:g} equity) with the largest p99 gap",
            pair_header,
            pair_rows,
        ),
    ]
    numbers = {
        "n_decisions": cache.n_pos,
        "n_games": cache.n_games,
        "tail_regret": [numeric_row(r) for r in tail_rows],
        "phase_regret": [numeric_row(r) for r in phase_rows],
        "worst_phase": worst,
        "similar_mean_pair": pair_entry,
    }
    return tables, numbers
