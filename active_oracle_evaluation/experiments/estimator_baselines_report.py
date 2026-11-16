"""Result tables, CSV and JSON summary of the estimator baselines."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from active_oracle_evaluation.experiments.estimator_baselines_stats import (
    ESTIMATORS,
    labels_to_match,
)
from active_oracle_evaluation.results import write_results

EXPERIMENT = "estimator_baselines"
CSV_FIELDS = ["estimator", "k", "bias", "sd", "rmse", "ci_width", "coverage", "tau", "top1"]


def _by_k_rows(rows: list[dict], estimators, ks: list[int], cell) -> list[list[str]]:
    """Table rows: one per estimator, one cell per K from ``cell(row)``, a dash when missing."""
    out = []
    for e in estimators:
        cells = []
        for k in ks:
            match = [x for x in rows if x["estimator"] == e and x["k"] == k]
            cells.append(cell(match[0]) if match else "—")
        out.append([f"`{e}`"] + cells)
    return out


def _report_tables(rows, match, *, target, baseline, n_seeds, n_pool, names, ks, corr_range):
    """The (title, header, rows) result tables of the report."""
    by_k = ["estimator"] + [f"K={k}" for k in ks]
    setup = [
        ["pool decisions", f"{n_pool:,}"],
        ["agents", ", ".join(names)],
        ["draws per (estimator, K)", str(n_seeds)],
        ["0-ply vs 2-ply regret correlation, trained agents (min)", f"{corr_range[0]:.2f}"],
        ["0-ply vs 2-ply regret correlation, trained agents (max)", f"{corr_range[1]:.2f}"],
    ]
    saving = []
    for e in ESTIMATORS:
        v = match.get(e, float("nan"))
        if np.isfinite(v):
            saving.append([f"`{e}`", f"{v:,.0f}", f"{target / v:.1f}×"])
        else:
            saving.append([f"`{e}`", "not reached within the measured range", "—"])
    top1 = _by_k_rows(
        rows,
        (*ESTIMATORS, "bai_halving"),
        ks,
        lambda r: f"{r['top1']:.0%}" if np.isfinite(r["top1"]) else "—",
    )
    return [
        ("Setup", ["quantity", "value"], setup),
        (
            "RMSE in PR points (lower is better)",
            by_k,
            _by_k_rows(rows, ESTIMATORS, ks, lambda r: f"{r['rmse']:.2f}"),
        ),
        (
            f"Labels needed to match `{baseline}` at K={target}",
            ["estimator", "labels", "saving"],
            saving,
        ),
        (
            "Interval width in PR points and coverage (95%)",
            by_k,
            _by_k_rows(
                rows, ESTIMATORS, ks, lambda r: f"{r['ci_width']:.2f} ({r['coverage']:.0%})"
            ),
        ),
        ("Best-agent identification, P(top-1 correct)", ["method"] + by_k[1:], top1),
        (
            "Bias in PR points",
            by_k,
            _by_k_rows(rows, ESTIMATORS, ks, lambda r: f"{r['bias']:+.3f}"),
        ),
    ]


def _finite_or_none(value):
    """Return ``value``, with a non-finite float replaced by None for JSON."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(
    rows: list[dict],
    out_dir: Path,
    *,
    n_seeds: int,
    n_pool: int,
    names: list[str],
    ks: list[int],
    corr_range: tuple[float, float],
    baseline: str,
) -> None:
    """Write the per-(estimator, K) CSV, the Markdown tables and the JSON summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "estimator_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {csv_path}")

    target = max(ks)
    match = labels_to_match(rows, baseline, target)
    tables = _report_tables(
        rows,
        match,
        target=target,
        baseline=baseline,
        n_seeds=n_seeds,
        n_pool=n_pool,
        names=names,
        ks=ks,
        corr_range=corr_range,
    )
    numbers = {
        "pool_decisions": n_pool,
        "agents": list(names),
        "draws": n_seeds,
        "ks": list(ks),
        "predictor_correlation_range": [_finite_or_none(c) for c in corr_range],
        "rows": [{key: _finite_or_none(v) for key, v in r.items()} for r in rows],
        "labels_to_match": {
            "baseline": baseline,
            "target_k": target,
            "labels": {e: _finite_or_none(match.get(e, float("nan"))) for e in ESTIMATORS},
        },
    }
    md_path, json_path = write_results(EXPERIMENT, tables, numbers, out_dir=out_dir)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
