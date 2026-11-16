"""Per-agent coverage of the analytic 95% intervals of the estimator baselines.

Every interval of the estimator_baselines experiment is ``estimate ± 1.96 · SE`` with an analytic
SE: simple random sampling with the finite-population correction for classical and PPI, the
independent-draw approximation ``(1/N²)·Σ(1−π)/π²·y²`` for Horvitz--Thompson and Hájek under a
systematic PPS draw, and LURE's weighted-term formula.

For each panel (all agents, and the competitive cluster of six agents without the random player
and the heuristic), estimator and budget K this reports:

* ``coverage`` -- pooled over agents and draws, as the estimator table computes it, with a
  draw-clustered standard error (all agents share each draw, so agents are not independent);
* ``coverage_per_agent``;
* ``sd_over_mean_se`` per agent -- the realized SD of the estimate across draws divided by the mean
  analytic SE. Below 1 the formula overstates the spread; above 1 it understates it.

The pooled coverage at the first draw count is checked against the coverage column of the
estimator_baselines CSVs; write them first with

    python -m active_oracle_evaluation.experiments.estimator_baselines --seeds 200
    python -m active_oracle_evaluation.experiments.estimator_baselines --seeds 200 \\
        --agents pubeval v1 v2 ppo az distilled \\
        --out-dir outputs/estimator_baselines/six_agent

then run, with the paper's settings (200 and 1000 draws per budget),

    python -m active_oracle_evaluation.experiments.estimator_interval_coverage

The results go to outputs/estimator_interval_coverage/estimator_interval_coverage.{md,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.score_cache import ScoreCache
from active_oracle_evaluation.paths import DATA, OUTPUTS, output_dir
from active_oracle_evaluation.results import markdown_table

EXPERIMENT = "estimator_interval_coverage"
COMMAND = f"python -m active_oracle_evaluation.experiments.{EXPERIMENT}"
CHEAP = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
EXPENSIVE = DATA / "decision_suites" / "full_suite_scores_2ply.npz"
BASELINE_CSVS = {
    "all_agents": OUTPUTS / "estimator_baselines" / "estimator_baselines.csv",
    "six_agents": OUTPUTS / "estimator_baselines" / "six_agent" / "estimator_baselines.csv",
}
BASELINE_COMMANDS = {
    "all_agents": "python -m active_oracle_evaluation.experiments.estimator_baselines --seeds 200",
    "six_agents": "python -m active_oracle_evaluation.experiments.estimator_baselines "
    "--seeds 200 --agents pubeval v1 v2 ppo az distilled "
    "--out-dir outputs/estimator_baselines/six_agent",
}


def calibrate(run, cheap, exp, names, seeds, cfg):
    """Coverage and SD/SE ratio per (K, estimator) for one panel and draw count."""
    raw: dict = {}
    run(cheap, exp, cfg["ks"], seeds, cfg["eps"], names=names, raw=raw)
    out = {}
    for k, r in raw.items():
        out[k] = {}
        for e in r["est"]:
            est, se, truth = r["est"][e], r["se"][e], r["truth"]
            half = cfg["z_95"] * se
            covered = np.abs(est - truth[None, :]) <= half
            finite = np.isfinite(half)
            per_draw = np.array(
                [covered[s][finite[s]].mean() for s in range(est.shape[0]) if finite[s].any()]
            )
            sd = est.std(axis=0, ddof=1)
            mean_se = np.nanmean(se, axis=0)
            clustered_se = (
                float(per_draw.std(ddof=1) / np.sqrt(len(per_draw))) if len(per_draw) > 1 else None
            )
            out[k][e] = {
                "coverage": float(np.nanmean(covered[finite])),
                "coverage_draw_clustered_se": clustered_se,
                "coverage_per_agent": {
                    n: float(covered[:, a][finite[:, a]].mean()) for a, n in enumerate(r["names"])
                },
                "sd_over_mean_se": {
                    n: float(sd[a] / mean_se[a]) if mean_se[a] > 0 else None
                    for a, n in enumerate(r["names"])
                },
            }
    return out


def baseline_coverage(path: Path) -> dict:
    """Read the coverage column of an estimator-baseline CSV, keyed by (estimator, K)."""
    return {
        (row["estimator"], int(row["k"])): float(row["coverage"])
        for row in csv.DictReader(path.open())
        if row["coverage"] not in ("", "nan")
    }


def build_parser(cfg) -> argparse.ArgumentParser:
    """Command-line options, with defaults from the settings file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=cfg["seeds"],
        help="draw counts; the first must equal the baseline CSVs' (200) for the control",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=f"JSON output (default outputs/{EXPERIMENT}/{EXPERIMENT}.json); "
        "the Markdown report is written beside it",
    )
    return parser


def summary_rows(result, panels, seed_counts, cfg):
    """Rows of the printed summary at the report budget, formatted as printed."""
    report_k = cfg["report_k"]
    rows = []
    for pname in panels:
        for seeds in seed_counts:
            cells = result["panels"][pname][str(seeds)][report_k]
            for e in cfg["report_estimators"]:
                r = cells[e]
                ratios = [x for x in r["sd_over_mean_se"].values() if x is not None]
                worst = min(r["coverage_per_agent"].items(), key=lambda kv: kv[1])
                rows.append(
                    {
                        "panel": pname,
                        "draws": seeds,
                        "estimator": e,
                        "coverage": f"{r['coverage']:.3f}",
                        "se": f"{r['coverage_draw_clustered_se']:.3f}",
                        "sd_se_min": f"{min(ratios):.2f}",
                        "sd_se_max": f"{max(ratios):.2f}",
                        "worst": worst,
                    }
                )
    return rows


def write_markdown(path, result, rows, report_k):
    """Write the control and summary tables as Markdown."""
    control_rows = [[name, f"{v:.2e}"] for name, v in result["controls"].items()]
    summary = [
        [
            r["panel"],
            str(r["draws"]),
            str(report_k),
            r["estimator"],
            r["coverage"],
            r["se"],
            r["sd_se_min"],
            r["sd_se_max"],
            r["worst"][0],
            repr(r["worst"][1]),
        ]
        for r in rows
    ]
    header = ["panel", "draws", "K", "estimator", "coverage", "clustered SE"]
    header += ["SD/SE min", "SD/SE max", "worst agent", "worst-agent coverage"]
    parts = [
        f"# {EXPERIMENT}",
        "## Control: max abs coverage difference from the estimator_baselines CSVs",
        markdown_table(["control", "max abs diff"], control_rows),
        f"## Pooled coverage of the 95% intervals at K={report_k}",
        markdown_table(header, summary),
    ]
    path.write_text("\n\n".join(parts) + "\n")


def main() -> int:
    """Run the coverage check for both panels, compare with the baselines and print a summary."""
    cfg = settings(EXPERIMENT)
    args = build_parser(cfg).parse_args()
    for pname, path in BASELINE_CSVS.items():
        if not path.exists():
            raise SystemExit(f"missing {path}; run: {BASELINE_COMMANDS[pname]}")
    if args.json is None:
        args.json = output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"

    from active_oracle_evaluation.experiments.estimator_baselines import run

    cheap = ScoreCache(CHEAP)
    exp = ScoreCache(EXPENSIVE)
    panels = {
        "all_agents": list(cheap.names),
        "six_agents": [a for a in cheap.names if a in cfg["cluster_agents"]],
    }

    result = {
        "tool": COMMAND,
        "ks": cfg["ks"],
        "panels": {},
        "controls": {},
    }
    for pname, names in panels.items():
        result["panels"][pname] = {"agents": names}
        for seeds in args.seeds:
            cal = calibrate(run, cheap, exp, names, seeds, cfg)
            result["panels"][pname][str(seeds)] = cal
            print(f"{pname}, {seeds} draws: done", flush=True)
        want = baseline_coverage(BASELINE_CSVS[pname])
        got = result["panels"][pname][str(args.seeds[0])]
        diff = max(
            abs(got[k][e]["coverage"] - v) for (e, k), v in want.items() if k in got and e in got[k]
        )
        result["controls"][f"{pname}_reproduces_baseline_coverage_max_abs_diff"] = diff

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.json}")
    for name, v in result["controls"].items():
        print(f"control {name}: {v:.2e}")
    rows = summary_rows(result, panels, args.seeds, cfg)
    for r in rows:
        print(
            f"{r['panel']:10s} {r['draws']:5d} draws K={cfg['report_k']} {r['estimator']:12s} "
            f"coverage {r['coverage']} (±{r['se']}); "
            f"SD/SE {r['sd_se_min']}..{r['sd_se_max']}; worst agent {r['worst']}"
        )
    md_path = args.json.with_suffix(".md")
    write_markdown(md_path, result, rows, cfg["report_k"])
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
