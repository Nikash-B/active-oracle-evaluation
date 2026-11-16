"""CSV, Markdown and JSON outputs of the per-phase interval experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from active_oracle_evaluation.paths import output_dir, repo_relative

RUN_COMMAND = "python -m active_oracle_evaluation.experiments.phase_intervals"
CONTRAST_COLUMNS = [
    "agent",
    "phase_hi",
    "phase_lo",
    "mean_hi",
    "mean_lo",
    "diff",
    "ci_lo",
    "ci_hi",
    "p",
    "p_holm",
    "is_worst_vs_runner_up",
    "separated",
    "separated_holm",
]
EDGE_COLUMNS = [
    "phase",
    "az_mean",
    "cluster_mean",
    "abs_edge",
    "abs_ci_lo",
    "abs_ci_hi",
    "abs_p",
    "rel_edge",
    "rel_ci_lo",
    "rel_ci_hi",
]


def fmt(x: float, nd: int = 4) -> str:
    """Format a float with ``nd`` decimals, or 'nan'."""
    return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    """Write ``rows`` to ``path`` with the columns ``cols``, floats at six decimals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {c: (fmt(row[c], 6) if isinstance(row[c], float) else row[c]) for c in cols}
            )
    print(f"wrote {path}")


def _json_safe(value):
    """Replace NaN floats with None, recursively, so the JSON file is standard JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, numbers: dict) -> None:
    """Write the result numbers to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(numbers), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _header_lines(cache_path, n_dec, n_games, n_boot, boot_seed, deff) -> list[str]:
    """Return the title and method lines of the summary."""
    return [
        "# Per-phase confidence intervals",
        "",
        "Game-clustered intervals on two per-phase statements: the per-agent *within-contact* "
        "ranking, and whether AlphaZero's advantage over the competitive cluster is uniform "
        "across phases.",
        "",
        f"- Source: `{cache_path}` — {n_dec:,} scored decisions from **{n_games} source games**. "
        "No agent re-scored, no oracle call.",
        f"- Method: game-clustered percentile bootstrap, {n_boot:,} replicates, "
        f"seed {boot_seed}. Whole games are drawn with replacement; every decision inside a "
        "drawn game is kept.",
        "- All statistics share one set of replicates, so a contrast between two phases (or two "
        "agents) is a bootstrap of the *difference*, not a comparison of two marginal intervals.",
        f"- Cost of clustering: intervals on the per-phase means run **{deff['mean_ratio']:.2f}x** "
        f"wider than a decision-level bootstrap on average (range {deff['min_ratio']:.2f}-"
        f"{deff['max_ratio']:.2f} over {deff['n_cells']} agent x phase cells).",
        "",
    ]


def _contact_minus_easy_lines(rows: list[dict]) -> list[str]:
    """Return section 1 of the summary: contact phases against easy phases."""
    n_pos = sum(1 for r in rows if r["ci_lo"] > 0)
    lines = [
        "## 1. Contact phases against easy phases",
        "",
        "Contact phases (contact / holding / blitz-backgame) minus easy phases (opening / race / "
        "bearoff), per agent; each group is the unweighted mean of its three phase means.",
        "",
        "| agent | contact | easy | difference | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['agent']} | {fmt(r['contact_mean'])} | {fmt(r['easy_mean'])} | "
            f"**{fmt(r['diff'])}** | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
            f"{r['p']:.4f} |"
        )
    lines += ["", f"The interval excludes zero for **{n_pos}/{len(rows)}** agents.", ""]
    return lines


def _within_contact_lines(contrasts: list[dict], contrasts_csv: Path) -> list[str]:
    """Return section 2 of the summary: each agent's worst contact phase against its runner-up."""
    worst_pairs = [r for r in contrasts if r["is_worst_vs_runner_up"]]
    sep = [r for r in worst_pairs if r["separated"]]
    sep_holm = [r for r in worst_pairs if r["separated_holm"]]
    lines = [
        "## 2. Within-contact ranking",
        "",
        "Each agent's worst contact phase against its own runner-up. `separated` = the 95% interval "
        "on the difference excludes zero; `Holm` additionally corrects across the "
        f"{len(worst_pairs)} agents.",
        "",
        "| agent | worst | runner-up | difference | 95% CI | p | Holm p | separated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in worst_pairs:
        mark = "**yes**" if r["separated"] else "no"
        lines.append(
            f"| {r['agent']} | {r['phase_hi']} | {r['phase_lo']} | {fmt(r['diff'])} | "
            f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {r['p']:.4f} | "
            f"{r['p_holm']:.4f} | {mark} |"
        )
    surv = ", ".join(f"{r['agent']} ({r['phase_hi']})" for r in sep) or "none"
    surv_holm = ", ".join(f"{r['agent']} ({r['phase_hi']})" for r in sep_holm) or "none"
    lines += [
        "",
        f"**Separated at 95% uncorrected: {len(sep)}/{len(worst_pairs)}** — {surv}. "
        f"**After Holm correction: {len(sep_holm)}/{len(worst_pairs)}** — {surv_holm}.",
        "",
        "The tested phase is the *observed* maximum of three noisy means, so its margin over the "
        "runner-up is biased upward by the selection, which favours separation.",
        "",
        f"Full pairwise contrasts among the three contact phases are in `{contrasts_csv}`.",
        "",
    ]
    return lines


def _edge_lines(edge_rows: list[dict], uniformity: dict, cluster) -> list[str]:
    """Return section 3 of the summary: AlphaZero's edge over the cluster, by phase."""
    lines = [
        "## 3. AlphaZero's edge over the competitive cluster",
        "",
        f"Cluster = {', '.join(cluster)} (unweighted mean, recomputed per replicate). "
        "Absolute edge = cluster − AlphaZero mean regret; relative edge = the share of the "
        "cluster's error AlphaZero avoids.",
        "",
        "| phase | az | cluster | abs. edge | 95% CI | rel. edge | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in edge_rows:
        lines.append(
            f"| {r['phase']} | {fmt(r['az_mean'])} | {fmt(r['cluster_mean'])} | "
            f"{fmt(r['abs_edge'])} | [{fmt(r['abs_ci_lo'])}, {fmt(r['abs_ci_hi'])}] | "
            f"{r['rel_edge'] * 100:.0f}% | [{r['rel_ci_lo'] * 100:.0f}%, "
            f"{r['rel_ci_hi'] * 100:.0f}%] |"
        )
    ua, ur = uniformity["abs"], uniformity["rel"]
    lines += [
        "",
        "**Uniformity test** — easy-phase edge minus contact-phase edge (an interval covering "
        "zero means the edge is uniform across phases):",
        "",
        f"- absolute: {fmt(ua['easy'])} − {fmt(ua['contact'])} = **{fmt(ua['gap'])}** equity, "
        f"95% CI [{fmt(ua['ci_lo'])}, {fmt(ua['ci_hi'])}], p = {ua['p']:.4f}",
        f"- relative: {ur['easy'] * 100:.0f}% − {ur['contact'] * 100:.0f}% = "
        f"**{ur['gap'] * 100:.0f} pp**, 95% CI [{ur['ci_lo'] * 100:.0f} pp, "
        f"{ur['ci_hi'] * 100:.0f} pp], p = {ur['p']:.4f}",
        "",
    ]
    return lines


def write_doc(
    path: Path,
    *,
    cache_path: Path,
    out_dir: Path,
    n_dec: int,
    n_games: int,
    n_boot: int,
    boot_seed: int,
    contrasts: list[dict],
    primary: list[dict],
    edge_rows: list[dict],
    uniformity: dict,
    deff: dict,
    tag: str,
    cluster,
) -> None:
    """Write the Markdown summary of both tests to ``path``."""
    contrasts_csv = repo_relative(out_dir / f"phase_contrasts{tag}.csv")
    edge_csv = repo_relative(out_dir / f"phase_az_edge{tag}.csv")
    lines = _header_lines(cache_path, n_dec, n_games, n_boot, boot_seed, deff)
    lines += _contact_minus_easy_lines(primary)
    lines += _within_contact_lines(contrasts, contrasts_csv)
    lines += _edge_lines(edge_rows, uniformity, cluster)
    lines += [
        "## Outputs",
        "",
        f"- `{contrasts_csv}`, `{edge_csv}`",
        "",
        f"Reproduce: `{RUN_COMMAND} --cache {cache_path}" + (f" --tag {tag}`" if tag else "`"),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def write_outputs(args, cfg: dict, cache, results: dict) -> None:
    """Write the two CSV files, the Markdown summary and the JSON numbers."""
    tag = args.tag
    out_dir = output_dir("phase_intervals") if args.out_dir is None else args.out_dir
    write_csv(out_dir / f"phase_contrasts{tag}.csv", results["contrasts"], CONTRAST_COLUMNS)
    edge_columns = EDGE_COLUMNS + [f"rel_vs_{nm}" for nm in cfg["cluster"]]
    write_csv(out_dir / f"phase_az_edge{tag}.csv", results["edge_rows"], edge_columns)
    doc = args.doc or out_dir / f"phase_intervals{tag}.md"
    write_doc(
        doc,
        cache_path=repo_relative(args.cache),
        out_dir=out_dir,
        n_dec=cache.n_pos,
        n_games=cache.n_games,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
        contrasts=results["contrasts"],
        primary=results["primary"],
        edge_rows=results["edge_rows"],
        uniformity=results["uniformity"],
        deff=results["deff"],
        tag=tag,
        cluster=cfg["cluster"],
    )
    numbers = {
        "cache": repo_relative(args.cache),
        "n_decisions": cache.n_pos,
        "n_games": cache.n_games,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "contact_minus_easy": results["primary"],
        "contact_contrasts": results["contrasts"],
        "az_edge": results["edge_rows"],
        "uniformity": results["uniformity"],
        "design_effect": results["deff"],
    }
    write_json(out_dir / f"phase_intervals{tag}.json", numbers)
