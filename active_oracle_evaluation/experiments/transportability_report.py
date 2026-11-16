"""Printed SD-ratio summary and Markdown report of the transportability experiment."""

import numpy as np

from active_oracle_evaluation.experiments.transportability_stats import pair_group
from active_oracle_evaluation.results import markdown_table


def print_sd_ratio(rows: list[dict], s: dict) -> None:
    """Print the per-pair SD ratios and their summary."""
    print(f"\n{'pairing':<22} {'|ppg|':>7} {'sd_PR':>7} {'sd_ppg':>7} {'sd ratio':>9}  group")
    for r in sorted(rows, key=lambda r: abs(r["ppg_gap"])):
        print(
            f"  {r['pairing']:<20} {abs(r['ppg_gap']):7.3f} {r['sd_pr_trial']:7.2f} "
            f"{r['paired_sd']:7.3f} {r['sd_ratio']:9.2f}  {pair_group(r['pairing'])}"
        )
    c = s["cluster"]
    print(
        f"\nsd ratio, cluster ({c['n_pairs']} pairs): median {c['median']:.2f}, "
        f"range [{c['range'][0]:.2f}, {c['range'][1]:.2f}]"
    )
    print(
        "  closest cluster pairs: "
        + ", ".join(
            f"{r['pairing']} ({abs(r['ppg_gap']):.3f} ppg) {r['sd_ratio']:.2f}"
            for r in s["closest_cluster_pairs"]
        )
    )
    h, rd = s["heuristic_pairs"], s["random_pairs"]
    print(
        f"  heuristic pairs ({h['n_pairs']}): [{h['range'][0]:.2f}, {h['range'][1]:.2f}]   "
        f"random pairs ({rd['n_pairs']}): [{rd['range'][0]:.2f}, {rd['range'][1]:.2f}], "
        f"sd_ppg [{rd['paired_sd_range'][0]:.3f}, {rd['paired_sd_range'][1]:.3f}]"
    )
    cr = s["corr_log_sd_ratio_log_abs_ppg"]
    print(f"  corr(log sd ratio, log |ppg|): all {cr['all']:+.3f}, cluster {cr['cluster']:+.3f}")
    if "measurement_pair" in s:
        p = s["measurement_pair"]
        extra = f", 2,000-trial {p['sd_ratio_2000']:.2f}" if "sd_ratio_2000" in p else ""
        print(f"  measurement pair {p['pairing']}: {p['sd_ratio_round_robin']:.2f}{extra}")
    lo, hi = s["implied_efficiency_over_cluster_range"]
    print(
        f"  implied efficiency (gap-ratio median {s['gap_ratio_median_measurable']:.2f} "
        f"/ cluster sd-ratio median {c['median']:.2f})^2 = "
        f"{s['implied_efficiency_typical']:.2f}x, over cluster range [{lo:.2f}, {hi:.2f}]"
    )


def by_closeness(rows: list[dict]) -> list[dict]:
    """Return the rows sorted by |ppg gap|, closest pair first."""
    return sorted(rows, key=lambda r: abs(r["ppg_gap"]))


def gap_table(rows: list[dict]) -> tuple:
    """Return the per-pair ppg gap, PR gap and ratio table."""
    body = [
        [
            r["pairing"],
            f"{r['ppg_gap']:+.4f}",
            f"{r['pr_gap']:+.2f}",
            f"{r['pr_gap'] / r['ppg_gap']:.2f}",
        ]
        for r in by_closeness(rows)
    ]
    return ("PR gap against ppg gap", ["pairing", "ppg gap", "PR gap", "PR/ppg"], body)


def vr_table(rows: list[dict], vr: dict) -> tuple:
    """Return the variance-reduction table, closest PR gap first."""
    body = [
        [rows[i]["pairing"], f"{vr['gap'][i]:.2f}", f"{vr['vrf'][i]:.2f}x"] for i in vr["order"]
    ]
    return ("Variance reduction against closeness", ["pairing", "PR gap", "VR"], body)


def sd_ratio_table(rows: list[dict]) -> tuple:
    """Return the per-pair trial-level SD ratio table."""
    body = [
        [
            r["pairing"],
            f"{abs(r['ppg_gap']):.3f}",
            f"{r['sd_pr_trial']:.2f}",
            f"{r['paired_sd']:.3f}",
            f"{r['sd_ratio']:.2f}",
            pair_group(r["pairing"]),
        ]
        for r in by_closeness(rows)
    ]
    header = ["pairing", "|ppg|", "sd_PR", "sd_ppg", "sd ratio", "group"]
    return ("Trial-level SD ratio", header, body)


def summary_table(rows: list[dict], vr: dict, prop: dict, s: dict) -> tuple:
    """Return the summary numbers, formatted as printed."""
    ratio, iqr_meas, c = prop["ratio"], prop["iqr_meas"], s["cluster"]
    cr = s["corr_log_sd_ratio_log_abs_ppg"]
    lo, hi = s["implied_efficiency_over_cluster_range"]
    body = [
        ["ratio PR-gap / ppg-gap, median", f"{np.median(ratio):.2f}"],
        [
            "ratio PR-gap / ppg-gap, IQR",
            f"[{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}]",
        ],
        ["ratio PR-gap / ppg-gap, range", f"[{ratio.min():.2f}, {ratio.max():.2f}]"],
        ["measurable pairs", f"{prop['measurable'].sum()}"],
        ["ratio over measurable pairs, median", f"{np.median(prop['ratio_meas']):.2f}"],
        ["ratio over measurable pairs, IQR", f"[{iqr_meas[0]:.2f}, {iqr_meas[1]:.2f}]"],
        ["log-log slope of |PR gap| on |ppg gap|", f"{prop['slope']:.3f}"],
        ["log-log r", f"{prop['corr']:.3f}"],
        [f"VR, closest {vr['k']}, median", f"{np.median(vr['close']):.2f}x"],
        [f"VR, farthest {vr['k']}, median", f"{np.median(vr['far']):.2f}x"],
        [f"VR, all {len(rows)}, median", f"{np.median(vr['vrf']):.2f}x"],
        ["correlation of VR with log(PR gap)", f"{vr['rho']:+.3f}"],
        ["VR, measurement pair", f"{vr['pair_vr']:.2f}x"],
        [f"sd ratio, cluster ({c['n_pairs']} pairs), median", f"{c['median']:.2f}"],
        ["sd ratio, cluster, range", f"[{c['range'][0]:.2f}, {c['range'][1]:.2f}]"],
        ["corr(log sd ratio, log |ppg|), all", f"{cr['all']:+.3f}"],
        ["corr(log sd ratio, log |ppg|), cluster", f"{cr['cluster']:+.3f}"],
        ["implied efficiency, typical", f"{s['implied_efficiency_typical']:.2f}x"],
        ["implied efficiency, over cluster range", f"[{lo:.2f}, {hi:.2f}]"],
    ]
    if "measurement_pair" in s:
        p = s["measurement_pair"]
        body.append([f"sd ratio, {p['pairing']}, round-robin", f"{p['sd_ratio_round_robin']:.2f}"])
        if "sd_ratio_2000" in p:
            body.append([f"sd ratio, {p['pairing']}, 2,000-trial", f"{p['sd_ratio_2000']:.2f}"])
    return ("Summary", ["quantity", "value"], body)


def write_markdown(experiment, rows, vr, prop, s, out_dir):
    """Write <experiment>.md with the result tables; return its path."""
    tables = [summary_table(rows, vr, prop, s), gap_table(rows), vr_table(rows, vr)]
    tables.append(sd_ratio_table(rows))
    parts = [f"# {experiment}"]
    for title, header, body in tables:
        parts += [f"## {title}", markdown_table(header, body)]
    md_path = out_dir / f"{experiment}.md"
    md_path.write_text("\n\n".join(parts) + "\n")
    return md_path
