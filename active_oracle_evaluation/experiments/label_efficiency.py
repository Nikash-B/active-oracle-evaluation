"""Label efficiency of the four selection designs.

For each selection strategy, budget K and seed we pick K positions, rank the agents by their mean
oracle regret on just those K, and compare with the full-suite ranking: Kendall-τ, pairwise
accuracy, tight-pair accuracy, top-1 recovery, PR-MAE and CI width. This runs on the full
eight-agent panel and on the six-agent competitive cluster, together with a leave-one-agent-out
check of disagreement selection. Each design is also compared with uniform sampling draw by
draw (same seed): draw counts and the paired margin in standard errors.

Every metric sub-indexes the cached ``losses[n_agents, n_pos]`` matrix (PR on a subset ``S`` is
``losses[:, S].mean(axis=1) * 500``), so no agent is re-scored here. Selection comes from
``active_oracle_evaluation.eval.eval_select``.

The 0-ply run (200 draws per budget) writes label_efficiency_0ply.csv and
label_efficiency_0ply_loao.csv:

    python -m active_oracle_evaluation.experiments.label_efficiency --seeds 200 \
        --out-csv outputs/label_efficiency/label_efficiency_0ply.csv

The 2-ply run (20 draws per budget) writes label_efficiency_2ply.csv:

    python -m active_oracle_evaluation.experiments.label_efficiency \
        --cache data/decision_suites/full_suite_scores_2ply.npz --seeds 20 \
        --out-csv outputs/label_efficiency/label_efficiency_2ply.csv \
        --out-md outputs/label_efficiency/label_efficiency_2ply.md

Each run also writes an HT-ranked companion CSV, a Markdown report (default
label_efficiency.md) and a JSON summary next to the report.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.eval import rank_recovery as rr
from active_oracle_evaluation.experiments.label_efficiency_report import render_md
from active_oracle_evaluation.experiments.label_efficiency_stats import (
    CLUSTER,
    DESIGN_SOURCE,
    EXPERIMENT,
    METRIC_COLS,
    PAIRED_METRICS,
    SETTINGS,
    STRATEGIES,
    TIGHT_PAIRS,
    aggregate,
    aggregate_loao,
    paired_vs_uniform,
    saving_vs_uniform,
)
from active_oracle_evaluation.paths import DATA, OUTPUTS

DEFAULT_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
# Plain columns beside their Horvitz-Thompson twins in the HT-ranked companion CSV.
HT_PAIRS = (
    ("kendall_tau", "kendall_tau_ht"),
    ("tight_pair_acc", "tight_pair_acc_ht"),
    ("top1", "top1_ht"),
)


def load_select():
    """Return ``(select_fn, strategies)`` from ``active_oracle_evaluation.eval.eval_select``.

    ``select_fn(strategy, k, *, losses, choices, phases, seed)`` returns an object with
    ``.indices`` (into pool order) and ``.inclusion_probs`` (π aligned to indices). ``active_rank``
    steers on the losses it then estimates (the skyline).
    """
    from active_oracle_evaluation.eval.eval_select import STRATEGIES as SELECT_STRATEGIES
    from active_oracle_evaluation.eval.eval_select import select

    return functools.partial(select, importance="skyline"), tuple(SELECT_STRATEGIES)


def load_cache(path: str | Path) -> dict:
    """Load the frozen full-suite score cache (.npz) into a dict the selectors/metrics consume."""
    d = np.load(path, allow_pickle=True)
    return {
        "names": [str(x) for x in d["names"]],
        "phases": d["phases"],
        "losses": d["losses"].astype(float),
        "choices": d["choices"],
        "best_equity": d["best_equity"],
        "n_actions": d["n_actions"],
        "n_pos": int(d["losses"].shape[1]),
    }


def filter_cache(cache: dict, agents: list[str]) -> dict:
    """Restrict the panel to ``agents`` (subset the agent rows; positions/phases unchanged)."""
    names = cache["names"]
    idx = [names.index(a) for a in agents]
    c = dict(cache)
    c["names"] = [names[i] for i in idx]
    c["losses"] = cache["losses"][idx]
    c["choices"] = cache["choices"][idx]
    return c


def run(cache: dict, ks, seeds: int, select_fn, *, ci_boot: int):
    """Run strategy × K × seed on this panel; return (per-seed rows, full_order, full_pr_vec).

    Ranking uses the plain subset-mean PR, which has lower variance than the Horvitz-Thompson
    estimate under concentrated sampling; ranking is scale-free, so its bias does not matter.
    PR-MAE uses the HT estimate, which is unbiased for the absolute PR. Selection sees only this
    panel's agents.
    """
    losses, choices, names = cache["losses"], cache["choices"], cache["names"]
    phases, n_pool, n_agents = list(cache["phases"]), cache["n_pos"], cache["losses"].shape[0]
    full_pr = losses.mean(axis=1) * rr.PR_SCALE
    full_order = rr.ranking(full_pr)
    tight = [(names.index(a), names.index(b)) for a, b in TIGHT_PAIRS if a in names and b in names]
    rows = []
    for strat in STRATEGIES:
        for K in ks:
            for seed in range(seeds):
                sel = select_fn(strat, K, losses=losses, choices=choices, phases=phases, seed=seed)
                idx, incl = list(sel.indices), list(sel.inclusion_probs)
                pr = rr.pr_subset(losses, idx)
                pr_ht = np.array([rr.ht_pr(losses[a], idx, incl, n_pool) for a in range(n_agents)])
                rows.append(
                    {
                        "strategy": strat,
                        "K": K,
                        "seed": seed,
                        "kendall_tau": rr.kendall_vs_full(rr.ranking(pr), full_order),
                        "pairwise_acc": rr.pairwise_accuracy(rr.ranking(pr), full_order),
                        "tight_pair_acc": _tight_pair_acc(pr, full_pr, tight),
                        "top1": float(rr.top1_recovered(rr.ranking(pr), full_order)),
                        "pr_mae": rr.pr_mae(pr_ht, full_pr),
                        "mean_ci_width": rr.ci_width(losses, idx, n_boot=ci_boot, seed=seed),
                        "n_labels": float(len(idx)),
                        "kendall_tau_ht": rr.kendall_vs_full(rr.ranking(pr_ht), full_order),
                        "tight_pair_acc_ht": _tight_pair_acc(pr_ht, full_pr, tight),
                        "top1_ht": float(rr.top1_recovered(rr.ranking(pr_ht), full_order)),
                    }
                )
    return rows, full_order, full_pr


def _tight_pair_acc(pr_sub, pr_full, tight_pairs) -> float:
    """Fraction of the tight contrasts ordered the same under the subset PR as under the full PR."""
    correct = sum(1 for a, b in tight_pairs if (pr_sub[a] > pr_sub[b]) == (pr_full[a] > pr_full[b]))
    return correct / len(tight_pairs)


def run_loao(cache, ks, seeds, select_fn) -> list[dict]:
    """Leave-one-agent-out disagreement selection.

    For each held-out agent ``a``: select K positions using only the other agents' disagreement,
    score the full panel on those positions by Horvitz-Thompson PR, and check whether ``a``'s rank
    matches its full-suite rank. Returns per-(K, seed) rows: the mean rank displacement of the
    held-out agent and the fraction of held-out agents whose rank is recovered exactly.
    """
    losses, choices, phases = cache["losses"], cache["choices"], list(cache["phases"])
    n_agents, n_pool = losses.shape[0], losses.shape[1]
    full_order = rr.ranking(losses.mean(axis=1) * rr.PR_SCALE)
    full_rank = {a: i for i, a in enumerate(full_order)}
    rows = []
    for K in ks:
        for seed in range(seeds):
            disp, exact = [], []
            for a in range(n_agents):
                others = [i for i in range(n_agents) if i != a]
                sel = select_fn(
                    "disagreement",
                    K,
                    losses=losses[others],
                    choices=choices[others],
                    phases=phases,
                    seed=seed,
                )
                idx, incl = list(sel.indices), list(sel.inclusion_probs)
                pr_ht = np.array([rr.ht_pr(losses[j], idx, incl, n_pool) for j in range(n_agents)])
                order = rr.ranking(pr_ht)
                induced_rank = {ag: i for i, ag in enumerate(order)}
                disp.append(abs(induced_rank[a] - full_rank[a]))
                exact.append(induced_rank[a] == full_rank[a])
            rows.append(
                {
                    "K": K,
                    "seed": seed,
                    "mean_rank_disp": float(np.mean(disp)),
                    "frac_exact": float(np.mean(exact)),
                }
            )
    return rows


def write_metric_csv(path: Path, rows_full, rows_cl) -> None:
    """Write both panels' per-seed metrics to one CSV, distinguished by a `panel` column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = ["panel,strategy,K,seed," + ",".join(METRIC_COLS)]
    for panel, rows in (("all8", rows_full), ("cluster", rows_cl)):
        for r in rows:
            csv_lines.append(
                f"{panel},{r['strategy']},{r['K']},{r['seed']},"
                + ",".join(f"{r[m]:.6f}" for m in METRIC_COLS)
            )
    path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def write_htrank_csv(path: Path, rows_full, rows_cl) -> None:
    """Write the plain-mean and Horvitz-Thompson ranking metrics per seed."""
    ht_lines = [
        "panel,strategy,design_source,K,seed," + ",".join(c for pair in HT_PAIRS for c in pair)
    ]
    for panel, rows in (("all8", rows_full), ("cluster", rows_cl)):
        for r in rows:
            ht_lines.append(
                f"{panel},{r['strategy']},{DESIGN_SOURCE[r['strategy']]},{r['K']},"
                f"{r['seed']}," + ",".join(f"{r[c]:.6f}" for pair in HT_PAIRS for c in pair)
            )
    path.write_text("\n".join(ht_lines) + "\n", encoding="utf-8")


def write_loao_csv(path: Path, loao_agg, ks) -> None:
    """Write the leave-one-agent-out exact-recovery share and rank displacement per K."""
    path.write_text(
        "K,frac_exact,mean_rank_disp\n"
        + "\n".join(
            f"{k},{loao_agg[k]['frac_exact']:.6f},{loao_agg[k]['mean_rank_disp']:.6f}"
            for k in sorted(ks)
        )
        + "\n",
        encoding="utf-8",
    )


def summary(args, cl_names, saving, agg_full, agg_cl, loao_agg, paired) -> dict:
    """Return the JSON summary of the run."""
    return {
        "ks": args.ks,
        "seeds": args.seeds,
        "cluster": cl_names,
        "design_source": {s: DESIGN_SOURCE[s] for s in STRATEGIES},
        "saving": saving,
        "agg_all8": {s: {int(k): agg_full[s][k] for k in args.ks} for s in STRATEGIES},
        "agg_cluster": {s: {int(k): agg_cl[s][k] for k in args.ks} for s in STRATEGIES},
        "loao": {int(k): loao_agg[k] for k in sorted(args.ks)},
        "paired_vs_uniform": paired,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings file."""
    out_dir = OUTPUTS / EXPERIMENT
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--ks", nargs="+", type=int, default=list(SETTINGS["ks"]))
    p.add_argument("--seeds", type=int, default=SETTINGS["seeds"])
    p.add_argument(
        "--ci-boot", type=int, default=SETTINGS["ci_boot"], help="bootstrap reps for CI width"
    )
    p.add_argument("--out-md", type=Path, default=out_dir / f"{EXPERIMENT}.md")
    p.add_argument("--out-csv", type=Path, default=out_dir / f"{EXPERIMENT}.csv")
    p.add_argument(
        "--out-htrank-csv",
        type=Path,
        default=None,
        help="HT-ranked companion CSV (default: <out-csv stem>_htrank.csv next to --out-csv)",
    )
    return p


def main() -> None:
    """Run both panels and the leave-one-agent-out check; write CSVs, report and JSON."""
    args = build_parser().parse_args()
    full_cache = load_cache(args.cache)
    cl_names = [a for a in CLUSTER if a in full_cache["names"]]
    cl_cache = filter_cache(full_cache, cl_names)
    select_fn, _ = load_select()

    print(
        f"running 2 panels × {len(STRATEGIES)} strategies × {len(args.ks)} K × {args.seeds} "
        "seeds ..."
    )
    rows_full, full_order, full_pr = run(
        full_cache, args.ks, args.seeds, select_fn, ci_boot=args.ci_boot
    )
    rows_cl, _, _ = run(cl_cache, args.ks, args.seeds, select_fn, ci_boot=args.ci_boot)
    agg_full, agg_cl = aggregate(rows_full, args.ks), aggregate(rows_cl, args.ks)
    saving = saving_vs_uniform(agg_cl, args.ks)
    paired = {
        panel: {m: paired_vs_uniform(rows, args.ks, m) for m in PAIRED_METRICS}
        for panel, rows in (("cluster", rows_cl), ("all8", rows_full))
    }

    for strat in STRATEGIES:
        cells = " ".join(f"K{k}:{agg_cl[strat][k]['kendall_tau'][0]:.3f}" for k in sorted(args.ks))
        print(f"  {strat:>15} cluster-τ: {cells}")
    for panel, metric in (("cluster", "tight_pair_acc"), ("all8", "kendall_tau")):
        counts = paired[panel][metric]["active_rank"]
        cells = " ".join(
            f"K{k}:{counts[k]['at_least']}/{counts[k]['strictly']}" for k in sorted(args.ks)
        )
        print(f"  {panel} {metric} active_rank draws at least as good / strictly better: {cells}")
    k_min = min(args.ks)
    margins = ", ".join(
        f"{s} {paired['cluster']['kendall_tau'][s][k_min]['margin_in_se']:.2f}"
        for s in STRATEGIES
        if s != "random"
    )
    print(f"  cluster kendall_tau K={k_min} margin over uniform in paired SEs: {margins}")
    print(
        f"half the labels: {'yes' if saving['half_labels'] else 'no'} "
        f"(target cluster-τ={saving['target_cluster_tau']:.3f}); "
        + ", ".join(f"{s}→K{saving['reach_k'][s]}" for s in STRATEGIES)
    )

    print("leave-one-agent-out robustness ...")
    loao_agg = aggregate_loao(run_loao(full_cache, args.ks, args.seeds, select_fn), sorted(args.ks))

    write_metric_csv(args.out_csv, rows_full, rows_cl)
    ht_csv = args.out_htrank_csv or args.out_csv.with_name(args.out_csv.stem + "_htrank.csv")
    write_htrank_csv(ht_csv, rows_full, rows_cl)
    # Named after --out-csv's stem, so a redirected run never overwrites the default table.
    write_loao_csv(args.out_csv.with_name(args.out_csv.stem + "_loao.csv"), loao_agg, args.ks)

    out_md = args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    report = render_md(
        cl_names,
        agg_full,
        agg_cl,
        args.ks,
        full_order,
        full_pr,
        full_cache["names"],
        saving,
        loao=loao_agg,
        seeds=args.seeds,
        paired=paired,
    )
    out_md.write_text(report, encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(summary(args, cl_names, saving, agg_full, agg_cl, loao_agg, paired), indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_csv}, {ht_csv}, {out_md}, {out_json}")


if __name__ == "__main__":
    main()
