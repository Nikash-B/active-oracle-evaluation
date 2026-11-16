"""Designs, aggregates, saving and paired margins over uniform sampling for label efficiency."""

import numpy as np

from active_oracle_evaluation.config import settings

EXPERIMENT = "label_efficiency"
SETTINGS = settings(EXPERIMENT)
STRATEGIES = tuple(SETTINGS["strategies"])
# What each design reads before any label is bought. `active_rank` steers on the estimand itself
# (eval_select's importance="skyline"), so its curve is an upper bound, not an operating point.
DESIGN_SOURCE = {
    "random": "none",
    "phase_balanced": "label_free",
    "disagreement": "label_free",
    "active_rank": "skyline",
}
METRIC_COLS = (
    "kendall_tau",
    "pairwise_acc",
    "tight_pair_acc",
    "top1",
    "pr_mae",
    "mean_ci_width",
    "n_labels",
)
# The same draws ranked by Horvitz-Thompson PR instead of the plain subset mean. They go to their
# own CSV, so the per-seed CSV keeps only METRIC_COLS.
HT_RANK_COLS = ("kendall_tau_ht", "tight_pair_acc_ht", "top1_ht")
# The near-tied competitive cluster. Aggregate τ over the full panel saturates because the outliers
# (random, heuristic) are trivially separable, so the cluster is the discriminating test.
CLUSTER = tuple(SETTINGS["cluster"])
# The near-tied cluster pairs whose CIs overlap: the hardest individual contrasts.
TIGHT_PAIRS = tuple(tuple(pair) for pair in SETTINGS["tight_pairs"])
# Thresholds for the labels-to-threshold table.
TAU_THRESHOLDS = tuple(SETTINGS["tau_thresholds"])
BAND_LO, BAND_HI = SETTINGS["band_percentiles"]
# Metrics compared draw by draw against uniform sampling.
PAIRED_METRICS = ("kendall_tau", "tight_pair_acc")
# Paired differences closer to zero than this count as ties (floating-point noise only).
TIE_TOL = 1e-12


def aggregate(rows, ks):
    """Return strategy -> K -> {metric: (mean, sd, lo, hi)} over seeds."""
    agg = {s: {k: {} for k in ks} for s in STRATEGIES}
    for s in STRATEGIES:
        for k in ks:
            cell = [r for r in rows if r["strategy"] == s and r["K"] == k]
            for m in METRIC_COLS + HT_RANK_COLS:
                if not cell:
                    continue
                vals = np.asarray([r[m] for r in cell])
                agg[s][k][m] = (
                    float(vals.mean()),
                    float(vals.std()),
                    float(np.percentile(vals, BAND_LO)),
                    float(np.percentile(vals, BAND_HI)),
                )
    return agg


def labels_to_threshold(agg, ks, metric: str, thresholds) -> dict:
    """Return ``{strategy: {threshold: K or None}}``: the smallest K whose mean reaches each."""
    ks = sorted(ks)
    out = {}
    for strat in STRATEGIES:
        out[strat] = {}
        for t in thresholds:
            hit = next((k for k in ks if agg[strat][k][metric][0] >= t), None)
            out[strat][t] = hit
    return out


def labels_to_target(agg, ks, target: float, metric: str = "kendall_tau") -> dict:
    """Smallest K at which each strategy's mean ``metric`` first reaches ``target`` (or None)."""
    ks = sorted(ks)
    return {
        strat: next((k for k in ks if agg[strat][k][metric][0] >= target), None)
        for strat in STRATEGIES
    }


def saving_vs_uniform(agg_cl, ks) -> dict:
    """Compare each design's labels to reach uniform's max-K cluster-τ with uniform's own reach.

    The target is uniform's mean cluster-τ at the largest K. A design passes when it reaches that
    target at no more than half the K at which uniform itself first reaches it; the saving is
    uniform's first reach divided by the design's.
    """
    ks = sorted(ks)
    k_max = ks[-1]
    target = agg_cl["random"][k_max]["kendall_tau"][0]
    reach = labels_to_target(agg_cl, ks, target, "kendall_tau")
    uniform_k = reach["random"]  # never None: random reaches its own max-K value by k_max
    out = {
        "k_max": k_max,
        "target_cluster_tau": target,
        "reach_k": reach,
        "uniform_reach_k": uniform_k,
        "savings": {},
    }
    half_labels = False
    for strat in STRATEGIES:
        if strat == "random":
            continue
        k = reach[strat]
        savings = (uniform_k / k) if k else None
        passed = k is not None and k <= uniform_k / 2
        half_labels = half_labels or passed
        out["savings"][strat] = {"reach_k": k, "x_fewer": savings, "passed": passed}
    out["half_labels"] = half_labels
    return out


def aggregate_loao(rows, ks) -> dict:
    """Return K -> {mean_rank_disp, frac_exact} averaged over seeds and held-out agents."""
    out = {}
    for k in ks:
        cell = [r for r in rows if r["K"] == k]
        out[k] = {
            "mean_rank_disp": float(np.mean([r["mean_rank_disp"] for r in cell])),
            "frac_exact": float(np.mean([r["frac_exact"] for r in cell])),
        }
    return out


def paired_vs_uniform(rows, ks, metric: str) -> dict:
    """Return strategy -> K -> draw counts and paired margin of ``metric`` over uniform sampling.

    Each draw of a design is paired with the uniform ("random") draw of the same seed, and
    d = design minus uniform. ``at_least`` counts d >= 0, ``strictly`` counts d > 0, and
    ``margin_in_se`` is mean(d) over its paired standard error, std(d, ddof=1) / sqrt(n).
    """
    by_cell = {(r["strategy"], r["K"], r["seed"]): r[metric] for r in rows}
    out = {}
    for strat in STRATEGIES:
        if strat == "random":
            continue
        out[strat] = {}
        for k in ks:
            seeds = sorted(sd for s, kk, sd in by_cell if s == strat and kk == k)
            d = np.asarray([by_cell[(strat, k, sd)] - by_cell[("random", k, sd)] for sd in seeds])
            paired_se = float(d.std(ddof=1) / np.sqrt(len(d)))
            mean_diff = float(d.mean())
            out[strat][k] = {
                "n_draws": len(d),
                "at_least": int((d >= -TIE_TOL).sum()),
                "strictly": int((d > TIE_TOL).sum()),
                "mean_diff": mean_diff,
                "paired_se": paired_se,
                "margin_in_se": mean_diff / paired_se if paired_se > 0 else None,
            }
    return out
