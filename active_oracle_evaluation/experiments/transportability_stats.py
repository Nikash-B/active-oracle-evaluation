"""Per-pair PR gaps, trial-level SDs and the SD-ratio summary of the transportability experiment."""

import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.paths import REPO

SD_RATIO_DEFINITION = (
    "sd_pr_trial: sample SD (ddof=1) across a pairing's trials of the per-trial "
    "PR difference (PR = 500 x mean decision loss per agent per pairing#trial cluster); "
    "sd_ratio = sd_pr_trial / paired_sd (paired_sd: per-trial SD of the duplicate-dice "
    "ppg difference)"
)


def pr_gaps_from_cache(path: Path, pr_per_equity: float) -> dict[str, dict[str, float]]:
    """Return ``{pairing: {agent: PR}}`` from the cached per-decision losses."""
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["agent_names"]]
    labels = [str(x) for x in d["cluster_labels"]]
    agent, cluster, loss = d["agent"], d["cluster"], d["loss"]
    pairing_of_cluster = np.array([labels[c].split("#")[0] for c in range(len(labels))])
    out: dict[str, dict[str, float]] = {}
    for pairing in sorted(set(pairing_of_cluster)):
        in_pairing = pairing_of_cluster[cluster] == pairing
        per = {}
        for agent_index, name in enumerate(names):
            sel = in_pairing & (agent == agent_index)
            if sel.any():
                per[name] = float(loss[sel].mean() * pr_per_equity)
        out[pairing] = per
    return out


def per_trial_pr_diffs(path: Path, pr_per_equity: float) -> dict[str, np.ndarray]:
    """Return ``{"a|b": per-trial PR(b) - PR(a)}`` from the cached losses, in trial order.

    A trial is one ``"pairing#trial"`` cluster: one duplicate-dice pair, two games. Each agent's PR
    in it is 500 x its mean decision loss there. A trial where either agent made no scored decision
    is dropped. The sign matches ``pr_gap`` (positive when ``a`` is stronger).
    """
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["agent_names"]]
    labels = [str(x) for x in d["cluster_labels"]]
    n_agents, n_clusters = len(names), len(labels)
    key = d["cluster"].astype(np.int64) * n_agents + d["agent"].astype(np.int64)
    size = n_clusters * n_agents
    sums = np.bincount(key, weights=d["loss"].astype(np.float64), minlength=size).reshape(
        n_clusters, n_agents
    )
    counts = np.bincount(key, minlength=size).reshape(n_clusters, n_agents)
    trials: dict[str, list[tuple[int, float]]] = {}
    for c, label in enumerate(labels):
        pairing, trial = label.split("#")
        ia, ib = (names.index(x) for x in pairing.split("|"))
        if counts[c, ia] and counts[c, ib]:
            pr_a = pr_per_equity * sums[c, ia] / counts[c, ia]
            pr_b = pr_per_equity * sums[c, ib] / counts[c, ib]
            trials.setdefault(pairing, []).append((int(trial), float(pr_b - pr_a)))
    return {p: np.array([v for _, v in sorted(t)]) for p, t in trials.items()}


def sd_pr_trial(diffs: np.ndarray) -> float:
    """Return the sample SD (ddof=1) of the per-trial PR differences; NaN below two trials."""
    return float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else float("nan")


def build_rows(ppg_side, pr_by_pairing, diffs_by_pairing) -> list[dict]:
    """Join the ppg side with the cached PR side into one row per pairing."""
    rows = []
    for key, g in ppg_side:
        na, nb = key.split("|")
        pr = pr_by_pairing[key]
        # mean_diff is a's points/game vs b; PR is lower-is-better, so a's *advantage* in PR is
        # PR(b) - PR(a). Both are then positive when a is the stronger agent.
        diffs = diffs_by_pairing.get(key, np.array([]))
        sd_pr = sd_pr_trial(diffs)
        rows.append(
            {
                "pairing": key,
                "ppg_gap": g["ppg_gap"],
                "pr_gap": float(pr[nb] - pr[na]),
                "paired_sd": g["paired_sd"],
                "unpaired_sd": g["unpaired_sd"],
                "vr_factor": g["vr_factor"],
                "pr_a": pr[na],
                "pr_b": pr[nb],
                "n_trials_pr": int(len(diffs)),
                "sd_pr_trial": sd_pr,
                "sd_ratio": sd_pr / g["paired_sd"],
            }
        )
    return rows


def pair_group(pairing: str) -> str:
    """Return the pair's group: ``random``, ``heuristic`` or ``cluster``."""
    a, b = pairing.split("|")
    if "random" in (a, b):
        return "random"
    if "heuristic" in (a, b):
        return "heuristic"
    return "cluster"


def repo_relative(path: Path) -> str:
    """Return the path relative to the repo, or its bare name if it lies outside the repo."""
    p = Path(path).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name


def value_range(x: np.ndarray) -> list[float]:
    """Return ``[min, max]`` of an array as floats."""
    return [float(x.min()), float(x.max())]


def corr_log(sd_ratio: np.ndarray, ppg: np.ndarray, mask: np.ndarray) -> float:
    """Return the correlation of log SD ratio with log |ppg gap| over the masked pairs."""
    m = mask & (np.abs(ppg) > 0) & np.isfinite(sd_ratio)
    return float(np.corrcoef(np.log(sd_ratio[m]), np.log(np.abs(ppg[m])))[0, 1])


def group_summaries(rows, sd_ratio, sd_ppg, group) -> dict:
    """Return the SD-ratio summaries of the cluster, heuristic, random and all pairs."""
    cl, heur, rnd = group == "cluster", group == "heuristic", group == "random"
    cluster_rows = sorted((r for r, c in zip(rows, cl) if c), key=lambda r: abs(r["ppg_gap"]))
    return {
        "cluster": {
            "n_pairs": int(cl.sum()),
            "median": float(np.median(sd_ratio[cl])),
            "range": value_range(sd_ratio[cl]),
            "iqr": [float(v) for v in np.percentile(sd_ratio[cl], [25, 75])],
        },
        "closest_cluster_pairs": [
            {"pairing": r["pairing"], "ppg_gap": r["ppg_gap"], "sd_ratio": r["sd_ratio"]}
            for r in cluster_rows[:3]
        ],
        "heuristic_pairs": {
            "n_pairs": int(heur.sum()),
            "range": value_range(sd_ratio[heur]),
            "definition": "heuristic vs a cluster agent (random|heuristic counts as random)",
        },
        "random_pairs": {
            "n_pairs": int(rnd.sum()),
            "range": value_range(sd_ratio[rnd]),
            "paired_sd_range": value_range(sd_ppg[rnd]),
        },
        "all_pairs": {
            "n_pairs": len(rows),
            "median": float(np.median(sd_ratio)),
            "range": value_range(sd_ratio),
        },
    }


def efficiency_summary(sd_ratio, ppg, pr_gap, cl, measurable_ppg) -> dict:
    """Return the gap-ratio median, the implied efficiency and the per-pair efficiency."""
    measurable = np.abs(ppg) > measurable_ppg
    gap_ratio_med = float(np.median(pr_gap[measurable] / ppg[measurable]))
    cl_med = float(np.median(sd_ratio[cl]))
    # Per-pair efficiency (sigma_ppg/d_ppg)^2 / (sigma_PR/d_PR)^2 == (gap ratio / sd ratio)^2.
    eff = (pr_gap / ppg / sd_ratio) ** 2
    return {
        "gap_ratio_median_measurable": gap_ratio_med,
        "measurable_ppg_threshold": measurable_ppg,
        "implied_efficiency_typical": (gap_ratio_med / cl_med) ** 2,
        "implied_efficiency_over_cluster_range": [
            (gap_ratio_med / sd_ratio[cl].max()) ** 2,
            (gap_ratio_med / sd_ratio[cl].min()) ** 2,
        ],
        "per_pair_efficiency_measurable": {
            "all": {
                "median": float(np.median(eff[measurable])),
                "iqr": [float(v) for v in np.percentile(eff[measurable], [25, 75])],
            },
            "cluster": {
                "median": float(np.median(eff[measurable & cl])),
                "iqr": [float(v) for v in np.percentile(eff[measurable & cl], [25, 75])],
            },
        },
    }


def measurement_pair_summary(rows, measurement_pairing, pair_json: Path) -> dict | None:
    """Return the measured pair's round-robin and 2,000-trial SD ratios, or None if absent."""
    pair_row = next((r for r in rows if r["pairing"] == measurement_pairing), None)
    if pair_row is None:
        return None
    summary = {"pairing": measurement_pairing, "sd_ratio_round_robin": pair_row["sd_ratio"]}
    pj = json.loads(Path(pair_json).read_text(encoding="utf-8"))
    summary.update(
        {
            "sd_pr_trial_2000": pj["sd_per_trial"],
            "paired_sd_2000": pj["outcome_sd_paired_per_trial"],
            "sd_ratio_2000": pj["sd_per_trial"] / pj["outcome_sd_paired_per_trial"],
            "source": repo_relative(pair_json),
        }
    )
    return summary


def sd_ratio_summary(rows: list[dict], config: dict, pair_json: Path) -> dict:
    """Summarize the SD ratio by pair group, with the implied efficiency and the measurement pair."""
    sd_ratio = np.array([r["sd_ratio"] for r in rows])
    ppg = np.array([r["ppg_gap"] for r in rows])
    pr_gap = np.array([r["pr_gap"] for r in rows])
    sd_ppg = np.array([r["paired_sd"] for r in rows])
    group = np.array([pair_group(r["pairing"]) for r in rows])
    cl = group == "cluster"
    out = {"definition": SD_RATIO_DEFINITION, "cluster_agents": list(config["cluster"])}
    out.update(group_summaries(rows, sd_ratio, sd_ppg, group))
    out["corr_log_sd_ratio_log_abs_ppg"] = {
        "all": corr_log(sd_ratio, ppg, np.ones_like(cl)),
        "cluster": corr_log(sd_ratio, ppg, cl),
    }
    out.update(efficiency_summary(sd_ratio, ppg, pr_gap, cl, config["measurable_ppg"]))
    pair = measurement_pair_summary(rows, config["measurement_pairing"], pair_json)
    if pair is not None:
        out["measurement_pair"] = pair
    return out


def pair_variance_reduction(pair_json: Path) -> float:
    """Return the measurement pair's duplicate-dice variance reduction from its JSON."""
    return float(
        json.loads(Path(pair_json).read_text(encoding="utf-8"))["outcome_variance_reduction"]
    )
