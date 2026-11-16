"""Two robustness checks of the label-efficiency result, each measured against a baseline.

**A. Leave-one-agent-out rank recovery, several designs, one code path.** The metric is that of
``label_efficiency.run_loao``: the held-out agent's rank in the full-panel Horvitz--Thompson
ranking, compared with its full-suite rank.

* ``loao_disagreement`` -- positions chosen by disagreement among the *other* agents. Its panel
  mean must reproduce the held-out CSV that ``label_efficiency`` writes
  (``outputs/label_efficiency/label_efficiency_0ply_loao.csv``).
* ``uniform`` -- positions drawn uniformly. The design never sees any agent, so this is the rank
  recovery that sampling alone gives.
* ``disagreement_all`` -- disagreement among *all* agents, the held-out one included.
* ``phase_balanced`` -- phase-balanced sampling, which shows whether the statistic can separate
  two designs at all.
* A ``_plain`` suffix ranks by the plain subset mean (the efficiency figures' ranking) instead of
  Horvitz--Thompson PR.

Results are reported over all eight agents and over the six-agent competitive cluster, since
``random`` and ``heuristic`` are recovered at every budget.

**B. Where each strategy first reaches uniform's cluster tau at K_max**, from the per-draw
efficiency CSVs of ``label_efficiency`` (0-ply over 200 draws, 2-ply over 20 draws). The saving
is reported two ways: ``K_max / reach_K`` and uniform's own first reach divided by ``reach_K``.

Settings are in the [label_efficiency_robustness] table of config/experiments.toml. Run from the
repo root, after the two ``label_efficiency`` runs named in ``PRODUCERS``:

    python -m active_oracle_evaluation.experiments.label_efficiency_robustness

The results go to outputs/label_efficiency_robustness/label_efficiency_robustness.{json,md}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import rank_recovery as rr
from active_oracle_evaluation.experiments.label_efficiency_robustness_report import (
    print_summary,
    write_markdown,
)
from active_oracle_evaluation.paths import DATA, OUTPUTS, REPO, output_dir

EXPERIMENT = "label_efficiency_robustness"
CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"
EFFICIENCY_DIR = OUTPUTS / "label_efficiency"
LOAO_CSV = EFFICIENCY_DIR / "label_efficiency_0ply_loao.csv"
EFFICIENCY_CSVS = {
    "0ply_200seed": EFFICIENCY_DIR / "label_efficiency_0ply.csv",
    "2ply_20seed": EFFICIENCY_DIR / "label_efficiency_2ply.csv",
}
_PRODUCER = (
    "python -m active_oracle_evaluation.experiments.label_efficiency"
    " --cache data/decision_suites/full_suite_scores_{depth}.npz"
    " --ks 50 100 250 500 1000 --seeds {seeds} --out-csv outputs/label_efficiency/{name}.csv"
)
# The command that writes each input file.
PRODUCERS = {
    LOAO_CSV: _PRODUCER.format(depth="0ply", seeds=200, name="label_efficiency_0ply"),
    EFFICIENCY_CSVS["0ply_200seed"]: _PRODUCER.format(
        depth="0ply", seeds=200, name="label_efficiency_0ply"
    ),
    EFFICIENCY_CSVS["2ply_20seed"]: _PRODUCER.format(
        depth="2ply", seeds=20, name="label_efficiency_2ply"
    ),
}


def _load_label_efficiency():
    """Return the label-efficiency experiment module."""
    from active_oracle_evaluation.experiments import label_efficiency

    return label_efficiency


def check_inputs():
    """Exit with the producer command if an input written by label_efficiency is missing."""
    for path, command in PRODUCERS.items():
        if not path.exists():
            raise SystemExit(f"missing {path}; run: {command}")


def _select(select_fn, strategy, k, losses, choices, phases, seed):
    """Run one selector draw."""
    return select_fn(strategy, k, losses=losses, choices=choices, phases=phases, seed=seed)


def rank_recovery(cache, select_fn, design: str, ks, seeds, shared_strategy):
    """Return ``{K: (exact[seed, agent], disp[seed, agent])}`` for one design.

    Mirrors ``run_loao``, which ranks the panel by Horvitz--Thompson PR. A ``_plain`` suffix ranks
    by the plain subset mean instead. HT's inverse-probability weights add variance under any
    non-uniform design, so the two rankings can disagree; for uniform they are identical.
    """
    plain = design.endswith("_plain")
    design = design.removesuffix("_plain")
    losses, choices, phases = cache["losses"], cache["choices"], list(cache["phases"])
    n_agents, n_pool = losses.shape
    full_order = rr.ranking(losses.mean(axis=1) * rr.PR_SCALE)
    full_rank = {agent: i for i, agent in enumerate(full_order)}
    out = {}
    for K in ks:
        exact = np.zeros((seeds, n_agents), dtype=bool)
        disp = np.zeros((seeds, n_agents))
        for seed in range(seeds):
            shared = None
            for held_out in range(n_agents):
                if design == "loao_disagreement":
                    others = [i for i in range(n_agents) if i != held_out]
                    sel = _select(
                        select_fn, "disagreement", K, losses[others], choices[others], phases, seed
                    )
                else:
                    if shared is None:
                        shared = _select(
                            select_fn, shared_strategy[design], K, losses, choices, phases, seed
                        )
                    sel = shared
                idx, incl = list(sel.indices), list(sel.inclusion_probs)
                if plain:
                    pr = rr.pr_subset(losses, idx)
                else:
                    pr = np.array([rr.ht_pr(losses[j], idx, incl, n_pool) for j in range(n_agents)])
                induced = {agent: i for i, agent in enumerate(rr.ranking(pr))}
                exact[seed, held_out] = induced[held_out] == full_rank[held_out]
                disp[seed, held_out] = abs(induced[held_out] - full_rank[held_out])
        out[K] = (exact, disp)
    return out


def summarize(res, agent_idx):
    """Mean exact-recovery share (with SE over seeds) and mean rank displacement per K."""
    rows = {}
    for K, (exact, disp) in res.items():
        per_seed = exact[:, agent_idx].mean(axis=1)
        rows[K] = {
            "frac_exact": float(per_seed.mean()),
            "frac_exact_se_over_seeds": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))),
            "mean_rank_disp": float(disp[:, agent_idx].mean()),
        }
    return rows


def paired(designs, agent_idx, pairs):
    """Per-seed paired differences in exact-recovery share for each pair in ``pairs``.

    The designs share seed numbers, so differences are taken seed by seed.
    """
    out = {}
    for first, second in pairs:
        key = f"{first}_minus_{second}"
        out[key] = {}
        for K in designs[first]:
            diff = designs[first][K][0][:, agent_idx].mean(axis=1) - designs[second][K][0][
                :, agent_idx
            ].mean(axis=1)
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            out[key][K] = {
                "mean_diff": float(diff.mean()),
                "paired_se": se,
                "z": float(diff.mean() / se) if se > 0 else None,
                "seeds_a_higher": int(np.sum(diff > 0)),
                "seeds_b_higher": int(np.sum(diff < 0)),
            }
    return out


def thresholds(path: Path):
    """First budget at which each strategy reaches uniform's mean cluster tau at K_max."""
    tau = defaultdict(list)
    for row in csv.DictReader(path.open()):
        if row["panel"] == "cluster":
            tau[(row["strategy"], int(row["K"]))].append(float(row["kendall_tau"]))
    ks = sorted({k for _, k in tau})
    mean = {s: {k: float(np.mean(tau[(s, k)])) for k in ks} for s, _ in tau}
    k_max = ks[-1]
    target = mean["random"][k_max]
    reach = {s: next((k for k in ks if mean[s][k] >= target), None) for s in mean}
    return {
        "seeds_per_cell": len(tau[("random", k_max)]),
        "cluster_tau_mean": mean,
        "target_uniform_tau_at_k_max": target,
        "uniform_first_reaches_target_at": reach["random"],
        "strategies": {
            s: {
                "reach_k": reach[s],
                "saving_vs_uniform_budget_at_k_max": (k_max / reach[s]) if reach[s] else None,
                "saving_vs_uniform_own_reach": (reach["random"] / reach[s]) if reach[s] else None,
            }
            for s in mean
            if s != "random"
        },
    }


def loao_control(designs, everyone, ks):
    """Largest absolute difference of the loao_disagreement design from the held-out CSV."""
    reference = {
        int(row["K"]): (float(row["frac_exact"]), float(row["mean_rank_disp"]))
        for row in csv.DictReader(LOAO_CSV.open())
    }
    loao_all = summarize(designs["loao_disagreement"], everyone)
    return max(
        max(
            abs(loao_all[k]["frac_exact"] - reference[k][0]),
            abs(loao_all[k]["mean_rank_disp"] - reference[k][1]),
        )
        for k in ks
    )


def build_results(cfg):
    """Run both checks and return the results dict and the control difference."""
    label_efficiency = _load_label_efficiency()
    cache = label_efficiency.load_cache(CACHE)
    select_fn, _ = label_efficiency.load_select()
    names = cache["names"]
    cluster = [names.index(a) for a in cfg["cluster"] if a in names]
    everyone = list(range(len(names)))
    ks, seeds, pairs = cfg["ks"], cfg["seeds"], cfg["pairs"]

    designs = {
        design: rank_recovery(cache, select_fn, design, ks, seeds, cfg["shared_strategy"])
        for design in cfg["designs"]
    }
    control = loao_control(designs, everyone, ks)
    out = {
        "tool": f"python -m active_oracle_evaluation.experiments.{EXPERIMENT}",
        "cache": str(CACHE.relative_to(REPO)),
        "seeds": seeds,
        "ks": list(ks),
        "loao": {
            "control_max_abs_diff_vs_label_efficiency_csv": control,
            "reproduces_label_efficiency_csv": control < cfg["control_tolerance"],
            "agents": names,
            "cluster": [names[i] for i in cluster],
            "all_agents": {d: summarize(res, everyone) for d, res in designs.items()},
            "cluster_agents": {d: summarize(res, cluster) for d, res in designs.items()},
            "per_agent_frac_exact": {
                d: {names[a]: {K: float(res[K][0][:, a].mean()) for K in ks} for a in everyone}
                for d, res in designs.items()
            },
            "paired_differences": {
                "all_agents": paired(designs, everyone, pairs),
                "cluster_agents": paired(designs, cluster, pairs),
            },
        },
        "threshold_logic": {name: thresholds(path) for name, path in EFFICIENCY_CSVS.items()},
    }
    return out, control


def main() -> int:
    """Run both checks, print a summary and write the results as JSON and Markdown."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=f"results JSON (default outputs/{EXPERIMENT}/{EXPERIMENT}.json); "
        "the Markdown tables go next to it with a .md suffix",
    )
    args = parser.parse_args()
    cfg = settings(EXPERIMENT)
    check_inputs()

    json_path = args.json if args.json else output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    out, control = build_results(cfg)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    print_summary(out, control, cfg["ks"])
    md_path = write_markdown(EXPERIMENT, out, control, cfg["ks"], json_path.with_suffix(".md"))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if control < cfg["control_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
