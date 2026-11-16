r"""Game-clustered confidence intervals for the per-phase regret results.

Tests two per-phase statements from the frozen score cache, with no agent re-scored and
no oracle call:

* whether each agent's error is higher in the three contact phases than in the easy
  phases, and whether its worst contact phase separates from its runner-up;
* whether AlphaZero's edge over the competitive cluster differs between the easy and
  the contact phases, in absolute and in relative terms.

The bootstrap resamples source games, not decisions. One shared set of replicates
serves every statistic, so a contrast between two phases or two agents is a bootstrap
of the difference rather than a comparison of two marginal intervals.

Writes, in outputs/phase_intervals/ by default:
    phase_contrasts{tag}.csv   per agent: contact-phase pairwise contrasts and CIs
    phase_az_edge{tag}.csv     per phase: AlphaZero's edge over the cluster and CIs
    phase_intervals{tag}.md    a summary of both tests
    phase_intervals{tag}.json  the numbers of the summary

The paper's settings (the 0-ply suite, then its 2-ply re-score), run from the repo root:

    python -m active_oracle_evaluation.experiments.phase_intervals \
        --cache data/decision_suites/full_suite_scores_0ply.npz
    python -m active_oracle_evaluation.experiments.phase_intervals \
        --cache data/decision_suites/full_suite_scores_2ply.npz --tag _2ply
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.score_cache import ScoreCache
from active_oracle_evaluation.eval.tail_risk import (
    bootstrap_p_two_sided,
    game_bootstrap_indices,
    holm_adjust,
    percentile_ci,
    phase_mean_replicates,
)
from active_oracle_evaluation.experiments.phase_intervals_report import write_outputs
from active_oracle_evaluation.paths import DATA

EXPERIMENT = "phase_intervals"
DEFAULT_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply.npz"


def load_settings() -> dict:
    """Return the experiment settings, with the phase and agent lists as tuples."""
    cfg = dict(settings(EXPERIMENT))
    for key in ("contact_phases", "easy_phases", "cluster"):
        cfg[key] = tuple(cfg[key])
    cfg["phase_order"] = cfg["easy_phases"] + cfg["contact_phases"]
    return cfg


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Build the command-line parser, with defaults from the settings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="per-agent regret score cache (.npz)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=cfg["n_boot"], help="clustered bootstrap replicates"
    )
    parser.add_argument("--boot-seed", type=int, default=cfg["boot_seed"], help="bootstrap seed")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="output directory (default outputs/<experiment>)"
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="summary path (default <out-dir>/phase_intervals<tag>.md)",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filenames, e.g. '_2ply'")
    return parser


def point(col: np.ndarray) -> float:
    """Return the observed value of a statistic, which is row 0 of its replicate column.

    Replicate matrices put the identity index first (see :func:`replicates`), so the same
    arithmetic gives both the point estimate and the bootstrap draws.
    """
    return float(col[0])


def draws(col: np.ndarray) -> np.ndarray:
    """Return the bootstrap replicates of a statistic (row 0 is the observed value)."""
    return col[1:]


def contact_contrasts(
    reps: dict[str, np.ndarray], phases: list[str], names: list[str], cfg: dict
) -> list[dict]:
    """Pairwise contrasts among the contact phases, per agent, with clustered CIs.

    Each pair of contact phases gets a difference (higher mean minus lower), a 95%
    interval and a two-sided bootstrap p. The pair of the agent's worst contact phase
    against its runner-up is flagged, and the flagged tests are Holm-corrected as one
    family with one test per agent.
    """
    contact_phases = cfg["contact_phases"]
    col = {ph: j for j, ph in enumerate(phases)}
    rows: list[dict] = []
    for nm in names:
        r = reps[nm]
        means = {ph: point(r[:, col[ph]]) for ph in contact_phases}
        ranked = sorted(contact_phases, key=lambda ph: -means[ph])
        worst, runner_up = ranked[0], ranked[1]
        for a, b in itertools.combinations(contact_phases, 2):
            hi_ph, lo_ph = (a, b) if means[a] >= means[b] else (b, a)
            d = r[:, col[hi_ph]] - r[:, col[lo_ph]]
            ci_lo, ci_hi = percentile_ci(draws(d))
            rows.append(
                {
                    "agent": nm,
                    "phase_hi": hi_ph,
                    "phase_lo": lo_ph,
                    "mean_hi": means[hi_ph],
                    "mean_lo": means[lo_ph],
                    "diff": point(d),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p": bootstrap_p_two_sided(draws(d)),
                    "is_worst_vs_runner_up": hi_ph == worst and lo_ph == runner_up,
                }
            )
    worst_pairs = [r for r in rows if r["is_worst_vs_runner_up"]]
    for r, p_adj in zip(worst_pairs, holm_adjust([r["p"] for r in worst_pairs])):
        r["p_holm"] = p_adj
        r["separated"] = bool(r["ci_lo"] > 0.0)
        r["separated_holm"] = bool(p_adj < cfg["holm_alpha"])
    for r in rows:
        r.setdefault("p_holm", float("nan"))
        r.setdefault("separated", bool(r["ci_lo"] > 0.0))
        r.setdefault("separated_holm", False)
    return rows


def contact_minus_easy(
    reps: dict[str, np.ndarray], phases: list[str], names: list[str], cfg: dict
) -> list[dict]:
    """Contact phases minus easy phases, per agent, with a clustered CI.

    Each group is the unweighted mean of its three phase means, so the contrast does not
    depend on how the suite is split across phases.
    """
    col = {ph: j for j, ph in enumerate(phases)}
    rows = []
    for nm in names:
        r = reps[nm]
        contact = np.nanmean(np.stack([r[:, col[ph]] for ph in cfg["contact_phases"]]), axis=0)
        easy = np.nanmean(np.stack([r[:, col[ph]] for ph in cfg["easy_phases"]]), axis=0)
        d = contact - easy
        lo, hi = percentile_ci(draws(d))
        rows.append(
            {
                "agent": nm,
                "contact_mean": point(contact),
                "easy_mean": point(easy),
                "diff": point(d),
                "ci_lo": lo,
                "ci_hi": hi,
                "p": bootstrap_p_two_sided(draws(d)),
            }
        )
    return rows


def _group_mean(by_phase: dict[str, np.ndarray], group: tuple[str, ...]) -> np.ndarray:
    """Unweighted mean over the phases of ``group``, per replicate."""
    return np.nanmean(np.stack([by_phase[ph] for ph in group]), axis=0)


def _uniformity(abs_by_phase: dict, rel_by_phase: dict, cfg: dict) -> dict:
    """Easy-phase edge minus contact-phase edge, in absolute and relative terms, with CIs."""
    uniformity = {}
    for label, by_phase in (("abs", abs_by_phase), ("rel", rel_by_phase)):
        easy = _group_mean(by_phase, cfg["easy_phases"])
        contact = _group_mean(by_phase, cfg["contact_phases"])
        gap = easy - contact
        lo, hi = percentile_ci(draws(gap))
        uniformity[label] = {
            "easy": point(easy),
            "contact": point(contact),
            "gap": point(gap),
            "ci_lo": lo,
            "ci_hi": hi,
            "p": bootstrap_p_two_sided(draws(gap)),
        }
    return uniformity


def az_edge(reps: dict[str, np.ndarray], phases: list[str], cfg: dict) -> tuple[list[dict], dict]:
    """Per-phase absolute and relative edge of the focus agent over the cluster, with CIs.

    The absolute edge is the cluster's mean regret minus AlphaZero's; the relative
    edge is the share of the cluster's error that AlphaZero avoids. The cluster is the
    unweighted mean of its members, recomputed inside every replicate. The uniformity
    test is the easy-phase edge minus the contact-phase edge.
    """
    col = {ph: j for j, ph in enumerate(phases)}
    az = reps[cfg["focus"]]
    cl = np.nanmean(np.stack([reps[nm] for nm in cfg["cluster"]]), axis=0)  # (n_boot, n_phases)

    rows = []
    abs_by_phase, rel_by_phase = {}, {}
    for ph in cfg["phase_order"]:
        j = col[ph]
        abs_edge = cl[:, j] - az[:, j]
        rel_edge = 1.0 - az[:, j] / cl[:, j]
        abs_by_phase[ph], rel_by_phase[ph] = abs_edge, rel_edge
        alo, ahi = percentile_ci(draws(abs_edge))
        rlo, rhi = percentile_ci(draws(rel_edge))
        row = {
            "phase": ph,
            "az_mean": point(az[:, j]),
            "cluster_mean": point(cl[:, j]),
            "abs_edge": point(abs_edge),
            "abs_ci_lo": alo,
            "abs_ci_hi": ahi,
            "abs_p": bootstrap_p_two_sided(draws(abs_edge)),
            "rel_edge": point(rel_edge),
            "rel_ci_lo": rlo,
            "rel_ci_hi": rhi,
        }
        for nm in cfg["cluster"]:
            row[f"rel_vs_{nm}"] = point(1.0 - az[:, j] / reps[nm][:, j])
        rows.append(row)
    return rows, _uniformity(abs_by_phase, rel_by_phase, cfg)


def design_effect(cache, phases: list[str], indices, *, n_boot: int, seed: int) -> dict:
    """Ratio of clustered to decision-level CI widths on the per-phase means.

    Averaged over the (agent, phase) cells. A ratio near 1 means the clustering does
    not change the intervals on this suite.
    """
    rng = np.random.default_rng(seed + 1)
    iid = [np.arange(cache.n_pos)]
    iid += [rng.integers(0, cache.n_pos, size=cache.n_pos) for _ in range(n_boot)]
    ratios = []
    for nm in cache.names:
        cl = phase_mean_replicates(cache.losses[nm], cache.phases, indices, phases)
        ii = phase_mean_replicates(cache.losses[nm], cache.phases, iid, phases)
        for j in range(len(phases)):
            a_lo, a_hi = percentile_ci(draws(cl[:, j]))
            b_lo, b_hi = percentile_ci(draws(ii[:, j]))
            if np.isfinite(a_hi - a_lo) and (b_hi - b_lo) > 0:
                ratios.append((a_hi - a_lo) / (b_hi - b_lo))
    r = np.asarray(ratios)
    return {
        "mean_ratio": float(r.mean()),
        "min_ratio": float(r.min()),
        "max_ratio": float(r.max()),
        "n_cells": int(r.size),
    }


def load_cache(cache_path: Path, cfg: dict) -> tuple[ScoreCache, list[str]]:
    """Load the score cache and check that it covers the needed agents and phases."""
    if not cache_path.exists():
        raise SystemExit(f"score cache not found: {cache_path}")
    cache = ScoreCache(cache_path)
    missing = [nm for nm in (cfg["focus"], *cfg["cluster"]) if nm not in cache.losses]
    if missing:
        raise SystemExit(f"cache is missing agent(s) {missing}; have {cache.names}")
    phase_order = cfg["phase_order"]
    phases = [ph for ph in phase_order if ph in set(cache.phases)]
    if set(phases) != set(phase_order):
        raise SystemExit(f"suite phases {sorted(set(cache.phases))} do not cover {phase_order}")
    return cache, phases


def replicates(cache: ScoreCache, phases: list[str], n_boot: int, seed: int):
    """Return the shared game-bootstrap indices and each agent's per-phase mean replicates."""
    # Row 0 is the identity index, so row 0 of every replicate matrix is the observed value.
    indices = [np.arange(cache.n_pos)]
    indices += game_bootstrap_indices(cache.cluster_ids, n_boot=n_boot, seed=seed)
    reps = {
        nm: phase_mean_replicates(cache.losses[nm], cache.phases, indices, phases)
        for nm in cache.names
    }
    return indices, reps


def main() -> None:
    """Load the score cache, bootstrap the per-phase statistics and write the outputs."""
    cfg = load_settings()
    args = build_parser(cfg).parse_args()
    cache, phases = load_cache(args.cache, cfg)

    print(
        f"{cache.n_pos:,} decisions from {cache.n_games} games; "
        f"{args.n_boot:,} clustered replicates (seed {args.boot_seed})"
    )
    indices, reps = replicates(cache, phases, args.n_boot, args.boot_seed)

    contrasts = contact_contrasts(reps, phases, cache.names, cfg)
    primary = contact_minus_easy(reps, phases, cache.names, cfg)
    edge_rows, uniformity = az_edge(reps, phases, cfg)
    deff = design_effect(cache, phases, indices, n_boot=args.n_boot, seed=args.boot_seed)
    results = {
        "contrasts": contrasts,
        "primary": primary,
        "edge_rows": edge_rows,
        "uniformity": uniformity,
        "deff": deff,
    }
    write_outputs(args, cfg, cache, results)
    print("done.")


if __name__ == "__main__":
    main()
