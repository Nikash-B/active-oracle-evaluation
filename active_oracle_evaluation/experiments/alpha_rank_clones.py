"""Test whether alpha-rank and Nash averaging are invariant to duplicating an agent of the panel.

Each agent of the panel is duplicated in turn. The clone inherits its original's wins and games
against every other agent and plays a typical-sized, evenly split match against the original, the
same injection as ``clone_invariance.elo_with_clone``. On the duplicated field we measure:

* Elo: the largest anchored shift among the other agents (``clone_invariance.ANCHOR`` pinned,
  ``ANCHOR_FALLBACK`` when the anchor is duplicated), the largest change in any Elo difference
  between two other agents, and whether the originals reorder;
* Nash averaging: ratings ``r = A p*`` with ``A = P - 1/2`` and ``p*`` the maximum-entropy
  equilibrium (Balduzzi et al., 2018), which should leave the other agents' ratings unchanged;
* alpha-rank (single population, ``T[i, j] = eta * logistic(alpha * (P[j, i] - P[i, j]))`` with
  ``eta = 1 / (n - 1)``): the largest change in another agent's stationary mass, the total-variation
  distance after merging the clone's mass back into its original, the split between original and
  clone, and whether the originals' order by mass changes (agents above ``mass_floor`` only).

The logistic uses ``scipy.special.expit``, which stays finite at large alpha; the control counts
the alphas at which ``cyclic_triple._alpha_rank`` raises ``OverflowError``. Stationary
distributions are solved by eigenvector and by linear solve; cells where the two disagree by more
than ``tolerance`` are reported as numerically unresolved and left out of the summary.

Controls: the Elo shifts match ``clone_invariance.elo_with_clone`` on the eight-agent panel and the
clone cells of ``nash_averaging`` on the seven-agent panel; alpha-rank matches
``cyclic_triple._alpha_rank`` and the seven-agent distributions in
``data/round_robins/cyclic_triple.json``; Nash averaging moves no other rating.

Settings are read from the [alpha_rank_clones] table of the config. Run from the repository root
after ``python -m active_oracle_evaluation.experiments.nash_averaging``:

    python -m active_oracle_evaluation.experiments.alpha_rank_clones

It writes outputs/alpha_rank_clones/alpha_rank_clones.json (every cell) and alpha_rank_clones.md
(the summary tables). Exit code 0 when every control passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo
from active_oracle_evaluation.experiments import clone_invariance as rci
from active_oracle_evaluation.experiments import cyclic_triple as ct
from active_oracle_evaluation.experiments import nash_averaging as nash
from active_oracle_evaluation.experiments.alpha_rank_clones_report import (
    print_summary,
    write_markdown,
)
from active_oracle_evaluation.paths import DATA, OUTPUTS, output_dir

EXPERIMENT = "alpha_rank_clones"
ROUND_ROBIN_JSON = DATA / "round_robins" / "elo_validity.json"
PRECISION = DATA / "round_robins" / "intransitivity_precision.json"
CYCLIC_TRIPLE_JSON = DATA / "round_robins" / "cyclic_triple.json"
NASH_AVERAGING_JSON = OUTPUTS / "nash_averaging" / "nash_averaging.json"
NASH_AVERAGING_COMMAND = "python -m active_oracle_evaluation.experiments.nash_averaging"
TOOL = "python -m active_oracle_evaluation.experiments.alpha_rank_clones"


def panel_round_robin_8():
    """Load the eight-agent head-to-head round-robin (names, wins, games, raw record)."""
    data = json.loads(ROUND_ROBIN_JSON.read_text())
    return (
        list(data["names"]),
        np.asarray(data["wins"], float),
        np.asarray(data["games"], float),
        data,
    )


def panel_precision_7():
    """Load the seven-agent high-precision round-robin (names, wins, games, raw record)."""
    data = json.loads(PRECISION.read_text())
    names = list(data["agents"])
    n = len(names)
    wins, games = np.zeros((n, n)), np.zeros((n, n))
    for pairing in data["pairings"]:
        i, j = names.index(pairing["a"]), names.index(pairing["b"])
        wins[i, j], wins[j, i] = pairing["a_wins"], pairing["games"] - pairing["a_wins"]
        games[i, j] = games[j, i] = pairing["games"]
    return names, wins, games, data


def inject(names, wins, games, clone_of):
    """Add a clone of ``clone_of``, built as in ``clone_invariance.elo_with_clone``."""
    i, n = names.index(clone_of), len(names)
    w, g = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
    w[:n, :n], g[:n, :n] = wins, games
    w[n, :n], g[n, :n] = wins[i], games[i]
    w[:n, n], g[:n, n] = wins[:, i], games[:, i]
    typical = float(np.median(games[games > 0]))
    g[i, n] = g[n, i] = typical
    w[i, n] = w[n, i] = typical / 2.0
    return [*names, f"{clone_of}_clone"], w, g


def winprob(wins, games):
    """Return the constant-sum win-probability matrix (0.5 where no games were played)."""
    P = np.full(wins.shape, 0.5)
    played = games > 0
    P[played] = wins[played] / games[played]
    assert np.allclose(P + P.T, 1.0), "win-probability matrix is not constant-sum"
    return P


def alpha_rank(P, alphas):
    """Return the alpha-rank stationary distribution and eig/solve gap for each alpha."""
    n = P.shape[0]
    eta = 1.0 / (n - 1)
    out = {}
    for alpha in alphas:
        T = eta * expit(alpha * (P.T - P))
        np.fill_diagonal(T, 0.0)
        np.fill_diagonal(T, 1.0 - T.sum(axis=1))
        vals, vecs = np.linalg.eig(T.T)
        k = int(np.argmin(np.abs(vals - 1.0)))
        pi_eig = np.abs(np.real(vecs[:, k]))
        pi_eig /= pi_eig.sum()
        M = T.T - np.eye(n)
        M[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi_lin = np.linalg.solve(M, b)
        gap = float(np.max(np.abs(pi_eig - pi_lin)))
        out[alpha] = {"pi": pi_eig, "method_gap": gap}
    return out


def ordered(masses, idx):
    """Sort agent indices by descending mass, ties broken by index."""
    return sorted(idx, key=lambda k: (-round(float(masses[k]), 12), k))


def bt(names, wins, games):
    """Fit Bradley-Terry Elo to a wins/games matrix."""
    record = RoundRobin(names=list(names), games=games, wins=wins)
    return bradley_terry_elo(record)


def check_repo(P, controls, alphas):
    """Compare this alpha-rank with ``cyclic_triple._alpha_rank`` where it computes."""
    for alpha in alphas:
        try:
            ref = np.array(ct._alpha_rank(P, alphas=(alpha,))[str(alpha)])
        except OverflowError:
            controls["repo_alpha_rank_overflows"] += 1
            continue
        ours = alpha_rank(P, alphas=(alpha,))[alpha]["pi"]
        controls["repo_alpha_rank_max_diff"] = max(
            controls["repo_alpha_rank_max_diff"], float(np.max(np.abs(ref - ours)))
        )


def elo_cell(names, elo_base, names2, w2, g2, clone_of):
    """Return the Elo shifts and reordering after duplicating ``clone_of``."""
    elo_after = bt(names2, w2, g2)
    centred = {a: elo_after[a] - elo_base[a] for a in names}
    anchor = rci.ANCHOR if clone_of != rci.ANCHOR else rci.ANCHOR_FALLBACK
    others = [a for a in names if a != clone_of]
    elo_shift = max(abs(centred[a] - centred[anchor]) for a in others)
    elo_spread = max(centred[a] for a in others) - min(centred[a] for a in others)
    elo_centred_all = max(abs(v) for v in centred.values())
    elo_order_changed = sorted(names, key=lambda a: -elo_base[a]) != sorted(
        names, key=lambda a: -elo_after[a]
    )
    return anchor, elo_shift, elo_spread, elo_centred_all, elo_order_changed


def alpha_rank_cell(base_ar, ar2, i, n, cfg):
    """Return the per-alpha mass shifts, merged total variation and reordering of one cell."""
    per_alpha = {}
    for alpha in cfg["alphas"]:
        pi, pi2 = base_ar[alpha]["pi"], ar2[alpha]["pi"]
        merged = pi2[:n].copy()
        merged[i] += pi2[n]
        floor = cfg["mass_floor"]
        keep = [k for k in range(n) if pi[k] >= floor and merged[k] >= floor]
        unresolved = max(base_ar[alpha]["method_gap"], ar2[alpha]["method_gap"]) > cfg["tolerance"]
        per_alpha[str(alpha)] = {
            "others_max_shift": float(max(abs(pi2[k] - pi[k]) for k in range(n) if k != i)),
            "merged_tv": float(0.5 * np.sum(np.abs(merged - pi))),
            "original_mass_before": float(pi[i]),
            "original_and_clone_after": [float(pi2[i]), float(pi2[n])],
            "clone_split": float(pi2[i] - pi2[n]),
            "order_changed": ordered(pi, keep) != ordered(merged, keep),
            "n_ordered": len(keep),
            "numerically_unresolved": bool(unresolved),
        }
    return per_alpha


def clone_cell(pname, base, clone_of, controls, cfg, nav_cells):
    """Duplicate ``clone_of`` on one panel, update the controls and return the cell record."""
    names, w, g, raw = base["names"], base["w"], base["g"], base["raw"]
    n = len(names)
    i = names.index(clone_of)
    names2, w2, g2 = inject(names, w, g, clone_of)
    P2 = winprob(w2, g2)
    check_repo(P2, controls, cfg["alphas"])
    anchor, elo_shift, elo_spread, elo_centred_all, elo_order_changed = elo_cell(
        names, base["elo"], names2, w2, g2, clone_of
    )
    if pname == "round_robin_8":
        ref = rci.elo_with_clone(raw, clone_of)
        controls["elo_injection_max_diff"] = max(
            controls["elo_injection_max_diff"],
            abs(ref["max_abs_shift"] - elo_shift),
            abs(ref["spread_others"] - elo_spread),
            abs(ref["max_abs_shift_centred_all"] - elo_centred_all),
        )
    if pname == "precision_7" and clone_of in nav_cells:
        controls["nash_averaging_committed_max_diff"] = max(
            controls.get("nash_averaging_committed_max_diff", 0.0),
            abs(nav_cells[clone_of] - elo_shift),
        )
    p2 = nash.nash_equilibrium(P2 - 0.5)
    r2 = (P2 - 0.5) @ p2
    nash_shift = max(abs(r2[k] - base["r"][k]) for k in range(n) if k != i)
    controls["nash_others_max_shift"] = max(controls["nash_others_max_shift"], float(nash_shift))
    per_alpha = alpha_rank_cell(base["ar"], alpha_rank(P2, cfg["alphas"]), i, n, cfg)
    return {
        "clone_of": clone_of,
        "elo": {
            "anchor": anchor,
            "max_shift_others": float(elo_shift),
            "spread_others": float(elo_spread),
            "max_shift_centred_all": float(elo_centred_all),
            "order_changed": bool(elo_order_changed),
        },
        "nash_averaging": {
            "max_rating_shift_others": float(nash_shift),
            "merged_mass_shift": float(abs(p2[i] + p2[n] - base["p_star"][i])),
        },
        "alpha_rank": per_alpha,
    }


def run_panel(pname, loader, controls, cfg, nav_cells):
    """Run every duplication cell of one panel and return its record."""
    alphas = cfg["alphas"]
    names, w, g, raw = loader()
    P = winprob(w, g)
    base_ar = alpha_rank(P, alphas)
    p_star = nash.nash_equilibrium(P - 0.5)
    r_base = (P - 0.5) @ p_star
    elo_base = bt(names, w, g)
    check_repo(P, controls, alphas)
    if pname == "precision_7":
        committed = json.loads(CYCLIC_TRIPLE_JSON.read_text())["alpha_rank_transitive_panel"]
        controls["cyclic_triple_committed_max_diff"] = max(
            float(np.max(np.abs(np.array(committed["stationary"][str(a)]) - base_ar[a]["pi"])))
            for a in alphas
        )
    base = {"names": names, "w": w, "g": g, "raw": raw}
    base.update({"ar": base_ar, "p_star": p_star, "r": r_base, "elo": elo_base})
    cells = [clone_cell(pname, base, clone_of, controls, cfg, nav_cells) for clone_of in names]
    return {
        "agents": names,
        "games_per_pairing": float(np.median(g[g > 0])),
        "base_alpha_rank": {str(a): [float(x) for x in base_ar[a]["pi"]] for a in alphas},
        "base_nash_mass": [float(x) for x in p_star],
        "cells": cells,
    }


def load_settings():
    """Read the [alpha_rank_clones] settings, restoring the alphas to a tuple of floats."""
    cfg = dict(settings(EXPERIMENT))
    cfg["alphas"] = tuple(float(a) for a in cfg["alphas"])
    return cfg


def load_nash_cells():
    """Read nash_averaging's clone cells (duplicated agent -> Elo shift), or exit if missing."""
    if not NASH_AVERAGING_JSON.exists():
        sys.exit(f"{NASH_AVERAGING_JSON} is missing; run {NASH_AVERAGING_COMMAND} first")
    clone_cells = json.loads(NASH_AVERAGING_JSON.read_text())["clone_cells"]
    return {c["duplicated"]: c["elo_max_shift_others"] for c in clone_cells}


def main() -> int:
    """Run every duplication cell on both panels, check the controls and report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="JSON path (the Markdown report goes beside it)"
    )
    args = parser.parse_args()
    if args.json is None:
        args.json = output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    cfg = load_settings()
    nav_cells = load_nash_cells()

    controls = {
        "repo_alpha_rank_max_diff": 0.0,
        "repo_alpha_rank_overflows": 0,
        "elo_injection_max_diff": 0.0,
        "nash_others_max_shift": 0.0,
    }
    loaders = (("round_robin_8", panel_round_robin_8), ("precision_7", panel_precision_7))
    panels = {
        pname: run_panel(pname, loader, controls, cfg, nav_cells) for pname, loader in loaders
    }

    exact = cfg["control_tolerance"]
    ok = {
        "injection_is_proto_clones": controls["elo_injection_max_diff"] < exact,
        "alpha_rank_matches_repo": controls["repo_alpha_rank_max_diff"] < exact,
        "reproduces_cyclic_triple_committed": controls["cyclic_triple_committed_max_diff"] < exact,
        "nash_averaging_invariant_as_proven": controls["nash_others_max_shift"] < cfg["tolerance"],
        "reproduces_nash_averaging_committed": (
            controls["nash_averaging_committed_max_diff"] < cfg["elo_print_precision"]
        ),
    }
    elo_convention = (
        f"max_shift_others: largest |Elo shift| among the agents other than the "
        f"duplicated one, {rci.ANCHOR} pinned ({rci.ANCHOR_FALLBACK} when {rci.ANCHOR} "
        f"is duplicated); spread_others: largest change in any Elo difference "
        f"between two other agents (anchor-free); max_shift_centred_all: the "
        f"mean-centred fit over all originals"
    )
    out = {
        "tool": TOOL,
        "alphas": list(cfg["alphas"]),
        "tolerance": cfg["tolerance"],
        "mass_floor": cfg["mass_floor"],
        "elo_convention": elo_convention,
        "controls": {**controls, **ok},
        "panels": panels,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    md_path = args.json.with_suffix(".md")
    write_markdown(md_path, EXPERIMENT, {**controls, **ok}, panels, cfg["alphas"])

    print_summary({**controls, **ok}, panels, cfg["alphas"])
    print(f"\nwrote {args.json}")
    print(f"wrote {md_path}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
