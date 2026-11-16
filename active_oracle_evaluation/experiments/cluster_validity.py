"""Rating validity on agent subsets, with Bradley-Terry Elo refit on each submatrix.

The panel-wide PR-vs-Elo fit includes two weak agents, ``random`` and ``heuristic``, whose
extreme PR and Elo values inflate every correlation. For each subset of agents this experiment
refits Elo on the head-to-head submatrix of the eight-agent round-robin and reports Kendall tau
(with an exact permutation p-value and an Elo-resampling interval), Spearman rho, and the PR-gap
win-probability slope with pair-level and agent-clustered bootstrap intervals. A subset passes
when the agent-clustered slope interval excludes zero and the tau permutation p is below the
threshold in the [cluster_validity] table of config/experiments.toml. The subsets and resampling
settings are in the same table. Each subset also reports a restricted tau: the eight-agent
anchored Elo from the input file, restricted to the subset's agents and compared with PR, together
with the pairs that this Elo order and the refit Elo order rank differently.

Run, with the paper's settings:

    python -m active_oracle_evaluation.experiments.cluster_validity

It writes outputs/cluster_validity/cluster_validity.md and cluster_validity.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.elo import RoundRobin, bradley_terry_elo
from active_oracle_evaluation.eval.validity import (
    agent_bootstrap_slope_ci,
    bootstrap_slope_ci,
    dyadic_robust_slope_se,
    kendall_tau,
    kendall_tau_pvalue,
    pr_gap_winprob_fit,
    spearman_rho,
)
from active_oracle_evaluation.paths import DATA, REPO, output_dir

EXPERIMENT = "cluster_validity"
DEFAULT_IN = DATA / "round_robins" / "elo_validity.json"
# Agent pinned at Elo 0 (paper, appendix "Validity of the oracle rating", paragraph "Anchored
# round-robin").
ANCHOR = "pubeval"


def _subset_matrices(data: dict, names_sub: list[str]):
    """Slice the full ``wins``/``games`` matrices down to ``names_sub`` (preserving its order)."""
    all_names = data["names"]
    idx = np.array([all_names.index(n) for n in names_sub])
    wins = np.asarray(data["wins"], dtype=float)
    games = np.asarray(data["games"], dtype=float)
    wins_sub = wins[np.ix_(idx, idx)]
    games_sub = games[np.ix_(idx, idx)]
    return wins_sub, games_sub


def _refit_elo(names_sub, wins_sub, games_sub) -> dict[str, float]:
    """Bradley–Terry Elo on the submatrix, anchored at pubeval when present."""
    rr = RoundRobin(
        names=list(names_sub),
        games=games_sub,
        wins=wins_sub,
    )
    anchor = ANCHOR if ANCHOR in names_sub else None
    return bradley_terry_elo(rr, anchor_name=anchor)


def _elo_resample_tau_ci(wins_sub, games_sub, names_sub, pr_order, *, reps, percentiles, seed):
    """Return a tau interval from resampling head-to-head outcomes and refitting Elo.

    For each i<j with games, draw ``wins[i][j] ~ Binomial(games[i][j], p_hat)`` (mirror j,i), refit
    Elo, recompute tau against the fixed PR order, and return the given percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(names_sub)
    taus = np.empty(reps)
    g = games_sub
    p_hat = np.divide(wins_sub, g, out=np.zeros_like(wins_sub), where=g > 0)
    for r in range(reps):
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if g[i, j] <= 0:
                    continue
                wij = rng.binomial(int(g[i, j]), p_hat[i, j])
                w[i, j] = wij
                w[j, i] = g[i, j] - wij
        elo = _refit_elo(names_sub, w, g)
        elo_order = sorted(names_sub, key=lambda nm: elo[nm], reverse=True)
        taus[r] = kendall_tau(pr_order, elo_order)
    lo, hi = np.percentile(taus, percentiles)
    return float(lo), float(hi)


def _discordant_pairs(data: dict, order_a: list[str], order_b: list[str]) -> list[dict]:
    """List the pairs that two best-first orders rank differently, with head-to-head wins."""
    names = data["names"]
    pairs = []
    for i, a in enumerate(order_a):
        for b in order_a[i + 1 :]:
            if order_b.index(a) > order_b.index(b):
                ia, ib = names.index(a), names.index(b)
                pairs.append(
                    {
                        "pair": [a, b],
                        "wins": [float(data["wins"][ia][ib]), float(data["wins"][ib][ia])],
                        "games": float(data["games"][ia][ib]),
                    }
                )
    return pairs


def analyze_subset(data: dict, key: str, names_sub: list[str], *, seed: int, cfg: dict) -> dict:
    """Refit Elo on one subset and compute its rank agreement, slope fit and intervals."""
    pr = {n: float(data["pr"][n]) for n in names_sub}
    wins_sub, games_sub = _subset_matrices(data, names_sub)
    elo = _refit_elo(names_sub, wins_sub, games_sub)

    # Orders: best-first. Lower PR = stronger; higher Elo = stronger.
    pr_order = sorted(names_sub, key=lambda nm: pr[nm])
    elo_order = sorted(names_sub, key=lambda nm: elo[nm], reverse=True)

    tau = kendall_tau(pr_order, elo_order)
    # The eight-agent anchored Elo from the input file, restricted to this subset (no refit).
    restricted_order = sorted(names_sub, key=lambda nm: float(data["elo"][nm]), reverse=True)
    tau_restricted = kendall_tau(pr_order, restricted_order)
    rho = spearman_rho(pr_order, elo_order)
    tau_p = kendall_tau_pvalue(pr_order, elo_order, seed=seed)
    slope, r2, n_pairs = pr_gap_winprob_fit(pr, wins_sub, games_sub, names_sub)
    slope_lo, slope_hi = bootstrap_slope_ci(pr, wins_sub, games_sub, names_sub, seed=seed)
    # All pairs are contrasts among the same few agents, so they are not independent;
    # resampling agents gives the wider, agent-clustered interval.
    agent_lo, agent_hi, agent_used = agent_bootstrap_slope_ci(
        pr, wins_sub, games_sub, names_sub, seed=seed
    )
    dyadic_se = dyadic_robust_slope_se(pr, wins_sub, games_sub, names_sub)
    tau_lo, tau_hi = _elo_resample_tau_ci(
        wins_sub,
        games_sub,
        names_sub,
        pr_order,
        reps=cfg["tau_ci_reps"],
        percentiles=cfg["tau_ci_percentiles"],
        seed=seed,
    )

    slope_ci_excludes_zero = slope_lo > 0.0 or slope_hi < 0.0
    agent_ci_excludes_zero = agent_lo > 0.0 or agent_hi < 0.0
    passes = bool(agent_ci_excludes_zero and tau_p < cfg["p_threshold"])

    return {
        "subset": key,
        "names": list(names_sub),
        "n": len(names_sub),
        "pr_order": pr_order,
        "elo_order": elo_order,
        "elo": elo,
        "tau": float(tau),
        "tau_pvalue": float(tau_p),
        "tau_ci": [tau_lo, tau_hi],
        "elo_order_restricted": restricted_order,
        "tau_restricted": float(tau_restricted),
        "refit_vs_restricted_pairs": _discordant_pairs(data, elo_order, restricted_order),
        "spearman_rho": float(rho),
        "slope": float(slope),
        "slope_ci": [slope_lo, slope_hi],
        "slope_ci_excludes_zero": slope_ci_excludes_zero,
        "slope_ci_agent": [agent_lo, agent_hi],
        "slope_ci_agent_excludes_zero": agent_ci_excludes_zero,
        "slope_ci_agent_replicates": agent_used,
        "dyadic_robust_se": float(dyadic_se),
        "r2": float(r2),
        "n_pairs": int(n_pairs),
        "passes": passes,
    }


def _matches_input(all8: dict, data: dict, sig: int) -> bool:
    """Check that the all8 subset matches the input file's tau, slope and R^2 to ``sig`` s.f."""
    targets = {k: float(data[k]) for k in ("tau", "slope", "r2")}
    got = {"tau": all8["tau"], "slope": all8["slope"], "r2": all8["r2"]}
    ok = True
    print(f"all8 refit vs input file ({sig} s.f.):")
    for k, want in targets.items():
        have = got[k]
        match = float(f"{have:.{sig}g}") == float(f"{want:.{sig}g}")
        flag = "PASS" if match else "FAIL"
        ok = ok and match
        print(f"  {k:6s}: got {have:.6g}  want {want:.{sig}g}  [{flag}]")
    print(f"  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def _fmt_ci(lo: float, hi: float, prec: int = 4) -> str:
    """Format an interval as ``[lo, hi]`` at ``prec`` significant figures."""
    return f"[{lo:.{prec}g}, {hi:.{prec}g}]"


def _signed(x: float) -> str:
    """Format a number as an integer with a true minus sign."""
    return f"{x:.0f}".replace("-", "−")


def _report_header(src: Path, data: dict, matches: bool, cfg: dict) -> list[str]:
    """Return the report title, source line and the paragraphs that explain the table."""
    pr, elo = data["pr"], data["elo"]
    sig = cfg["match_sig_figs"]
    sizes = [len(names) for names in cfg["subsets"].values()]
    n_max = max(sizes)
    return [
        "# Cluster-only validity\n",
        f"Source: `{src.relative_to(REPO)}`\n",
        f"The headline PR-vs-Elo validity is fit over all {len(data['names'])} agents, including "
        f"the two weak agents `random` (PR {pr['random']:.0f}, Elo {_signed(elo['random'])}) "
        f"and `heuristic` (PR {pr['heuristic']:.0f}, Elo {_signed(elo['heuristic'])}). "
        "This re-fits Bradley–Terry "
        "Elo on agent subsets and re-tests the PR→strength relationship without those agents.\n",
        f"**The all8 refit matches the input file:** {'PASS' if matches else 'FAIL'} "
        f"(τ≈{data['tau']:.{sig}g}, slope≈{data['slope']:.{sig}g}, "
        f"R²≈{data['r2']:.{sig}g} to {sig} s.f.).\n",
        "**Pass rule:** a subset passes when the agent-clustered slope CI excludes 0 **and** "
        f"the τ permutation p < {cfg['p_threshold']:g}. R² is reported but is not part of "
        "the rule.\n",
        "**Which interval.** A subset's ordered pairs are all pairwise "
        f"contrasts among the *same* handful of agents — with {n_max} agents, each appears in "
        f"{n_max - 1} of the {n_max * (n_max - 1) // 2} unordered pairs — so one agent's PR or "
        "Elo error propagates into all of them and they are "
        "not independent observations. The **agent-clustered** interval (resample agents with "
        "replacement, refit on the pairs among the drawn multiset) accounts for this; "
        "the pair-level interval, shown for comparison, treats the pairs as independent. "
        "A two-way (dyadic) cluster-robust SE is also reported, but with only "
        f"{min(sizes)}–{n_max} clusters and a design where every unordered pair appears twice "
        "with both coordinates negated it is unreliable: it can come out *narrower* than the "
        "pair-level interval.\n",
        "**Which Elo.** τ compares PR with Elo refit on the subset's own head-to-head games; "
        f"τ (restricted) compares PR with the {len(data['names'])}-agent anchored Elo of the "
        "input file, restricted to the subset's agents without a refit.\n",
    ]


def _summary_table(results: list[dict]) -> list[str]:
    """Return the one-row-per-subset Markdown summary table."""
    lines = [
        "| subset | n | τ [perm p] | τ CI | τ (restricted) | ρ | slope [agent-clustered CI] "
        "| pair-level CI | R² | passes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tau_cell = f"{r['tau']:.3f} [p={r['tau_pvalue']:.3g}]"
        tau_ci = _fmt_ci(*r["tau_ci"], prec=3)
        slope_cell = f"{r['slope']:.4f} {_fmt_ci(*r['slope_ci_agent'])}"
        passes = "yes" if r["passes"] else "no"
        lines.append(
            f"| `{r['subset']}` | {r['n']} | {tau_cell} | {tau_ci} | {r['tau_restricted']:.3f} "
            f"| {r['spearman_rho']:.3f} "
            f"| {slope_cell} | {_fmt_ci(*r['slope_ci'])} | {r['r2']:.3f} | **{passes}** |"
        )
    return lines


def _pair_line(p: dict) -> str:
    """Format one refit-vs-restricted pair with its head-to-head wins."""
    a, b = p["pair"]
    wa, wb = p["wins"]
    return f"  - {a} vs {b}: head-to-head {wa:.0f}–{wb:.0f} over {p['games']:.0f} games"


def _subset_detail(r: dict) -> list[str]:
    """Return the detail lines of one subset."""
    pairs = r["refit_vs_restricted_pairs"]
    return [
        f"### `{r['subset']}` (n={r['n']})",
        f"- members: {', '.join(r['names'])}",
        f"- PR order (best→worst): {' > '.join(r['pr_order'])}",
        f"- Elo order (best→worst): {' > '.join(r['elo_order'])}",
        f"- τ = {r['tau']:.4f} (perm p = {r['tau_pvalue']:.4g}), "
        f"resample-Elo τ CI = {_fmt_ci(*r['tau_ci'], prec=4)}",
        f"- restricted Elo order (best→worst): {' > '.join(r['elo_order_restricted'])}",
        f"- τ (restricted) = {r['tau_restricted']:.4f}",
        f"- pairs that the refit and restricted Elo orders rank differently: {len(pairs)}",
        *[_pair_line(p) for p in pairs],
        f"- Spearman ρ = {r['spearman_rho']:.4f}",
        f"- slope = {r['slope']:.5f} per PR-pt, over {r['n_pairs']} ordered pairs among "
        f"{r['n']} agents",
        f"  - **agent-clustered CI = {_fmt_ci(*r['slope_ci_agent'])}** "
        f"(excludes 0: {r['slope_ci_agent_excludes_zero']}; "
        f"{r['slope_ci_agent_replicates']:,} usable replicates)",
        f"  - pair-level CI = {_fmt_ci(*r['slope_ci'])} "
        f"(excludes 0: {r['slope_ci_excludes_zero']})",
        f"  - two-way cluster-robust SE = {r['dyadic_robust_se']:.5f} (see caveat above)",
        f"- R² = {r['r2']:.4f} (not part of the pass rule)",
        f"- **passes: {'yes' if r['passes'] else 'no'}**\n",
    ]


def build_report(results: list[dict], matches: bool, src: Path, data: dict, cfg: dict) -> str:
    """Build the markdown report: summary table plus per-subset detail."""
    lines = _report_header(src, data, matches, cfg)
    lines += _summary_table(results)
    lines.append("")
    lines.append("## Per-subset detail\n")
    for r in results:
        lines += _subset_detail(r)
    return "\n".join(lines)


def _parse_args():
    """Parse the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="inp",
        default=str(DEFAULT_IN),
        help="input round-robin JSON (python -m active_oracle_evaluation.experiments.elo_validity)",
    )
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="output markdown report path (default outputs/cluster_validity/cluster_validity.md)",
    )
    return parser.parse_args()


def main() -> None:
    """Analyze every subset and write the markdown report and JSON."""
    args = _parse_args()
    cfg = settings(EXPERIMENT)

    src = Path(args.inp)
    data = json.loads(src.read_text())
    seed = int(data.get("seed", 0))

    results = [
        analyze_subset(data, key, names, seed=seed, cfg=cfg)
        for key, names in cfg["subsets"].items()
    ]

    all8 = next(r for r in results if r["subset"] == "all8")
    matches = _matches_input(all8, data, cfg["match_sig_figs"])

    report = build_report(results, matches, src, data, cfg)

    if args.out is None:
        out_md = output_dir(EXPERIMENT) / f"{EXPERIMENT}.md"
    else:
        out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")

    out_md.write_text(report)
    out_json.write_text(
        json.dumps(
            {
                "source": str(src.relative_to(REPO)),
                "seed": seed,
                "all8_matches_input": matches,
                "pass_rule": (
                    "agent-clustered slope CI excludes 0 AND tau permutation p < "
                    f"{cfg['p_threshold']:g}"
                ),
                "subsets": results,
            },
            indent=1,
        )
    )

    print(report)
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
