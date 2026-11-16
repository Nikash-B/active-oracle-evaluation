"""Search ablation: the same AlphaZero network scored with and without MCTS at play time.

Both conditions use one network and one frozen decision suite; only MCTS at play time differs. The
no-search condition is the network's row in the score cache: every panel agent is scored as a plain
afterstate agent (one forward pass per legal move). This experiment scores the with-search
condition at one or more simulation counts.

Both conditions see the same decisions, so the contrast is paired per decision. Intervals use the
game-clustered bootstrap, as in the phase_intervals experiment.

Outputs (deterministic, since MCTS runs at temperature 0):
    outputs/search_ablation/search_ablation.csv         paired-contrast rows
    outputs/search_ablation/search_ablation.md          summary tables
    outputs/search_ablation/search_ablation.json        the same numbers
    outputs/search_ablation/az_search_scores_<n>.npz    per-decision regret with n simulations

    python -m active_oracle_evaluation.experiments.search_ablation --sims 100 200
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval import decision_scoring as ds
from active_oracle_evaluation.eval.decision_suite import read_jsonl
from active_oracle_evaluation.eval.score_cache import PR_PER_EQUITY, ScoreCache
from active_oracle_evaluation.eval.tail_risk import (
    bootstrap_p_two_sided,
    game_bootstrap_indices,
    percentile_ci,
    phase_mean_replicates,
)
from active_oracle_evaluation.paths import DATA, MODELS, REPO, output_dir

EXPERIMENT = "search_ablation"
SUITES = DATA / "decision_suites"
DEFAULT_SUITE = SUITES / "backgammon_oracle_suite_0ply_v1.jsonl"
DEFAULT_CACHE = SUITES / "full_suite_scores_0ply.npz"
PHASE_ORDER = ("opening", "race", "bearoff", "contact", "holding", "blitz_backgame")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, with defaults from the [search_ablation] settings."""
    cfg = settings(EXPERIMENT)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        type=Path,
        default=MODELS / "az75.pt",
        help="the network used by both conditions",
    )
    p.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=list(cfg["sims"]),
        help="MCTS simulation counts for the with-search conditions",
    )
    p.add_argument("--n-boot", type=int, default=cfg["n_boot"])
    p.add_argument("--boot-seed", type=int, default=cfg["boot_seed"])
    p.add_argument("--out-dir", type=Path, default=None, help="default: outputs/search_ablation")
    p.add_argument("--doc", type=Path, default=None, help="default: <out-dir>/search_ablation.md")
    return p


def score_with_search(model_path: Path, suite: Path, *, sims: int, c_puct: float) -> np.ndarray:
    """Return per-decision regret of the network with MCTS, aligned to the eligible records."""
    from active_oracle_evaluation.backgammon.nn.mlp import load_checkpoint
    from active_oracle_evaluation.backgammon.search.mcts_agent import MCTSAgent

    encoder = BinarySlotEncoder()
    model = load_checkpoint(str(model_path))
    # MCTSAgent plays the most-visited move, so the condition is deterministic.
    agent = MCTSAgent(model, encoder, simulations=sims, c_puct=c_puct)
    elig = ds.eligible_records(read_jsonl(suite))
    t0 = time.perf_counter()
    losses = np.asarray(ds.score_matrices({"az_search": agent}, elig)[0]["az_search"], dtype=float)
    print(f"  sims={sims}: scored {len(elig):,} decisions in {time.perf_counter() - t0:,.0f}s")
    return losses


def contrast_row(label: str, scope: str, replicates: np.ndarray) -> dict:
    """Return one contrast row: point estimate, percentile interval and two-sided p-value."""
    lo, hi = percentile_ci(replicates[1:])
    return {
        "condition": label,
        "scope": scope,
        "delta": float(replicates[0]),
        "ci_lo": lo,
        "ci_hi": hi,
        "p": bootstrap_p_two_sided(replicates[1:]),
    }


def paired_rows(
    no_search: np.ndarray,
    with_search: np.ndarray,
    phases: np.ndarray,
    indices,
    label: str,
    easy_phases,
    contact_phases,
) -> list[dict]:
    """Return the paired search-minus-no-search contrast overall and per phase, with CIs.

    Negative values mean search reduces regret. The difference is formed per decision before
    resampling.
    """
    diff = with_search - no_search
    order = [ph for ph in PHASE_ORDER if ph in set(phases)]
    reps = phase_mean_replicates(diff, phases, indices, order)
    rows = [contrast_row(label, ph, reps[:, j]) for j, ph in enumerate(order)]

    def group(names) -> np.ndarray:
        """Average the phase-mean replicates over a group of phases."""
        return np.nanmean(np.stack([reps[:, order.index(p)] for p in names]), axis=0)

    rows.append(contrast_row(label, "EASY_PHASES", group(easy_phases)))
    rows.append(contrast_row(label, "CONTACT_PHASES", group(contact_phases)))

    # Overall: a plain mean over every decision, not a mean of phase means.
    overall = np.array([diff[idx].mean() for idx in indices])
    rows.append(contrast_row(label, "ALL", overall))

    # Easy-phase minus contact-phase contrast: positive means search helps more in contact.
    gap = group(easy_phases) - group(contact_phases)
    rows.append(contrast_row(label, "EASY_minus_CONTACT", gap))
    return rows


def _f(x: float, nd: int = 4) -> str:
    """Format a float with nd decimals, or "nan" if it is not finite."""
    return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"


def _rel(path: Path) -> str:
    """Return the path relative to the repository root when possible."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_doc(
    path: Path,
    *,
    model: Path,
    n_dec: int,
    n_games: int,
    n_boot: int,
    boot_seed: int,
    conditions: dict[int, np.ndarray],
    no_search: np.ndarray,
    rows: list[dict],
) -> None:
    """Write the Markdown summary of both conditions and the paired contrasts."""
    base_pr = no_search.mean() * PR_PER_EQUITY
    lines = [
        "# search-vs-value ablation for AlphaZero",
        "",
        f"The same network (`{model.name}`) scored over the same frozen suite with and without "
        "MCTS at play time.",
        "",
        f"- {n_dec:,} decisions from **{n_games} source games**; oracle-scored regret, identical "
        "positions in every condition.",
        "- The no-search condition is the network's row in the score cache: every panel agent is "
        "built by `eval.agent_specs.resolve_agent` as a plain afterstate agent (`k=1`, one "
        "forward pass).",
        "- With-search conditions run MCTS at temperature 0 (most-visited move), so they are "
        "deterministic.",
        f"- Intervals: game-clustered bootstrap, {n_boot:,} replicates, seed {boot_seed}. "
        "Contrasts are paired per decision: both conditions see the same positions.",
        "",
        "## Conditions",
        "",
        "| condition | mean regret | PR |",
        "|---|---|---|",
        f"| no search | {_f(float(no_search.mean()), 6)} | **{base_pr:.1f}** |",
    ]
    for sims in sorted(conditions):
        m = float(conditions[sims].mean())
        lines.append(f"| MCTS, {sims} sims | {_f(m, 6)} | **{m * PR_PER_EQUITY:.1f}** |")
    lines += [
        "",
        "## Paired contrast — search minus no search (negative = search helps)",
        "",
        "| condition | scope | Δ mean regret | 95% CI | p |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | `{r['scope']}` | {_f(r['delta'])} | "
            f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | {r['p']:.4f} |"
        )
    lines += [
        "",
        "`EASY_minus_CONTACT` is positive when search reduces regret more in the contact "
        "phases, near zero when the gain is uniform, and negative when search helps more in "
        "the easy phases.",
        "",
        "Reproduce: `python -m active_oracle_evaluation.experiments.search_ablation --model "
        f"{_rel(model)}` (the scored conditions are cached in `outputs/search_ablation/`).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def write_json(
    path: Path,
    *,
    n_dec: int,
    n_games: int,
    n_boot: int,
    boot_seed: int,
    conditions: dict[int, np.ndarray],
    no_search: np.ndarray,
    rows: list[dict],
) -> None:
    """Write the condition means and the paired-contrast rows as JSON."""
    numbers = {
        "n_decisions": n_dec,
        "n_games": n_games,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "conditions": [
            {
                "condition": "no search",
                "mean_regret": float(no_search.mean()),
                "pr": float(no_search.mean() * PR_PER_EQUITY),
            }
        ]
        + [
            {
                "condition": f"mcts{sims}",
                "simulations": sims,
                "mean_regret": float(conditions[sims].mean()),
                "pr": float(conditions[sims].mean()) * PR_PER_EQUITY,
            }
            for sims in sorted(conditions)
        ],
        "contrasts": rows,
    }
    path.write_text(json.dumps(numbers, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def search_conditions(args, cfg: dict, cache: ScoreCache, out_dir: Path) -> dict[int, np.ndarray]:
    """Score each with-search condition, or reuse its cached per-decision regret."""
    conditions: dict[int, np.ndarray] = {}
    for sims in args.sims:
        out = out_dir / f"az_search_scores_{sims}.npz"
        if out.exists():  # reuse a scored condition instead of re-running MCTS
            conditions[sims] = np.asarray(np.load(out)["losses"], dtype=float)
            print(f"  sims={sims}: reused {out.name}")
            continue
        conditions[sims] = score_with_search(
            args.model,
            DEFAULT_SUITE,
            sims=sims,
            c_puct=cfg["c_puct"],
        )
        np.savez_compressed(
            out,
            losses=conditions[sims],
            position_ids=cache.position_ids,
            phases=cache.phases,
            game_ids=cache.game_ids,
            agent_sources=cache.agent_sources,
            cluster_ids=cache.cluster_ids,
            simulations=sims,
        )
    return conditions


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the paired-contrast rows as CSV, floats with six decimals."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "scope", "delta", "ci_lo", "ci_hi", "p"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path}")


def main() -> None:
    """Score the search conditions, form the paired contrasts and write the outputs."""
    args = build_parser().parse_args()
    cfg = settings(EXPERIMENT)
    no_search_agent = cfg["no_search_agent"]
    cache = ScoreCache(DEFAULT_CACHE)
    if no_search_agent not in cache.losses:
        raise SystemExit(f"{no_search_agent!r} not in {DEFAULT_CACHE}; have {cache.names}")
    no_search = cache.losses[no_search_agent]
    print(
        f"no-search condition: {no_search_agent} PR {no_search.mean() * PR_PER_EQUITY:.1f} "
        f"over {cache.n_pos:,} decisions from {cache.n_games} games"
    )

    out_dir = args.out_dir if args.out_dir is not None else output_dir(EXPERIMENT)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = search_conditions(args, cfg, cache, out_dir)

    indices = [np.arange(cache.n_pos)]
    indices += game_bootstrap_indices(cache.cluster_ids, n_boot=args.n_boot, seed=args.boot_seed)
    rows: list[dict] = []
    for sims in sorted(conditions):
        rows += paired_rows(
            no_search,
            conditions[sims],
            cache.phases,
            indices,
            f"mcts{sims}",
            cfg["easy_phases"],
            cfg["contact_phases"],
        )

    write_csv(out_dir / "search_ablation.csv", rows)

    summary = dict(
        n_dec=cache.n_pos,
        n_games=cache.n_games,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
        conditions=conditions,
        no_search=no_search,
        rows=rows,
    )
    write_doc(args.doc or out_dir / "search_ablation.md", model=args.model, **summary)
    write_json(out_dir / "search_ablation.json", **summary)
    print("done.")


if __name__ == "__main__":
    main()
