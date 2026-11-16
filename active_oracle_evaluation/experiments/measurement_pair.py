r"""Per-decision PR difference of the duplicate-dice measurement pair, and its efficiency ratio.

The input is the per-decision oracle loss of both agents over the pair's duplicate-dice trials
(``data/measurement_pair/``). No games are replayed and no oracle is called. A trial is one
duplicate-dice pair of games. Decisions inside one trial share a dice stream and a trajectory, so
the bootstrap draws whole trials with replacement and keeps every decision inside a drawn trial.

From the bootstrap the experiment reports each agent's PR, the PR difference with its 95%
interval, and the per-trial standard deviation of the difference. It then compares the trials
that each scoring needs to resolve its own observed effect, ``m >= (z * sigma / d)^2``.

The per-decision side is computed from the loss caches. The outcome side (the points-per-game
difference of the same games and its paired and unpaired standard deviations) is an input, set
in the ``[measurement_pair]`` table of ``config/experiments.toml``.

AIVAT corrects each game's outcome with the oracle as its value function. The per-game
corrections and raw points of the same games are in
``data/measurement_pair/aivat_corrections_0ply.npz``. When that file covers the same games and
oracle depth as the loss cache, the experiment also reports AIVAT's variance ratio over
duplicate-dice outcomes and the trials AIVAT needs relative to per-decision scoring. Both
outcome estimators target the same quantity, so AIVAT's trials are taken at the duplicate-dice
outcome gap.

The experiment writes ``measurement_pair_ci.json``, ``measurement_pair.md`` and
``measurement_pair.json`` to ``outputs/measurement_pair/``. With the paper's settings:

    python -m active_oracle_evaluation.experiments.measurement_pair \
        --decisions data/measurement_pair/decisions_0ply.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.paths import DATA, output_dir
from active_oracle_evaluation.results import write_results

EXPERIMENT = "measurement_pair"
PR_PER_POINT = 500.0
DEFAULT_DECISIONS = DATA / "measurement_pair" / "decisions_0ply.npz"
DEFAULT_AIVAT = DATA / "measurement_pair" / "aivat_corrections_0ply.npz"
CI_JSON_NAME = "measurement_pair_ci.json"


def build_parser(config) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS,
        help="per-decision loss cache of the pair",
    )
    parser.add_argument(
        "--aivat",
        type=Path,
        default=DEFAULT_AIVAT,
        help="per-game AIVAT corrections and raw points of the pair's games",
    )
    parser.add_argument("--n-boot", type=int, default=config["n_boot"], help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=config["seed"], help="bootstrap seed")
    parser.add_argument(
        "--outcome-diff",
        type=float,
        default=config["outcome_diff"],
        help="outcome mean diff (ppg)",
    )
    parser.add_argument(
        "--outcome-sd",
        type=float,
        default=config["outcome_sd"],
        help="outcome paired SD per trial",
    )
    parser.add_argument(
        "--outcome-sd-unpaired",
        type=float,
        default=config["outcome_sd_unpaired"],
        help="outcome unpaired SD per game, for the duplicate-dice variance reduction",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory of the outputs (default outputs/measurement_pair)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help=f"path of the interval JSON (default <out-dir>/{CI_JSON_NAME})",
    )
    return parser


def percentile_ci(x: np.ndarray, percentiles) -> tuple[float, float]:
    """Return the lower and upper percentiles of ``x``."""
    low, high = percentiles
    return float(np.percentile(x, low)), float(np.percentile(x, high))


def trial_bootstrap(trial, loss, is_a, is_b, n_boot, seed) -> np.ndarray:
    """Return (PR_a, PR_b, PR_a - PR_b) for each bootstrap replicate over whole trials."""
    trials = np.unique(trial)
    n_trials = len(trials)
    order = np.argsort(trial, kind="stable")
    starts = np.searchsorted(trial[order], trials)
    ends = np.append(starts[1:], len(order))
    groups = [order[s:e] for s, e in zip(starts, ends)]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, n_trials, size=n_trials)])
        loss_a, loss_b = loss[idx][is_a[idx]], loss[idx][is_b[idx]]
        pr_a = loss_a.mean() * PR_PER_POINT
        pr_b = loss_b.mean() * PR_PER_POINT
        reps[i] = (pr_a, pr_b, pr_a - pr_b)
    return reps


def load_pair(path: Path):
    """Return the agent ids, trial ids, losses and agent labels of a loss cache."""
    data = np.load(path, allow_pickle=True)
    agent = np.asarray([str(x) for x in data["agent"]])
    trial = np.asarray(data["trial"], dtype=np.int64)
    loss = np.asarray(data["loss"], dtype=float)
    labels = [str(x) for x in data["labels"]]
    return agent, trial, loss, labels


def same_games(aivat_path: Path, decisions_path: Path) -> bool:
    """Return whether the AIVAT file and the loss cache cover the same games and oracle depth."""
    aivat, decisions = np.load(aivat_path), np.load(decisions_path, allow_pickle=True)
    n_pairs = int(decisions["n_pairs"][0])
    return (
        int(aivat["seed"][0]) == int(decisions["seed"][0])
        and int(aivat["plies"][0]) == int(decisions["plies"][0])
        and aivat["raw_points"].shape[0] == n_pairs
    )


def aivat_ratios(aivat_path: Path, m_decision: float, z: float) -> dict:
    """Return AIVAT's mean and SD, its variance ratio and its trials relative to per-decision."""
    data = np.load(aivat_path)
    raw = np.asarray(data["raw_points"], dtype=float)
    corrected = raw - np.asarray(data["corrections"], dtype=float)
    raw_trials, aivat_trials = raw.mean(axis=1), corrected.mean(axis=1)
    sd_raw, sd_aivat = float(raw_trials.std(ddof=1)), float(aivat_trials.std(ddof=1))
    gap = abs(float(raw_trials.mean()))
    m_aivat = (z * sd_aivat / gap) ** 2
    return {
        "duplicate_mean_ppg": float(raw_trials.mean()),
        "duplicate_sd_per_trial": sd_raw,
        "aivat_mean_ppg": float(aivat_trials.mean()),
        "aivat_sd_per_trial": sd_aivat,
        "variance_ratio_duplicate_over_aivat": (sd_raw / sd_aivat) ** 2,
        "m_trials_aivat": m_aivat,
        "trials_ratio_aivat_over_per_decision": m_aivat / m_decision,
    }


def analyse(args, config) -> dict:
    """Compute the point estimates, bootstrap intervals and trial counts, printing each."""
    agent, trial, loss, labels = load_pair(args.decisions)
    label_a, label_b = labels[0], labels[1]

    n_trials = len(np.unique(trial))
    is_a, is_b = agent == "a", agent == "b"
    pr_a, pr_b = loss[is_a].mean() * PR_PER_POINT, loss[is_b].mean() * PR_PER_POINT
    print(
        f"{len(loss):,} decisions over {n_trials:,} trials "
        f"({label_a} {int(is_a.sum()):,} / {label_b} {int(is_b.sum()):,})"
    )
    print(
        f"point estimates: PR({label_a}) = {pr_a:.1f}   PR({label_b}) = {pr_b:.1f}   "
        f"diff = {pr_a - pr_b:+.2f}"
    )

    reps = trial_bootstrap(trial, loss, is_a, is_b, args.n_boot, args.seed)
    percentiles = config["interval_percentiles"]
    ci_a, ci_b, ci_diff = (percentile_ci(reps[:, k], percentiles) for k in range(3))
    se_diff = float(np.std(reps[:, 2], ddof=1))
    sd_trial = se_diff * np.sqrt(n_trials)  # SE = sigma / sqrt(trials)
    diff = pr_a - pr_b
    z = config["z"]
    m_decision = (z * sd_trial / abs(diff)) ** 2
    m_outcome = (z * args.outcome_sd / abs(args.outcome_diff)) ** 2
    aivat = None
    if same_games(args.aivat, args.decisions):
        aivat = aivat_ratios(args.aivat, m_decision, z)
    else:
        print("AIVAT: the corrections cover other games than this loss cache; skipped")
    return {
        "n_decisions": len(loss),
        "n_trials": n_trials,
        "labels": labels,
        "pr_a": pr_a,
        "pr_b": pr_b,
        "diff": diff,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "ci_diff": ci_diff,
        "se_diff": se_diff,
        "sd_trial": sd_trial,
        "m_decision": m_decision,
        "m_outcome": m_outcome,
        "aivat": aivat,
    }


def print_report(r: dict, args) -> None:
    """Print the bootstrap intervals and the trials each scoring needs."""
    label_a, label_b = r["labels"][0], r["labels"][1]
    pr_a, pr_b, diff = r["pr_a"], r["pr_b"], r["diff"]
    ci_a, ci_b, ci_diff = r["ci_a"], r["ci_b"], r["ci_diff"]
    se_diff, sd_trial = r["se_diff"], r["sd_trial"]
    print(f"\n95% trial-clustered bootstrap ({args.n_boot:,} replicates, seed {args.seed}):")
    print(f"  PR({label_a})      {pr_a:6.2f}  [{ci_a[0]:.2f}, {ci_a[1]:.2f}]")
    print(f"  PR({label_b})      {pr_b:6.2f}  [{ci_b[0]:.2f}, {ci_b[1]:.2f}]")
    print(f"  difference   {diff:+6.2f}  [{ci_diff[0]:+.2f}, {ci_diff[1]:+.2f}]   SE {se_diff:.3f}")
    print(f"  -> the gap is {abs(diff) / se_diff:.1f} SE from zero")
    print(f"  per-trial SD of the difference: {sd_trial:.2f} PR   (a trial = 2 games)")

    print("\ntrials needed to resolve each side's own observed effect, m >= (z*sigma/d)^2:")
    print(
        f"  per decision : sigma {sd_trial:.2f} PR,  d {abs(diff):.2f} PR   "
        f"-> m = {r['m_decision']:.1f} trials"
    )
    print(
        f"  game outcome : sigma {args.outcome_sd:.4f} ppg, d {abs(args.outcome_diff):.4f} ppg"
        f" -> m = {r['m_outcome']:.1f} trials"
    )
    print(
        f"  efficiency ratio (outcome / per-decision): {r['m_outcome'] / r['m_decision']:.1f}x "
        "fewer trials per decision"
    )
    if r["aivat"] is not None:
        a = r["aivat"]
        print("\nAIVAT-corrected outcome, oracle as value function:")
        print(
            f"  duplicate dice: mean {a['duplicate_mean_ppg']:+.4f} ppg, "
            f"SD {a['duplicate_sd_per_trial']:.4f};  AIVAT: mean {a['aivat_mean_ppg']:+.4f} ppg, "
            f"SD {a['aivat_sd_per_trial']:.4f}"
        )
        print(
            f"  variance ratio (duplicate dice / AIVAT): "
            f"{a['variance_ratio_duplicate_over_aivat']:.2f}x"
        )
        print(
            f"  AIVAT at the duplicate-dice gap: m = {a['m_trials_aivat']:.1f} trials, "
            f"{a['trials_ratio_aivat_over_per_decision']:.2f}x the per-decision trials"
        )


def summary_numbers(r: dict, args) -> dict:
    """Return the JSON summary of the interval and the trial counts."""
    labels = r["labels"]
    label_a, label_b = labels[0], labels[1]
    n_trials = r["n_trials"]
    m_decision, m_outcome = r["m_decision"], r["m_outcome"]
    out = {"decisions": int(r["n_decisions"]), "n_trials": int(n_trials)}
    out.update(
        {
            "labels": labels,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "pr": {label_a: r["pr_a"], label_b: r["pr_b"]},
            "diff": r["diff"],
            "ci": {label_a: list(r["ci_a"]), label_b: list(r["ci_b"]), "diff": list(r["ci_diff"])},
            "se_diff": r["se_diff"],
            "sd_per_trial": float(r["sd_trial"]),
            "outcome_diff_ppg": float(args.outcome_diff),
            "outcome_sd_paired_per_trial": float(args.outcome_sd),
            "outcome_sd_unpaired_per_game": float(args.outcome_sd_unpaired),
            "outcome_variance_reduction": float(
                args.outcome_sd_unpaired**2 / (2.0 * args.outcome_sd**2)
            ),
            "outcome_z": float(abs(args.outcome_diff) / (args.outcome_sd / n_trials**0.5)),
            "sigma_units": "PR points per duplicate-dice trial (2 games)",
            "m_trials_per_decision": float(m_decision),
            "m_trials_game_outcome": float(m_outcome),
            "efficiency_ratio": float(m_outcome / m_decision),
        }
    )
    if r["aivat"] is not None:
        out["aivat"] = r["aivat"]
    return out


def result_tables(r: dict, args) -> list:
    """Return the Markdown result tables, with numbers formatted as printed."""
    label_a, label_b = r["labels"][0], r["labels"][1]
    ci_a, ci_b, ci_diff = r["ci_a"], r["ci_b"], r["ci_diff"]
    diff, se_diff, sd_trial = r["diff"], r["se_diff"], r["sd_trial"]
    pr_rows = [
        [f"PR({label_a})", f"{r['pr_a']:.2f}", f"[{ci_a[0]:.2f}, {ci_a[1]:.2f}]", "–"],
        [f"PR({label_b})", f"{r['pr_b']:.2f}", f"[{ci_b[0]:.2f}, {ci_b[1]:.2f}]", "–"],
        [
            "difference",
            f"{diff:+.2f}",
            f"[{ci_diff[0]:+.2f}, {ci_diff[1]:+.2f}]",
            f"{se_diff:.3f}",
        ],
    ]
    trial_rows = [
        ["per decision", f"{sd_trial:.2f} PR", f"{abs(diff):.2f} PR", f"{r['m_decision']:.1f}"],
        [
            "game outcome",
            f"{args.outcome_sd:.4f} ppg",
            f"{abs(args.outcome_diff):.4f} ppg",
            f"{r['m_outcome']:.1f}",
        ],
    ]
    summary_rows = [
        ["decisions", f"{r['n_decisions']:,}"],
        ["trials", f"{r['n_trials']:,}"],
        ["gap in SE from zero", f"{abs(diff) / se_diff:.1f}"],
        ["per-trial SD of the difference (PR, a trial = 2 games)", f"{sd_trial:.2f}"],
        ["efficiency ratio (outcome / per-decision)", f"{r['m_outcome'] / r['m_decision']:.1f}x"],
    ]
    if r["aivat"] is not None:
        a = r["aivat"]
        trial_rows.append(
            [
                "AIVAT outcome (at the duplicate-dice gap)",
                f"{a['aivat_sd_per_trial']:.4f} ppg",
                f"{abs(a['duplicate_mean_ppg']):.4f} ppg",
                f"{a['m_trials_aivat']:.1f}",
            ]
        )
        summary_rows += [
            ["AIVAT mean (ppg)", f"{a['aivat_mean_ppg']:+.4f}"],
            [
                "variance ratio (duplicate-dice outcome / AIVAT)",
                f"{a['variance_ratio_duplicate_over_aivat']:.2f}x",
            ],
            [
                "trials ratio (AIVAT / per-decision)",
                f"{a['trials_ratio_aivat_over_per_decision']:.2f}x",
            ],
        ]
    pr_title = f"95% trial-clustered bootstrap ({args.n_boot:,} replicates, seed {args.seed})"
    return [
        (pr_title, ["Quantity", "Estimate", "95% interval", "SE"], pr_rows),
        (
            "Trials needed to resolve each side's own observed effect, m >= (z*sigma/d)^2",
            ["Scoring", "sigma", "d", "m (trials)"],
            trial_rows,
        ),
        ("Summary", ["Quantity", "Value"], summary_rows),
    ]


def main() -> None:
    """Bootstrap the pair's PR difference and print the trials each scoring needs."""
    config = settings(EXPERIMENT)
    args = build_parser(config).parse_args()
    result = analyse(args, config)
    print_report(result, args)

    out_dir = output_dir(EXPERIMENT) if args.out_dir is None else args.out_dir
    out_json = out_dir / CI_JSON_NAME if args.out_json is None else args.out_json
    numbers = summary_numbers(result, args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(f"\nwrote {out_json}")
    md_path, json_path = write_results(
        EXPERIMENT, result_tables(result, args), numbers, out_dir=out_dir
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
