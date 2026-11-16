r"""Whether the PR gap scales with the points-per-game gap across the 28 round-robin pairs.

The efficiency ratio of per-decision scoring over outcome scoring transports to closer pairs when
the ppg gap and the PR gap shrink by the same factor. Both gaps are read off the same games: the
anchored round-robin is replayed (games only, same seed, pair order and dice bank) for the ppg
gap, and the PR gap comes from the round-robin's cached per-decision losses
(``data/round_robins/validity_pr_decision_losses.npz``), whose cluster labels are
``"pairing#trial"``. No oracle is needed.

The experiment reports the ratio PR gap / ppg gap and the log-log slope of one on the other
(slope 1 means proportional), the duplicate-dice variance reduction against closeness, and the
ratio of the two trial-level SDs, ``sd_ratio = sd_pr_trial / paired_sd``, with its summary by pair
group. It reads the measured pair's 2,000-trial SDs from
``outputs/measurement_pair/measurement_pair_ci.json``, written by
``python -m active_oracle_evaluation.experiments.measurement_pair``.

It writes ``transportability.json`` and ``transportability.md`` to ``outputs/transportability/``.
With the paper's settings (the ``[transportability]`` table of ``config/experiments.toml``),
replaying the games:

    python -m active_oracle_evaluation.experiments.transportability

``--from-json`` reads the ppg side (``ppg_gap``, ``paired_sd``, ``unpaired_sd``, ``vr_factor``)
from a saved run of this experiment and recomputes the PR side from the cache; it plays no games:

    python -m active_oracle_evaluation.experiments.transportability \
        --from-json data/round_robins/transportability.json
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.experiments.transportability_report import (
    print_sd_ratio,
    write_markdown,
)
from active_oracle_evaluation.experiments.transportability_stats import (
    build_rows,
    pair_variance_reduction,
    per_trial_pr_diffs,
    pr_gaps_from_cache,
    sd_ratio_summary,
)
from active_oracle_evaluation.paths import DATA, OUTPUTS, output_dir

EXPERIMENT = "transportability"
LOSSES_NPZ = DATA / "round_robins" / "validity_pr_decision_losses.npz"
MEASUREMENT_PAIR_JSON = OUTPUTS / "measurement_pair" / "measurement_pair_ci.json"
PRODUCER = "python -m active_oracle_evaluation.experiments.measurement_pair"
#: Saved ppg-side fields that --from-json reads back.
PPG_FIELDS = ("ppg_gap", "paired_sd", "unpaired_sd", "vr_factor")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help=f"result JSON (default outputs/{EXPERIMENT}/{EXPERIMENT}.json)",
    )
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="read the ppg side from a saved run of this experiment instead of replaying games",
    )
    p.add_argument(
        "--pair-json",
        type=Path,
        default=MEASUREMENT_PAIR_JSON,
        help="measurement-pair difference-CI JSON, for its 2,000-trial SD ratio",
    )
    return p


def replay_ppg_side(config: dict) -> list[tuple[str, dict[str, float]]]:
    """Replay the round-robin's games without scoring; return ``[(pairing, ppg-side fields)]``."""
    from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
    from active_oracle_evaluation.eval import DiceBank, play_duplicate_pair
    from active_oracle_evaluation.eval.agent_specs import resolve_agent
    from active_oracle_evaluation.eval.power import variance_reduction

    encoder = BinarySlotEncoder()
    agents = dict(resolve_agent(s, encoder) for s in config["agent_specs"])
    t0 = time.perf_counter()
    bank = DiceBank(config["seed"])
    out = []
    for (na, a), (nb, b) in itertools.combinations(agents.items(), 2):
        res = play_duplicate_pair(a, b, bank, config["pairs"])
        # Variance reduction: unpaired_sd^2 / (2 * paired_sd^2).
        vr = variance_reduction(res.paired_sd, res.unpaired_sd)
        out.append(
            (
                f"{na}|{nb}",
                {
                    "ppg_gap": float(res.mean_diff),
                    "paired_sd": float(res.paired_sd),
                    "unpaired_sd": float(res.unpaired_sd),
                    "vr_factor": float(vr.factor),
                },
            )
        )
    print(f"replayed {len(out)} pairings (games only) in {time.perf_counter() - t0:.0f}s\n")
    return out


def ppg_side_from_json(path: Path) -> list[tuple[str, dict[str, float]]]:
    """Return the ppg side of a saved run of this experiment, in its saved order."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    return [(r["pairing"], {k: float(r[k]) for k in PPG_FIELDS}) for r in saved["pairs"]]


def warn_pr_drift(rows: list[dict], saved: dict, saved_path: Path) -> None:
    """Print the pairings whose PR gap differs from the saved run's."""
    drift = [
        r["pairing"] for r, s in zip(rows, saved["pairs"]) if abs(r["pr_gap"] - s["pr_gap"]) > 1e-9
    ]
    if drift:
        print(f"  !! PR gap differs from {saved_path.name} for {drift}: different cache?")


def print_gap_table(rows: list[dict]) -> None:
    """Print each pairing's ppg gap, PR gap and their ratio, closest pair first."""
    print(f"{'pairing':<22} {'ppg gap':>9} {'PR gap':>8} {'PR/ppg':>8}")
    for r in sorted(rows, key=lambda r: abs(r["ppg_gap"])):
        print(
            f"  {r['pairing']:<20} {r['ppg_gap']:+9.4f} {r['pr_gap']:+8.2f} "
            f"{r['pr_gap'] / r['ppg_gap']:8.2f}"
        )


def variance_reduction_vs_closeness(rows, prg, pair_json, close_far_group_min) -> dict:
    """Print the duplicate-dice variance reduction against closeness; return its numbers."""
    gap = np.abs(prg)
    vrf = np.array([r["vr_factor"] for r in rows])
    order = np.argsort(gap)
    print("\nvariance reduction vs closeness (PR gap ascending = closest first):")
    for i in order:
        print(f"  {rows[i]['pairing']:<20} PR gap {gap[i]:6.2f}   VR {vrf[i]:5.2f}x")
    k = max(close_far_group_min, len(rows) // 4)
    close, far = vrf[order[:k]], vrf[order[-k:]]
    rho = float(np.corrcoef(np.log(gap), vrf)[0, 1])
    print(
        f"\n  closest {k}: median VR {np.median(close):.2f}x   "
        f"farthest {k}: median {np.median(far):.2f}x"
    )
    print(
        f"  correlation of VR with log(PR gap): r = {rho:+.3f}  "
        "(negative == saving rises as pairs close)"
    )
    pair_vr = pair_variance_reduction(pair_json)
    pair_note = f"   (measurement pair: {pair_vr:.2f}x)"
    print(
        f"  all {len(rows)}: median {np.median(vrf):.2f}x, range [{vrf.min():.2f}, {vrf.max():.2f}]"
        + pair_note
    )
    return {
        "gap": gap,
        "vrf": vrf,
        "order": order,
        "k": k,
        "close": close,
        "far": far,
        "rho": rho,
        "pair_vr": pair_vr,
    }


def proportionality(ppg, prg, measurable_ppg) -> dict:
    """Print the PR/ppg gap ratio and the log-log slope; return their numbers."""
    ratio = prg / ppg
    # On a log-log plot of |PR gap| vs |ppg gap|, slope 1 == proportional.
    ok = (np.abs(ppg) > 1e-9) & (np.abs(prg) > 1e-9)
    slope, _ = np.polyfit(np.log(np.abs(ppg[ok])), np.log(np.abs(prg[ok])), 1)
    corr = float(np.corrcoef(np.log(np.abs(ppg[ok])), np.log(np.abs(prg[ok])))[0, 1])
    print(
        f"\nratio PR-gap / ppg-gap: median {np.median(ratio):.2f}, "
        f"IQR [{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}], "
        f"range [{ratio.min():.2f}, {ratio.max():.2f}]"
    )
    measurable = np.abs(ppg) > measurable_ppg
    ratio_meas = ratio[measurable]
    iqr_meas = [float(v) for v in np.percentile(ratio_meas, [25, 75])]
    print(
        f"  over the {measurable.sum()} measurable pairs (|ppg gap| > {measurable_ppg}): "
        f"median {np.median(ratio_meas):.2f}, IQR [{iqr_meas[0]:.2f}, {iqr_meas[1]:.2f}]"
    )
    print(
        f"log-log slope of |PR gap| on |ppg gap|: {slope:.3f}  "
        f"(1.000 == proportional), r = {corr:.3f}"
    )
    return {
        "ratio": ratio,
        "slope": slope,
        "corr": corr,
        "measurable": measurable,
        "ratio_meas": ratio_meas,
        "iqr_meas": iqr_meas,
    }


def result_json(rows, vr, prop, sds, args, saved, config) -> dict:
    """Return the result JSON of the experiment."""
    ratio = prop["ratio"]
    return {
        "pairs": rows,
        "ratio_median": float(np.median(ratio)),
        "ratio_range": [float(ratio.min()), float(ratio.max())],
        "ratio_median_measurable": float(np.median(prop["ratio_meas"])),
        "ratio_iqr_measurable": prop["iqr_meas"],
        "n_measurable": int(prop["measurable"].sum()),
        "loglog_slope": float(prop["slope"]),
        "loglog_r": prop["corr"],
        "vr_median": float(np.median(vr["vrf"])),
        "vr_vs_loggap_r": vr["rho"],
        "sd_ratio_summary": sds,
        "ppg_side": f"from-json:{args.from_json.name}" if args.from_json else "replay",
        "seed": saved["seed"] if saved else config["seed"],
        "pairs_per_pairing": saved["pairs_per_pairing"] if saved else config["pairs"],
    }


def main() -> None:
    """Compute and print the gap ratio, the variance reduction and the SD ratio."""
    args = build_parser().parse_args()
    config = settings(EXPERIMENT)
    if not args.pair_json.exists():
        sys.exit(f"missing {args.pair_json}; run {PRODUCER} first")

    pr_by_pairing = pr_gaps_from_cache(LOSSES_NPZ, config["pr_per_equity"])
    diffs_by_pairing = per_trial_pr_diffs(LOSSES_NPZ, config["pr_per_equity"])
    print(f"PR gaps: {len(pr_by_pairing)} pairings read from {LOSSES_NPZ.name} (no oracle)")

    if args.from_json is not None:
        ppg_side = ppg_side_from_json(args.from_json)
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        print(f"ppg side: {len(ppg_side)} pairings read from {args.from_json.name} (no replay)\n")
    else:
        ppg_side, saved = replay_ppg_side(config), None
    rows = build_rows(ppg_side, pr_by_pairing, diffs_by_pairing)
    if saved is not None:
        warn_pr_drift(rows, saved, args.from_json)

    ppg = np.array([r["ppg_gap"] for r in rows])
    prg = np.array([r["pr_gap"] for r in rows])
    print_gap_table(rows)
    vr = variance_reduction_vs_closeness(rows, prg, args.pair_json, config["close_far_group_min"])
    prop = proportionality(ppg, prg, config["measurable_ppg"])

    # The other ratio: trial-level SDs, sigma_PR / sigma_ppg.
    sds = sd_ratio_summary(rows, config, args.pair_json)
    print_sd_ratio(rows, sds)

    out_json = args.out_json or output_dir(EXPERIMENT) / f"{EXPERIMENT}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result_json(rows, vr, prop, sds, args, saved, config), indent=1),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    md_path = write_markdown(EXPERIMENT, rows, vr, prop, sds, output_dir(EXPERIMENT))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
