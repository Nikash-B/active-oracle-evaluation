#!/usr/bin/env bash
#
# Runs every backgammon analysis that reads only the shipped data in data/ and the small
# checkpoints in models/. None of them needs the oracle or the large checkpoints. Each experiment
# writes its results to outputs/<experiment>/ (or to $OUTPUTS/<experiment>/ when OUTPUTS is set).
# The whole script takes about 30 minutes on a laptop CPU.
#
# Set PYTHON to choose the interpreter (default: python).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT="${OUTPUTS:-outputs}"
SUITES=data/decision_suites

run() {
    echo
    echo "=== $*"
    "$PYTHON" -m "active_oracle_evaluation.experiments.$1" "${@:2}"
}

# Per-decision against outcome scoring on the duplicate-dice measurement pair, at 0 and 2 ply.
run measurement_pair
run measurement_pair --decisions data/measurement_pair/decisions_2ply.npz \
    --out-dir "$OUT/measurement_pair/2ply" \
    --out-json "$OUT/measurement_pair/2ply/measurement_pair_ci.json"

# Tail regret, catastrophic-error rates and the per-phase regret breakdown, at 0 and 2 ply.
run tail_regret
run tail_regret --cache "$SUITES/full_suite_scores_2ply.npz" --out-dir "$OUT/tail_regret/2ply"
run tail_differences
run phase_intervals
run phase_intervals --cache "$SUITES/full_suite_scores_2ply.npz" --tag _2ply
run bearoff_domain

# Rating validity against the head-to-head round-robin.
run cluster_validity
run validity_intervals --elo-only
run equal_games
run transportability --from-json data/round_robins/transportability.json
run nash_averaging
run alpha_rank_clones
run clone_invariance
run lineage_drift
run annotator_depth

# Paired PR contrasts: pubeval against the warm-start network, and the AlphaZero learning curve.
run pair_contrasts

# Label efficiency of the four selection designs at 0 and 2 ply, and its robustness checks.
run label_efficiency --out-csv "$OUT/label_efficiency/label_efficiency_0ply.csv" \
    --out-md "$OUT/label_efficiency/label_efficiency_0ply.md"
run label_efficiency --cache "$SUITES/full_suite_scores_2ply.npz" --seeds 20 \
    --out-csv "$OUT/label_efficiency/label_efficiency_2ply.csv" \
    --out-md "$OUT/label_efficiency/label_efficiency_2ply.md"
run label_efficiency_robustness
run cheap_steering

# Estimator baselines and the coverage of their intervals.
run estimator_baselines
run estimator_baselines --agents pubeval v1 v2 ppo az distilled \
    --out-dir "$OUT/estimator_baselines/six_agent"
run estimator_interval_coverage

# The selected evaluation subsets for each selection design and budget.
run select_positions
