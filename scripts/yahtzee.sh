#!/usr/bin/env bash
#
# Runs the solitaire Yahtzee experiments. They need no oracle and no checkpoints: the ground
# truth is an exact dynamic program, which each experiment solves first (about two minutes).
# They read the panel's exact evaluations from data/yahtzee/yahtzee_panel.json. Each experiment
# writes its results to outputs/<experiment>/. The whole script takes about one hour on a
# laptop CPU.
#
# Set PYTHON to choose the interpreter (default: python).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

run() {
    echo
    echo "=== $*"
    "$PYTHON" -m "active_oracle_evaluation.experiments.$1" "${@:2}"
}

# Rank recovery of the frozen suite against the exact ordering: on all ten agents, on the paper's
# nine-agent panel (its regret table and selection margin) and on the three near-tied agents.
run yahtzee_rank_recovery
run yahtzee_rank_recovery --preset estimand
run yahtzee_rank_recovery --agents optimal noisy_s0.5 noisy_s1 --tag near_tied

# Rating bias as a function of the reference's quality, with the reference sharing a noise draw
# with the panel and with independent draws, and selection steered by a degraded reference.
run yahtzee_reference_quality
run yahtzee_reference_quality --independent --draws 4
run yahtzee_degraded_steering

# Label efficiency of the selection rules, and the trials each pair needs to resolve.
run yahtzee_label_efficiency
run yahtzee_trials
