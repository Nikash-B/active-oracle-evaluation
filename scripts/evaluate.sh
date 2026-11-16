#!/usr/bin/env bash
#
# Scores agents on the two frozen decision suites by lookup, with no oracle. Each result is
# printed, and outputs/score_agent/score_agent.md and score_agent.json collect one row per agent
# and suite.
#
#   scripts/evaluate.sh                      # random and heuristic
#   scripts/evaluate.sh v2:models/v2.pt      # any agent spec that score_agent accepts
#
# Set PYTHON to choose the interpreter (default: python).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

if [ "$#" -eq 0 ]; then
    set -- random heuristic
fi

for suite in backgammon_oracle_suite_0ply_v1 panel_suite_0ply_v1; do
    for agent in "$@"; do
        "$PYTHON" -m active_oracle_evaluation.experiments.score_agent \
            --suite "data/decision_suites/${suite}.jsonl" --agent "$agent"
    done
done
