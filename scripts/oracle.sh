#!/usr/bin/env bash
#
# Runs the backgammon experiments that call the gnubg-nn oracle or play games with the trained
# agents. They need the [oracle] extra and the checkpoints in models/, including the four large
# ones from the release download. Each experiment writes its results to outputs/<experiment>/
# (or to $OUTPUTS/<experiment>/ when OUTPUTS is set). The whole script takes about four hours on an
# 8-core laptop CPU; the round-robins take most of it.
#
# The score caches are computed on the suite that build_suite writes here. The other experiments
# read the shipped round-robin files in data/round_robins/, so each reproduces its own result
# without depending on an earlier step of this script.
#
# Set PYTHON to choose the interpreter (default: python).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT="${OUTPUTS:-outputs}"
SUITE="$OUT/build_suite/backgammon_oracle_suite_0ply_v1.jsonl"
PANEL="random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt az:models/az75.pt
distilled:models/distilled.pt"

run() {
    echo
    echo "=== $*"
    "$PYTHON" -m "active_oracle_evaluation.experiments.$1" "${@:2}"
}

# Build the frozen decision suite, then score the agent panel and the 18-agent pool on it once,
# the pool also on the suite re-labelled at 2 ply and on the panel suite (read by pair_contrasts).
run build_suite --agents $PANEL --n 2400 --seed 20260628 --plies 0 --out "$SUITE"
run score_panel_cache --suite "$SUITE"
run score_panel_cache --suite "$SUITE" --pool large
run score_panel_cache --suite "$SUITE" --pool large --plies 2 --workers 4
run score_panel_cache --pool large --suite data/decision_suites/panel_suite_0ply_v1.jsonl \
    --out "$OUT/score_panel_cache/panel_suite_scores_0ply_large.npz"

# Panel ratings on the paired suite at 0 and 2 ply, and the AlphaZero learning curve.
run panel_ratings
run panel_ratings --plies 2 --out "$OUT/panel_ratings/panel_ratings_2ply.md" \
    --out-json "$OUT/panel_ratings/panel_ratings_2ply.json" \
    --compare-to "$OUT/panel_ratings/panel_ratings.json"
run agent_ratings --agents az25:models/az25.pt az50:models/az50.pt az75:models/az75.pt \
    pubeval random --pairs 60 --seed 20260628

# Head-to-head reference round-robin, the same round-robin with GNU Backgammon as a ninth player
# and the anchor, its validity intervals, and their transport to the measurement pair
# (transportability reads the measurement-pair result).
run elo_validity
run elo_validity --agents gnubg $PANEL --anchor gnubg0 \
    --out "$OUT/elo_validity/elo_validity_gnubg.md"
run validity_intervals
run measurement_pair
run transportability

# Robustness to the oracle's search depth and to a Monte Carlo playout of the oracle, to search
# at inference time and to exact bear-off.
run oracle_depth --agents $PANEL
run rollout_check
run search_ablation
run bearoff_exact

# The cyclic triple and the oracle-strength round-robin.
run cyclic_triple rr
run cyclic_triple pr
run cyclic_triple analyze
run oracle_pool_elo
