# Active Oracle Evaluation: What Oracle Labels Can and Cannot Measure

## Overview

This repository contains the code and data for Active Oracle Evaluation, which rates an agent by
its mean regret against an oracle on a frozen suite of labelled decisions. It covers backgammon,
with GNU Backgammon as the oracle, and solitaire Yahtzee, where an exact dynamic program gives
the true regret of every agent.

## Installation

```bash
git clone https://github.com/Nikash-B/active-oracle-evaluation
cd active-oracle-evaluation
conda create -n aoe python=3.13 -y && conda activate aoe
pip install -r requirements.txt
curl -L https://github.com/Nikash-B/active-oracle-evaluation/releases/download/v1.0/models_large.tar.gz | tar -xz
```

Everything runs on a laptop CPU; no GPU is needed. The last command downloads four large
checkpoints (about 140 MB) into `models/`, which `scripts/oracle.sh` and the wildbg round-robin
use. The code needs Python 3.12 or 3.13: the pinned numpy and scipy need 3.12, and the oracle
bindings, `gnubg-nn`, ship wheels up to 3.13.

## Reproducing paper results

Every command runs from the repository root.

```bash
# Score agents on the two frozen decision suites by lookup (seconds).
./scripts/evaluate.sh random heuristic v2:models/v2.pt

# Every backgammon analysis of the shipped data (about 30 minutes).
./scripts/analyze.sh

# The solitaire Yahtzee experiments (about one hour).
./scripts/yahtzee.sh

# The backgammon experiments that call the oracle or play games (about four hours).
./scripts/oracle.sh
```

- Each experiment writes its results to `outputs/<experiment>/`, as Markdown tables and a JSON
  file with the same numbers.
- `evaluate.sh`, `analyze.sh` and `yahtzee.sh` read only `data/` and the small checkpoints in
  `models/`. `data/` holds the frozen suites with the oracle's equity for every legal move, the
  per-agent score caches and the round-robin outputs. These scripts need neither the oracle nor
  the large checkpoints.
- Every setting comes from `config/experiments.toml`, where each value has a comment that says
  where it comes from. A command-line flag overrides it.

Three steps are not in the scripts, because they are slow or need another engine:

```bash
# Recompute the Yahtzee panel's exact evaluations, which data/yahtzee/ ships (about 45 minutes).
python -m active_oracle_evaluation.experiments.yahtzee_panel

# Export positions for eXtreme Gammon, then score the agents against its analysis.
python -m active_oracle_evaluation.experiments.export_xg_match
python -m active_oracle_evaluation.experiments.xg_oracle --analysis-xg analysed_suite.xg

# The oracle-strength round-robin with wildbg, which needs a local wildbg server.
python -m active_oracle_evaluation.experiments.oracle_pool_wildbg
```

- `docs/experiments.md` gives the eXtreme Gammon and wildbg procedures and a short description of
  every experiment.

## Tips

- An agent spec is `random`, `heuristic`, `pubeval`, `gnubg:<plies>` or `<name>:<checkpoint>.pt`.
  `pubeval` and `gnubg` play through `gnubg-nn`, so scoring them needs the oracle bindings.
- Scoring a new agent on a frozen suite is a lookup, so it needs no oracle. Pass its spec to
  `scripts/evaluate.sh`.
- Set `OUTPUTS` to write results somewhere other than `outputs/`, and `PYTHON` to choose the
  interpreter that the scripts use.
- `heuristic` and `random` break ties between equal moves at random, so their ratings change
  slightly with the tie-break seed. `panel_ratings` reports them under five other seeds.
- The shipped data and the paper's values come from a Linux machine. On macOS, small
  floating-point differences in the networks' outputs change a few moves: the MCTS conditions of
  `search_ablation` differ by up to 0.4 PR and the cyclic triple's ratings by up to 0.2 PR.

## Acknowledgments

The `puretd/` folder is taken from
[PureTD](https://github.com/alexstrehl/puretd). The oracle is
[GNU Backgammon](https://www.gnu.org/software/gnubg), through its Python bindings
[gnubg-nn](https://github.com/StonesAndDice/gnubg-nn-pypi).
