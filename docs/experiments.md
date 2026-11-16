# Experiments

Every experiment is a module in `active_oracle_evaluation/experiments/`. Run one from the
repository root with `python -m active_oracle_evaluation.experiments.<name>`; `--help` lists its
flags. Each experiment takes its settings from its table in `config/experiments.toml` and writes
its results to `outputs/<name>/`.

## The experiments

The "Script" column names the run script that runs the experiment with the paper's settings.

- **analyze**: `scripts/analyze.sh`. It needs only `data/` and the checkpoints that ship in
  `models/`.
- **yahtzee**: `scripts/yahtzee.sh`. It needs no oracle and no checkpoints.
- **oracle**: `scripts/oracle.sh`. It needs the `gnubg-nn` oracle. Experiments marked † also need
  the four large checkpoints from the release download.
- **evaluate**: `scripts/evaluate.sh`.
- **manual**: run by hand; see the README and the sections below.

| Experiment | Script | What it measures |
| --- | --- | --- |
| `agent_ratings` | oracle | Performance Rating (PR) and error per move of each agent on its own games. |
| `alpha_rank_clones` | analyze | Whether alpha-rank and Nash averaging change when an agent of the panel is duplicated. |
| `annotator_depth` | analyze | Rank agreement of the panel when one engine annotates at 0, 1, 2 and 3 ply. |
| `bearoff_domain` | analyze | How many suite decisions fall inside the exact bear-off table's domain. |
| `bearoff_exact` † | oracle | The oracle-regret rating against exact values on the solved bear-off slice. |
| `build_suite` | oracle | The frozen decision suite, from seeded self-play, labelled by the oracle. |
| `cheap_steering` | analyze | Label efficiency of `active_rank` under different design sources. |
| `clone_invariance` | analyze | Whether PR, head-to-head Elo and the selection designs stay stable when agents are duplicated. |
| `cluster_validity` | analyze | Rating validity on agent subsets, with Elo refitted on each subset and with the eight-agent anchored Elo restricted to each subset. |
| `cyclic_triple` † | oracle | Ratings on a constructed cyclic triple of agents. |
| `elo_validity` | oracle | The head-to-head round-robin: anchored Elo against oracle PR, also with GNU Backgammon as a ninth player and the anchor. |
| `equal_games` | analyze | How many agent pairs PR and Elo each resolve at an equal number of games, and how many head-to-head win rates a Wilson 95% interval separates. |
| `estimator_baselines` | analyze | Estimator baselines at a matched label budget. |
| `estimator_interval_coverage` | analyze | Per-agent coverage of the estimator baselines' 95% intervals. |
| `export_xg_match` | manual | The suite's source games as one Jellyfish `.mat` file for eXtreme Gammon. |
| `label_efficiency` | analyze | Label efficiency of the four selection designs over budgets and draws, with draw counts and paired margins against uniform sampling. |
| `label_efficiency_robustness` | analyze | Two robustness checks of the label-efficiency result. |
| `lineage_drift` | analyze | How much of the V2 warm start survives in the AlphaZero and PPO checkpoints. |
| `measurement_pair` | analyze | The duplicate-dice measurement pair: the trials that outcome, AIVAT-corrected outcome and per-decision scoring each need. |
| `nash_averaging` | analyze | Agreement between PR and Nash averaging on the round-robin panel. |
| `oracle_depth` | oracle | Sensitivity of the agent ranking to the oracle's search depth. |
| `oracle_pool_elo` † | oracle | The nine-agent anchored-Elo round-robin with the oracle in the pool. |
| `oracle_pool_wildbg` † | manual | The same round-robin with wildbg added as a tenth agent. |
| `pair_contrasts` | analyze | Paired PR contrasts with bootstrap intervals: pubeval against the warm-start network, and the AlphaZero learning curve, at 0 and 2 ply. |
| `panel_ratings` | oracle | The agent-panel PR table on the paired panel suite, with its Kendall tau against anchored Elo and, with `--compare-to`, against another run's order, and each agent's PR under other tie-break draws. |
| `phase_intervals` | analyze | Game-clustered intervals for the per-phase regret results. |
| `rollout_check` | oracle | Agreement of the agent ranking under a Monte Carlo playout of the oracle's network with the 0-ply and 2-ply rankings. |
| `score_agent` | evaluate | PR of any agent on a frozen decision suite, by lookup. |
| `score_panel_cache` | oracle | The per-agent score cache of a decision suite, optionally re-labelled at a deeper oracle depth. |
| `search_ablation` | oracle | The same AlphaZero network scored with and without MCTS. |
| `select_positions` | analyze | The selected evaluation subsets for each selection design and budget. |
| `tail_differences` | analyze | Paired intervals on differences in tail statistics between agents. |
| `tail_regret` | analyze | Catastrophic-error rates and the per-phase regret of every agent. |
| `transportability` | analyze, oracle | Whether the PR gap scales with the points-per-game gap across the round-robin pairs. |
| `validity_intervals` | analyze, oracle | 95% intervals for both columns of the validity table. `--elo-only` needs no oracle. |
| `xg_oracle` | manual | The agent ranking under eXtreme Gammon against the ranking under GNU Backgammon. |
| `yahtzee_degraded_steering` | yahtzee | Regret-importance selection steered by a degraded reference. |
| `yahtzee_label_efficiency` | yahtzee | Label efficiency against the exact answer: estimator error against estimand error. |
| `yahtzee_panel` | manual | The exact evaluation of every Yahtzee panel agent, which `data/yahtzee/` ships. |
| `yahtzee_rank_recovery` | yahtzee | Whether the suite rating recovers the exact Yahtzee ranking, its gap to the true regret, and the label margin of disagreement over uniform sampling. |
| `yahtzee_reference_quality` | yahtzee | Rating bias as a function of the reference's quality, with shared and with independent noise draws. |
| `yahtzee_trials` | yahtzee | Trials to resolve a skill difference: outcome scoring against per-decision scoring. |

`transportability` in `analyze.sh` reads the game side of the round-robin from
`data/round_robins/transportability.json`; in `oracle.sh` it replays the games.

## Agent labels

The data files and the outputs name agents by short labels. A pair label such as `pubeval|ppo` or
`pubeval - az` names the two agents of a pairing or a difference.

**The eight-agent panel.** These labels appear in the score caches and the round-robin files.

| Label | Agent |
| --- | --- |
| `random` | Uniform over legal moves. |
| `heuristic` | Pip difference, with bonuses for hitting and penalties for leaving blots. |
| `pubeval` | Tesauro's public-domain linear evaluator. The anchored Elo fits pin it at 0. |
| `v1` | The scalar-head TD(λ) agent. |
| `v2` | The outcome-head TD(λ) agent (win, gammon and backgammon probabilities). |
| `ppo` | The PPO agent. It starts from the warm-start network `v2init`. |
| `az` | The AlphaZero agent. It is the same checkpoint as `az75`. |
| `distilled` | The distilled agent. It is the same checkpoint as `dxlarge`. |

**The 18-agent pool.** The large score cache, `full_suite_scores_0ply_large.npz`, holds the panel
plus ten near-duplicates of the same lineage. In it the panel's AlphaZero and distilled agents
appear as `az75` and `dxlarge`.

| Label | Agent |
| --- | --- |
| `az25`, `az50`, `az75` | AlphaZero checkpoints along training, in order. |
| `v2init` | The warm-start network: the `v2` network with a policy head added. |
| `newbest` | A later outcome-head TD(λ) checkpoint, not in the panel. |
| `dxlarge` | The panel's distilled agent, trained on the largest distillation corpus. |
| `dlarge`, `dphasecap` | Distilled students on a smaller corpus and on a corpus capped per game phase. |
| `rblended`, `rdisag`, `rscaled` | Students trained on corpora relabelled by rollouts: blended by confidence, chosen by disagreement, and scaled up. |
| `searchonly` | A student trained on 2-ply search labels only, the control for the rollout students. |

**The oracle-strength pool.** These labels appear in `oracle_pool_elo.json`,
`oracle_pool_elo_wildbg.json` and the outputs of `annotator_depth`.

| Label | Agent |
| --- | --- |
| `az`, `v2net` | The panel's AlphaZero and outcome-head TD(λ) agents. |
| `gnubg0`, `gnubg1`, `gnubg2` | GNU Backgammon as a player at 0, 1 and 2 ply. `gnubg0` is the oracle that labels the suite. |
| `search2` | A 2-ply search over the strongest network, `models/champion.pt`, which is not in the panel. |
| `sel3` | The same search with a selective third ply. |
| `puretd` | A third-party TD network, played at 0 ply through `puretd/`. |
| `wildbg` | wildbg, an independent open-source engine. |

**Other labels.**

| Label | Where | Meaning |
| --- | --- | --- |
| `td`, `az` | `data/measurement_pair/` | The measurement pair: the outcome-head TD(λ) agent and the AlphaZero agent. |
| `cyc0`, `cyc1`, `cyc2` | `cyclic_triple` | Three agents that share one network and play at full strength only against a designated opponent. |
| `strong_ref`, `weak_ref` | `cyclic_triple` | The same network played always at full strength and always weakened. |
| `optimal` | Yahtzee | The optimal policy. |
| `noisy_s0.5` to `noisy_s50` | Yahtzee | The optimal value table perturbed by Gaussian noise of scale 0.5, 1, 2, 5, 15 and 50. |
| `greedy`, `heuristic` | Yahtzee | The myopic greedy agent and a rule-based heuristic. |
| `fixed_order` | Yahtzee | Fills categories in a fixed order. The nine-agent regret table (`--preset estimand`) omits it. |

## eXtreme Gammon

eXtreme Gammon is a commercial engine with its own interface, so this check has one manual step.

1. Export the suite's source games:
   `python -m active_oracle_evaluation.experiments.export_xg_match`. It writes
   `outputs/export_xg_match/suite.mat`. To test the import first, export a two-game file with
   `--n 2`.
2. Import the `.mat` file into eXtreme Gammon. Run Analyze Session at the World Class setting and
   save the result as an `.xg` file.
3. Score the agents against that analysis:
   `python -m active_oracle_evaluation.experiments.xg_oracle --analysis-xg <file>.xg`. Reading
   the file needs the `[xg]` extra (`xgdatatools`).

## wildbg

`oracle_pool_wildbg` plays wildbg through `active_oracle_evaluation/eval/wildbg_agent.py`. At
each unforced decision the agent sends the dice and the board to wildbg's HTTP server
(`GET /move`, with no match score, so wildbg plays for money equity). It plays wildbg's top move,
and raises `WildbgParityError` unless that move maps onto exactly one legal afterstate.

1. Install a Rust toolchain with `rustup`. Clone
   [wildbg](https://github.com/carsten-wenderdel/wildbg), check out commit `2954391` and run
   `cargo build --release`.
2. Start `target/release/server` and leave it running. The agent spec `wildbg` connects to
   `http://localhost:8080`; `wildbg:<url>` connects elsewhere.
3. Run `python -m active_oracle_evaluation.experiments.oracle_pool_wildbg`. It adds wildbg to
   the nine-agent round-robin in `data/round_robins/oracle_pool_elo.json`, so it plays only
   wildbg's pairings. It needs the oracle and the four large checkpoints.
