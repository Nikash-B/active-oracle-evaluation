"""Rank the agent panel under eXtreme Gammon (XG) and under GNU Backgammon.

XG has no headless mode. Import the source games (the ``export_xg_match`` experiment) into XG,
run one Analyze Session at the World Class setting, and save the ``.xg`` file. This experiment
reads every checker decision XG analyzed in that file (``MoveEntry``: pre-move board
``PositionI``, ``Dice``, and each candidate's ``PosPlayed`` and ``Eval``), rebuilds the position
(``xg_pos_to_board``), and scores each agent under XG and under GNU Backgammon at 0-ply and
2-ply on those positions. It reports each agent's PR under the three oracles and the Kendall tau
between the three rankings. XG stores each candidate in the mover's frame, so each decision
pairs XG candidates to our legal afterstates with whichever of the identity and mirror
encodings matches more of them. Reading the ``.xg`` file needs the ``xgdatatools`` package.

With the paper's settings (the [xg_oracle] table of config/experiments.toml):

    python -m active_oracle_evaluation.experiments.xg_oracle \
        --agents random heuristic pubeval v1:models/v1.pt v2:models/v2.pt ppo:models/ppo.pt \
            az:models/az75.pt distilled:models/distilled.pt \
        --analysis-xg suite60_full.xg

The results go to outputs/xg_oracle/xg_oracle.md and xg_oracle.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_oracle_evaluation.backgammon.core.game import Game
from active_oracle_evaluation.backgammon.core.moves import legal_moves
from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder
from active_oracle_evaluation.config import settings
from active_oracle_evaluation.eval.agent_specs import resolve_agent
from active_oracle_evaluation.eval.oracle import GnubgOracle, OracleConfig
from active_oracle_evaluation.eval.validity import kendall_tau
from active_oracle_evaluation.eval.xg_oracle import (
    board_to_xg_pos,
    cubeless_equity_from_xg_eval,
    mirror_xg_pos,
    xg_pos_to_board,
)
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "xg_oracle"


def _decisions_from_xg(path: str, min_candidates: int):
    """Yield ``(board, dice, {afterstate: xg_mover_equity})`` for each analyzed checker decision."""
    from xgdatatools import xgimport, xgstruct

    imp = xgimport.Import(str(path))
    version = -1
    for seg in imp.getfilesegment():
        if seg.type != xgimport.Import.Segment.XG_GAMEFILE:
            continue
        seg.fd.seek(0)
        while True:
            rec = xgstruct.GameFileRecord(version=version).fromstream(seg.fd)
            if rec is None:
                break
            if isinstance(rec, xgstruct.HeaderMatchEntry):
                version = rec.Version
            elif isinstance(rec, xgstruct.MoveEntry):
                out = _one_decision(rec, min_candidates)
                if out is not None:
                    yield out


def _one_decision(rec, min_candidates: int):
    """Pair one XG ``MoveEntry``'s candidates to our legal afterstates, or return None."""
    position, dice_raw = rec.get("PositionI"), rec.get("Dice")
    data_moves = rec.get("DataMoves")
    if position is None or dice_raw is None or not data_moves:
        return None
    dice = tuple(int(x) for x in dice_raw)
    if not (len(dice) == 2 and 1 <= dice[0] <= 6 and 1 <= dice[1] <= 6):
        return None
    board = xg_pos_to_board(tuple(int(x) for x in position))
    afters = [m.afterstate for m in legal_moves(board, dice)]
    if len(afters) < min_candidates:
        return None
    my_keys = {board_to_xg_pos(a): a for a in afters}
    n_moves = int(data_moves.get("NMoves") or 0)
    played = data_moves.get("PosPlayed") or ()
    evals = data_moves.get("Eval") or ()
    cands = []  # (identity_key, mirror_key, mover_equity)
    for i in range(min(n_moves, len(played), len(evals))):
        raw = tuple(int(x) for x in played[i])
        cands.append((raw, mirror_xg_pos(played[i]), cubeless_equity_from_xg_eval(evals[i])))
    hit_id = sum(1 for c in cands if c[0] in my_keys)
    hit_mir = sum(1 for c in cands if c[1] in my_keys)
    use = 0 if hit_id >= hit_mir else 1
    xg_eq = {}
    for c in cands:
        a = my_keys.get(c[use])
        if a is not None:
            xg_eq[a] = float(c[2])  # mover equity of playing this afterstate (higher is better)
    if len(xg_eq) < min_candidates:
        return None
    return board, dice, xg_eq


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Build the command-line parser with defaults from the settings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", default=list(cfg["agents"]))
    p.add_argument("--analysis-xg", required=True, type=Path)
    p.add_argument(
        "--out", type=Path, default=None, help="output directory (default outputs/xg_oracle)"
    )
    return p


def _order(pr_by_agent):
    """Return agent names sorted from best (lowest PR) to worst."""
    return sorted(pr_by_agent, key=lambda n: pr_by_agent[n])


def score_decisions(decisions, agents, oracles):
    """Return each agent's equity losses under each GNU Backgammon depth and under XG."""
    (label_lo, gg_lo), (label_hi, gg_hi) = oracles
    names = list(agents)
    losses = {
        label_lo: {n: [] for n in names},
        label_hi: {n: [] for n in names},
        "xg": {n: [] for n in names},
    }
    for board, dice, xg_eq in decisions:
        afters = list(xg_eq)  # only the afterstates XG covered
        g_lo = {a: -gg_lo.equity(a) for a in afters}
        g_hi = {a: -gg_hi.equity(a) for a in afters}
        best_xg, best_lo = max(xg_eq.values()), max(g_lo.values())
        best_hi = max(g_hi.values())
        game = Game(board=board)
        game.set_dice(dice)
        for name, agent in agents.items():
            chosen = agent.choose_move(game).afterstate
            if chosen not in xg_eq:  # the agent's move is outside XG's candidate list
                continue
            losses["xg"][name].append(best_xg - xg_eq[chosen])
            losses[label_lo][name].append(best_lo - g_lo[chosen])
            losses[label_hi][name].append(best_hi - g_hi[chosen])
    return losses


def rank_agreement(pr, names, label_lo, label_hi):
    """Return Kendall tau between the XG, shallow and deep rankings over agents XG scored."""
    common = [n for n in names if n in pr["xg"]]
    ox = [n for n in _order(pr["xg"]) if n in common]
    tau_x_lo = kendall_tau(ox, [n for n in _order(pr[label_lo]) if n in common])
    o_hi = [n for n in _order(pr[label_hi]) if n in common]
    tau_x_hi = kendall_tau(ox, o_hi)
    tau_lo_hi = kendall_tau([n for n in _order(pr[label_lo]) if n in common], o_hi)
    return tau_x_lo, tau_x_hi, tau_lo_hi


def report_lines(pr, losses, taus, labels, n_dec, xg_name, tau_threshold):
    """Return the Markdown report as a list of lines."""
    label_lo, label_hi = labels
    tau_x_lo, tau_x_hi, tau_lo_hi = taus
    if tau_x_lo >= tau_threshold:
        agreement = f"XG agrees with GNU Backgammon {label_lo} (τ ≥ {tau_threshold})"
    else:
        agreement = f"XG disagrees with GNU Backgammon {label_lo} (τ < {tau_threshold})"
    tau_line = f"- τ(XG, {label_lo}) = **{tau_x_lo:+.3f}**"
    tau_line += f"  · τ(XG, {label_hi}) = **{tau_x_hi:+.3f}**"
    tau_line += f"  · τ({label_lo}, {label_hi}) = {tau_lo_hi:+.3f}"
    lines = [
        "# Agent rankings under XG and GNU Backgammon",
        "",
        f"- Evaluation set: **{n_dec} checker decisions** taken directly from the XG analysis "
        f"`{xg_name}` (reconstructed via the inverse position codec).",
        f"- Agents scored under XG (World Class) vs gnubg {label_lo} / {label_hi} on the same "
        "positions.",
        "",
        "## PR under each oracle (lower = better; over each agent's XG-covered decisions)",
        "",
        f"| agent | {label_lo} PR | {label_hi} PR | XG PR | n |",
        "|---|---|---|---|---|",
    ]
    for n in _order(pr[label_lo]):
        p_hi = f"{pr[label_hi][n]:.1f}"
        px = f"{pr['xg'][n]:.1f}"
        lines.append(f"| {n} | {pr[label_lo][n]:.1f} | {p_hi} | {px} | {len(losses['xg'][n])} |")
    lines += [
        "",
        "## Ranking agreement (Kendall τ)",
        "",
        tau_line,
        "",
        f"**{agreement}.**",
        "",
        f"XG evaluates at World Class (stronger than {label_lo}), so its absolute PR scale "
        "differs; this comparison uses the rankings only.",
    ]
    return lines


def main() -> None:
    """Score the panel under XG and GNU Backgammon and write the comparison."""
    cfg = settings(EXPERIMENT)
    args = build_parser(cfg).parse_args()
    enc = BinarySlotEncoder()
    agents = dict(resolve_agent(s, enc) for s in args.agents)
    names = list(agents)
    plies_lo, plies_hi = (int(p) for p in cfg["gnubg_plies"])
    labels = (f"{plies_lo}-ply", f"{plies_hi}-ply")
    gg_lo = GnubgOracle(OracleConfig(plies=plies_lo))
    gg_hi = GnubgOracle(OracleConfig(plies=plies_hi))

    decisions = list(_decisions_from_xg(str(args.analysis_xg), int(cfg["min_candidates"])))
    n_dec = len(decisions)
    print(f"XG-covered checker decisions: {n_dec}")

    losses = score_decisions(decisions, agents, ((labels[0], gg_lo), (labels[1], gg_hi)))
    pr_scale = int(cfg["pr_scale"])
    pr = {
        orc: {n: float(np.mean(v) * pr_scale) for n, v in d.items() if v}
        for orc, d in losses.items()
    }
    taus = rank_agreement(pr, names, *labels)
    lines = report_lines(
        pr, losses, taus, labels, n_dec, args.analysis_xg.name, float(cfg["tau_threshold"])
    )

    out_dir = output_dir(EXPERIMENT) if args.out is None else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{EXPERIMENT}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "n_decisions": n_dec,
        "pr": pr,
        f"tau_xg_{plies_lo}ply": taus[0],
        f"tau_xg_{plies_hi}ply": taus[1],
        f"tau_{plies_lo}ply_{plies_hi}ply": taus[2],
    }
    md.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
