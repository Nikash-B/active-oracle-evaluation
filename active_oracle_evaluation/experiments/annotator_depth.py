"""Annotator-depth ladder: rank agreement when one engine annotates at 0, 1, 2 and 3 ply.

One frozen 1,000-position suite and six agents are scored four times by gnubg-nn at four search
depths. The pool contains copies of the annotator itself (``gnubg0``, ``gnubg1``, ``gnubg2``), so
the annotator sits below, level with, and above the pool as the depth varies. An agent identical
to the annotator scores PR = 0 by construction. The experiment writes the PR matrix, the ranks
and the Kendall tau-a and tau-b between scorings as annotator_depth.md and annotator_depth.json
in outputs/annotator_depth/. The settings are in the [annotator_depth] table of
config/experiments.toml.

    python -m active_oracle_evaluation.experiments.annotator_depth
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.paths import output_dir

EXPERIMENT = "annotator_depth"

#: Performance Rating by annotator depth, rounded to one decimal, as in the paper's table of PR
#: at four search depths (appendix "Validity of the oracle rating"). ``None`` marks the annotator
#: rating its own twin (PR = 0 by construction).
PR: dict[int, dict[str, float | None]] = {
    0: {"gnubg0": None, "gnubg2": 1.6, "gnubg1": 1.9, "sel3": 2.0, "puretd": 2.2, "search2": 2.9},
    1: {"gnubg1": None, "sel3": 1.4, "gnubg2": 1.5, "puretd": 1.9, "gnubg0": 2.2, "search2": 2.4},
    2: {"gnubg2": None, "sel3": 0.9, "gnubg1": 1.2, "puretd": 1.3, "gnubg0": 1.7, "search2": 2.4},
    3: {"gnubg2": 0.5, "gnubg1": 0.7, "sel3": 1.0, "puretd": 1.3, "gnubg0": 2.2, "search2": 2.2},
}


def _scored(depth: int) -> dict[str, float]:
    """Return PR at one depth with the self-annotated agent set to 0.0."""
    return {agent: (0.0 if value is None else value) for agent, value in PR[depth].items()}


def tau_a(x: list[float], y: list[float]) -> float:
    """Kendall tau-a: (concordant - discordant) / all pairs; ties count as neither."""
    n = len(x)
    concordant = discordant = 0
    for i, j in itertools.combinations(range(n), 2):
        sign = (x[i] - x[j]) * (y[i] - y[j])
        concordant += sign > 0
        discordant += sign < 0
    return (concordant - discordant) / (n * (n - 1) / 2)


def tau_b(x: list[float], y: list[float]) -> float:
    """Kendall tau-b, with ties removed from the denominator."""
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        else:
            concordant += dx * dy > 0
            discordant += dx * dy < 0
    denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return (concordant - discordant) / denom if denom else 0.0


class Ladder:
    """PR columns, orderings and tau between annotator depths for one agent panel."""

    def __init__(self, agents: list[str], depths: list[int]):
        """Score every depth and sort the agents under each annotator."""
        self.agents = agents
        self.depths = depths
        self.cols = {depth: _scored(depth) for depth in depths}
        self.order = {
            depth: sorted(agents, key=lambda agent: self.cols[depth][agent]) for depth in self.cols
        }

    def pair(self, p: int, q: int) -> tuple[float, float]:
        """Return (tau-a, tau-b) between the scorings at depths p and q."""
        x = [self.cols[p][agent] for agent in self.agents]
        y = [self.cols[q][agent] for agent in self.agents]
        return tau_a(x, y), tau_b(x, y)

    def rank(self, depth: int, agent: str) -> int:
        """Return the 1-based rank of an agent under the annotator at one depth."""
        return self.order[depth].index(agent) + 1


def _matrix_lines(ladder: Ladder, suite_size: int) -> list[str]:
    """Return the report title, the setup paragraph and the PR matrix."""
    lines = [
        "# Annotator-depth ladder",
        "",
        f"One frozen {suite_size:,}-position suite, {len(ladder.agents)} agents, scored four "
        "times by the same engine at four search depths (gnubg-nn 0/1/2/3-ply). The pool "
        "contains copies of the annotator (`gnubg0`, `gnubg1`, `gnubg2`), so as depth varies "
        "the annotator sits below, level with, and above the agents it rates. Only the "
        "annotator changes.",
        "",
        "## PR matrix (lower is better; ≡ marks the annotator rating itself)",
        "",
        "| agent | 0-ply | 1-ply | 2-ply | 3-ply |",
        "|---|---|---|---|---|",
    ]
    for agent in ladder.agents:
        cells = []
        for depth in ladder.depths:
            value = PR[depth][agent]
            cells.append("**0.0 ≡**" if value is None else f"{value}")
        lines.append(f"| `{agent}` | " + " | ".join(cells) + " |")
    return lines


def _twin_lines(ladder: Ladder) -> list[str]:
    """Return section 1: the annotator's twin ranks first at every depth."""
    n_agents = len(ladder.agents)
    lines = [
        "",
        "## 1. The annotator's twin ranks first by construction",
        "",
        "At every depth, the agent identical to the annotator scores 0.0 and takes rank 1, because "
        "it plays the move the oracle calls best. Follow `gnubg0` across the row:",
        "",
        "| annotator | `gnubg0` PR | `gnubg0` rank |",
        "|---|---|---|",
    ]
    for depth in ladder.depths:
        lines.append(
            f"| {depth}-ply | {ladder.cols[depth]['gnubg0']} | "
            f"{ladder.rank(depth, 'gnubg0')} of {n_agents} |"
        )
    lines += [
        "",
        f"`gnubg0` falls from rank {ladder.rank(0, 'gnubg0')} to rank {ladder.rank(1, 'gnubg0')} "
        "once the annotator searches deeper than it, with nothing about the agent changed.",
        "",
        "Agents that out-search the annotator are penalized for it. "
        f"`sel3` scores {PR[0]['sel3']} under the 0-ply annotator (rank {ladder.rank(0, 'sel3')}) "
        f"but {PR[2]['sel3']} under 2-ply (rank {ladder.rank(2, 'sel3')}).",
    ]
    return lines


def _agreement_lines(ladder: Ladder, pool_max_ply: int, depth_pairs: list) -> list[str]:
    """Return section 2: tau-a and tau-b between pairs of annotator depths."""
    lines = [
        "",
        "## 2. Rank agreement between depths",
        "",
        f"The deepest agent in the pool is `gnubg:{pool_max_ply}`, so only the 3-ply annotator is "
        "strictly stronger than everything it rates.",
        "",
        "| comparison | crosses the pool's strength band? | τ-a | τ-b |",
        "|---|---|---|---|",
    ]
    for p, q in depth_pairs:
        ta, tb = ladder.pair(p, q)
        crosses = "**yes**" if p < pool_max_ply else "no"
        lines.append(f"| {p}-ply vs {q}-ply | {crosses} | {ta:+.3f} | {tb:+.3f} |")
    return lines


def _scope_lines(ladder: Ladder, ta02: float, ta23: float) -> list[str]:
    """Return the closing summary of the two quoted tau values and section 3 (scope)."""
    n_agents = len(ladder.agents)
    n_pairs = n_agents * (n_agents - 1) // 2
    return [
        "",
        f"τ(2-ply, 3-ply) = {ta23:+.3f} against τ(0-ply, 2-ply) = {ta02:+.3f}. Two annotators "
        "that both match or exceed the pool leave the ladder largely intact; crossing the pool's "
        "own strength band reorders it. τ-b differs from τ-a only where a column has ties (two "
        f"agents share PR {ladder.cols[3]['gnubg0']} at 3-ply, from rounding to one decimal).",
        "",
        "## 3. Scope",
        "",
        "- The pool contains the annotator at three depths so that the strength requirement can be "
        "violated on demand. The ladder shows what a violation looks like; it does not certify that "
        "another panel is free of one.",
        f"- {n_agents} agents, one suite, one engine family. Over {n_pairs} pairs a single rank "
        f"swap moves τ-a by {2 / n_pairs:.2f}.",
    ]


def _summary(ladder: Ladder, pool_max_ply: int, depth_pairs: list) -> dict:
    """Return the report's numbers as a JSON-serialisable dict."""
    comparisons = []
    for p, q in depth_pairs:
        ta, tb = ladder.pair(p, q)
        comparisons.append(
            {
                "depths": [p, q],
                "crosses_pool_band": p < pool_max_ply,
                "tau_a": round(ta, 3),
                "tau_b": round(tb, 3),
            }
        )
    return {
        "agents": ladder.agents,
        "pool_max_ply": pool_max_ply,
        "pr": {str(depth): {a: PR[depth][a] for a in ladder.agents} for depth in ladder.depths},
        "rank": {
            str(depth): {a: ladder.rank(depth, a) for a in ladder.agents} for depth in ladder.depths
        },
        "comparisons": comparisons,
    }


def main() -> None:
    """Compute ranks and tau between annotator depths and write the report and summary."""
    config = settings(EXPERIMENT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report (default: outputs/annotator_depth/annotator_depth.md); "
        "annotator_depth.json is written next to it",
    )
    args = ap.parse_args()
    out = output_dir(EXPERIMENT) / f"{EXPERIMENT}.md" if args.out is None else args.out

    ladder = Ladder(list(config["agents"]), list(config["depths"]))
    pool_max_ply = config["pool_max_ply"]
    depth_pairs = config["depth_pairs"]
    lines = _matrix_lines(ladder, config["suite_size"])
    lines += _twin_lines(ladder)
    lines += _agreement_lines(ladder, pool_max_ply, depth_pairs)
    ta02, _ = ladder.pair(0, 2)
    ta23, _ = ladder.pair(2, 3)
    lines += _scope_lines(ladder, ta02, ta23)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    json_path = out.parent / f"{EXPERIMENT}.json"
    summary = _summary(ladder, pool_max_ply, depth_pairs)
    json_path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"tau(0,2)={ta02:+.3f}  tau(2,3)={ta23:+.3f}; wrote {out}")


if __name__ == "__main__":
    main()
