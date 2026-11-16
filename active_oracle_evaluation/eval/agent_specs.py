"""Resolve agent specs (``random``, ``heuristic``, ``pubeval``, ``label:path``, ...) to agents.

Shared by the experiments. Not re-exported from ``eval/__init__`` so that
``import active_oracle_evaluation.eval`` stays torch-free. ``random`` and ``heuristic`` need only the standard library; torch, gnubg-nn and
the external engines are imported only when a spec needs them.
"""

from __future__ import annotations

from active_oracle_evaluation.backgammon.agents.base import Agent
from active_oracle_evaluation.backgammon.agents.heuristic import HeuristicAgent
from active_oracle_evaluation.backgammon.agents.random_agent import RandomAgent
from active_oracle_evaluation.backgammon.encoding.binary_slot import BinarySlotEncoder

#: Base URL of a locally running wildbg server, used by the bare ``wildbg`` spec (the wildbg
#: server's default port).
WILDBG_DEFAULT_URL = "http://localhost:8080"


def load_checkpoint(*args, **kwargs):
    """Load a checkpoint with ``nn.mlp.load_checkpoint``, importing torch on use."""
    from active_oracle_evaluation.backgammon.nn.mlp import load_checkpoint as _load_checkpoint

    return _load_checkpoint(*args, **kwargs)


def make_agent_for_model(model, encoder) -> Agent:
    """Wrap a loaded model in the agent that matches its head, importing torch on use.

    ``PolicyValueMLP`` gives a :class:`PolicyAgent`, ``OutcomeMLP`` an :class:`OutcomeAgent`
    (selection by expected equity), and any other model the scalar-value :class:`TDAgent`.
    """
    from active_oracle_evaluation.backgammon.agents.outcome_agent import OutcomeAgent
    from active_oracle_evaluation.backgammon.agents.policy_agent import PolicyAgent
    from active_oracle_evaluation.backgammon.agents.td_agent import TDAgent
    from active_oracle_evaluation.backgammon.nn.outcome_mlp import OutcomeMLP
    from active_oracle_evaluation.backgammon.nn.policy_value_mlp import PolicyValueMLP

    if isinstance(model, PolicyValueMLP):
        return PolicyAgent(model, encoder)
    if isinstance(model, OutcomeMLP):
        return OutcomeAgent(model, encoder)
    return TDAgent(model, encoder)


def resolve_agent(spec: str, encoder: BinarySlotEncoder) -> tuple[str, Agent]:
    """Return ``(label, agent)`` for a built-in spec or a ``label:path`` checkpoint spec."""
    label, _, path = spec.partition(":")
    if label == "random" and not path:
        return "random", RandomAgent()
    if label == "heuristic" and not path:
        return "heuristic", HeuristicAgent()
    if label == "pubeval" and not path:
        from active_oracle_evaluation.eval.pubeval import PubevalAgent  # needs gnubg-nn

        return "pubeval", PubevalAgent()
    if label == "gnubg":
        from active_oracle_evaluation.eval.gnubg_agent import GnubgAgent  # needs gnubg-nn

        return f"gnubg{path or 0}", GnubgAgent(plies=int(path) if path else 0)
    if label == "wildbg":
        # wildbg:<base-url> plays through a running wildbg server's 0-ply evaluator over HTTP.
        from active_oracle_evaluation.eval.wildbg_agent import WildbgAgent

        return "wildbg", WildbgAgent(path or WILDBG_DEFAULT_URL)
    if label == "puretd":
        # puretd:<checkpoint> plays the PureTD network of puretd/.
        from active_oracle_evaluation.eval.puretd_agent import PureTDAgent

        return "puretd", PureTDAgent(path)
    if label.startswith("muzero"):
        # muzero:path, muzero25:path, ...: one-step lookahead with the learned model.
        from active_oracle_evaluation.backgammon.agents.muzero_agent import MuZeroAgent

        model = load_checkpoint(path)
        return label, MuZeroAgent(model, encoder)
    if label == "search2":
        # 2-ply value search over the checkpoint.
        from active_oracle_evaluation.backgammon.search.agent import SearchAgent

        model = load_checkpoint(path)
        return "search2", SearchAgent(model, encoder)
    if label == "sel3":
        # Selective 3-ply extension of the 2-ply search.
        from active_oracle_evaluation.backgammon.search.nply import SelectiveDepthAgent

        model = load_checkpoint(path)
        return "sel3", SelectiveDepthAgent(model, encoder)
    if not path:
        raise SystemExit(f"agent spec {spec!r}: bare name must be random/heuristic/pubeval/gnubg")
    model = load_checkpoint(path)
    return label, make_agent_for_model(model, encoder)
