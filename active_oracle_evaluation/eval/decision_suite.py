"""On-disk oracle decision suite: one JSONL record per decision.

Each :class:`DecisionRecord` stores the pre-decision board, the dice, every legal afterstate and
the oracle's mover equity for each afterstate. All fields are JSON-native, so a record written
with :func:`write_jsonl` and read back with :func:`read_jsonl` is equal to the original.
Unlabeled records (``oracle_equities`` = ``None``) are valid. Forced moves and pools built
without an oracle use them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.backgammon.core.types import Player
from active_oracle_evaluation.eval.phase_labels import label_phase

#: Hex digits kept from the SHA-1 digest as the position id (chosen for this study).
_POSITION_ID_HEX_DIGITS = 16

# ---------------------------------------------------------------------------- #
# Board (de)serialization — lossless, JSON-native
# ---------------------------------------------------------------------------- #


def serialize_board(board: Board) -> dict[str, Any]:
    """Return a board as JSON-native fields: signed points, bar and off counts, side to move."""
    return {
        "points": list(board.points),
        "white_bar": board.bar(Player.WHITE),
        "black_bar": board.bar(Player.BLACK),
        "white_off": board.off(Player.WHITE),
        "black_off": board.off(Player.BLACK),
        "turn": int(board.turn),  # Player is an IntEnum (+1 White / -1 Black)
    }


def deserialize_board(d: dict[str, Any]) -> Board:
    """Rebuild the :class:`Board` written by :func:`serialize_board`."""
    return Board(
        tuple(int(x) for x in d["points"]),
        int(d["white_bar"]),
        int(d["black_bar"]),
        int(d["white_off"]),
        int(d["black_off"]),
        Player(int(d["turn"])),
    )


def _position_id(board_dict: dict[str, Any], dice: list[int] | None) -> str:
    """Return a deterministic 16-hex-digit id of a (board, dice) decision."""
    payload = json.dumps(
        [
            board_dict["points"],
            board_dict["white_bar"],
            board_dict["black_bar"],
            board_dict["white_off"],
            board_dict["black_off"],
            board_dict["turn"],
            list(dice) if dice is not None else None,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_POSITION_ID_HEX_DIGITS]


# ---------------------------------------------------------------------------- #
# The record
# ---------------------------------------------------------------------------- #


@dataclass
class DecisionRecord:
    """One decision with its position, legal actions and oracle equities.

    ``legal_actions`` is index-aligned with ``oracle_equities``; ``best_action`` indexes both.
    """

    position_id: str
    board: dict[str, Any]  # serialized pre-decision board (see serialize_board)
    player_to_move: int  # Player value (+1 White / -1 Black)
    dice: list[int] | None  # [d1, d2] for checker decisions, None for cube
    cube_state: dict[str, Any] | None  # {"owner", "value", "centered"} or None
    legal_actions: list[dict[str, Any]]  # serialized afterstates, aligned to oracle_equities
    agent_source: str  # which agent generated the position
    phase: str  # a phase_labels.label_phase label
    decision_type: str  # always "checker" (make_checker_record)
    forced_move: bool  # True when <= 1 legal action (no skill signal)
    oracle_equities: list[float] | None  # mover equity per legal action, or None if unlabeled
    best_action: int | None  # argmax(oracle_equities), or None if unlabeled
    best_equity: float | None  # max(oracle_equities), or None if unlabeled
    metadata: dict[str, Any] = field(default_factory=dict)  # seed/game_id/ply_index/oracle_*

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-native dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionRecord":
        """Build a record from a dict written by :meth:`to_dict`."""
        return cls(
            position_id=d["position_id"],
            board=d["board"],
            player_to_move=d["player_to_move"],
            dice=d["dice"],
            cube_state=d["cube_state"],
            legal_actions=d["legal_actions"],
            agent_source=d["agent_source"],
            phase=d["phase"],
            decision_type=d["decision_type"],
            forced_move=d["forced_move"],
            oracle_equities=d["oracle_equities"],
            best_action=d["best_action"],
            best_equity=d["best_equity"],
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------- #
# Reconstruction helpers
# ---------------------------------------------------------------------------- #


def to_board(record: DecisionRecord) -> Board:
    """Rebuild the pre-decision :class:`Board` from a record."""
    return deserialize_board(record.board)


def afterstates_of(record: DecisionRecord) -> tuple[Board, ...]:
    """Rebuild the legal afterstate boards (aligned with ``oracle_equities``)."""
    return tuple(deserialize_board(a) for a in record.legal_actions)


def dice_of(record: DecisionRecord) -> tuple[int, int]:
    """Return the roll as a tuple."""
    return (int(record.dice[0]), int(record.dice[1]))


# ---------------------------------------------------------------------------- #
# Builders
# ---------------------------------------------------------------------------- #


def make_checker_record(
    board: Board,
    dice: tuple[int, int],
    legal_afterstates: tuple[Board, ...],
    *,
    agent_source: str,
    equities: list[float] | tuple[float, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Build a checker :class:`DecisionRecord` from engine objects.

    ``equities`` are oracle mover equities aligned to ``legal_afterstates``; ``best_action`` and
    ``best_equity`` are their argmax and max. Pass ``None`` for an unlabeled record. The phase is
    computed with :func:`label_phase`.
    """
    board_dict = serialize_board(board)
    dice_list = [int(dice[0]), int(dice[1])]
    eq = [float(e) for e in equities] if equities is not None else None
    best_action = max(range(len(eq)), key=eq.__getitem__) if eq else None
    best_equity = eq[best_action] if (eq and best_action is not None) else None
    ply = (metadata or {}).get("ply_index")
    phase = label_phase(board, ply_index=ply)
    return DecisionRecord(
        position_id=_position_id(board_dict, dice_list),
        board=board_dict,
        player_to_move=int(board.turn),
        dice=dice_list,
        cube_state=None,
        legal_actions=[serialize_board(a) for a in legal_afterstates],
        agent_source=agent_source,
        phase=phase,
        decision_type="checker",
        # A decision with at most one legal afterstate is forced (the paper's appendix "Experimental
        # protocol" scores unforced decisions only).
        forced_move=len(legal_afterstates) <= 1,
        oracle_equities=eq,
        best_action=best_action,
        best_equity=best_equity,
        metadata=dict(metadata or {}),
    )


def record_from_suite_position(
    pos: Any,  # active_oracle_evaluation.eval.suite.SuitePosition
    *,
    agent_source: str,
    metadata: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Build a labeled checker :class:`DecisionRecord` from a :class:`SuitePosition`."""
    return make_checker_record(
        pos.board,
        tuple(pos.dice),
        tuple(pos.legal_afterstates),
        agent_source=agent_source,
        equities=pos.equities,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------- #
# JSONL IO — exact round-trip
# ---------------------------------------------------------------------------- #


def write_jsonl(records: list[DecisionRecord], path: str | Path) -> None:
    """Write records to ``path`` as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: str | Path) -> list[DecisionRecord]:
    """Read a JSONL file written by :func:`write_jsonl` back into records."""
    out: list[DecisionRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(DecisionRecord.from_dict(json.loads(line)))
    return out
