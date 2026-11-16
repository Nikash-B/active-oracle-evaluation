"""Board-to-feature encoder interface.

Agents and search code take an :class:`Encoder`; the concrete encoder is
:class:`~.binary_slot.BinarySlotEncoder`.
"""

from __future__ import annotations

import abc
from typing import Sequence

from active_oracle_evaluation.backgammon.core.board import Board


class Encoder(abc.ABC):
    """Maps a (canonicalized) board to a fixed-length feature vector.

    Implementations encode :meth:`Board.canonical` output so the network always
    sees the position from the mover's perspective.
    """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Length of the feature vector this encoder produces."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, board: Board) -> Sequence[float]:
        """Return the feature vector for ``board``."""
        raise NotImplementedError
