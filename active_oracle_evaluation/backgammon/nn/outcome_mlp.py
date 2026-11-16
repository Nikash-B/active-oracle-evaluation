"""Outcome-head value network: a six-class outcome head.

Instead of a single scalar value, the network predicts a probability distribution
over the six mutually-exclusive game outcomes, in the canonical
:class:`~active_oracle_evaluation.backgammon.core.types.Outcome` order:

    index: 0    1     2      3    4     5
    class: W    WG    WBG    L    LG    LBG

Expected point-equity (used for move selection, the same role as a scalar value) is
the dot product with the point weights ``[+1,+2,+3,-1,-2,-3]``:

    EV(p) = (pW - pL) + 2*(pWG - pLG) + 3*(pWBG - pLBG)

Default architecture:

    input (247) -> 256 -> 256 -> 128 -> 6 logits   (softmax applied for probs)
"""

from __future__ import annotations

import torch
from torch import nn

from active_oracle_evaluation.backgammon.core.types import Outcome
from active_oracle_evaluation.backgammon.nn.device import default_torch_device

#: Class order = the Outcome enum order (wins then losses, by magnitude).
OUTCOME_ORDER: tuple[Outcome, ...] = (
    Outcome.SINGLE_WIN,
    Outcome.GAMMON_WIN,
    Outcome.BACKGAMMON_WIN,
    Outcome.SINGLE_LOSS,
    Outcome.GAMMON_LOSS,
    Outcome.BACKGAMMON_LOSS,
)
NUM_CLASSES = len(OUTCOME_ORDER)  # 6
#: Signed point value of each class (the EV weights), from backgammon scoring rules.
CLASS_POINTS: tuple[int, ...] = tuple(o.points for o in OUTCOME_ORDER)  # (1,2,3,-1,-2,-3)
#: Default hidden-layer widths (the paper's hyperparameter table, outcome-head trunk
#: 247->256->256->128).
DEFAULT_HIDDEN = (256, 256, 128)


class OutcomeMLP(nn.Module):
    """Feed-forward afterstate network predicting the six-class outcome logits."""

    def __init__(self, input_size: int, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
        """Build the MLP body with ReLU hidden layers and a six-logit output."""
        super().__init__()
        self.input_size = int(input_size)
        self.hidden = tuple(int(h) for h in hidden)

        layers: list[nn.Module] = []
        prev = self.input_size
        for width in self.hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, NUM_CLASSES))
        self.body = nn.Sequential(*layers)

        # Point weights as a buffer so they ride along with .to(device) / state.
        self.register_buffer("class_points", torch.tensor(CLASS_POINTS, dtype=torch.float32))
        self.to(default_torch_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits ``(..., 6)``."""
        x = x.to(next(self.parameters()).device)
        return self.body(x)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax outcome distribution ``(..., 6)``."""
        return torch.softmax(self.forward(x), dim=-1)

    def equity(self, x: torch.Tensor) -> torch.Tensor:
        """Expected point-equity ``(...)`` for the player to move."""
        return self.probabilities(x) @ self.class_points
