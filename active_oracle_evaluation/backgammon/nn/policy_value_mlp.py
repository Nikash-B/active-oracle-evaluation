"""Policy/value network for legal-afterstate agents.

The policy head scores each legal candidate afterstate independently. A caller
enumerates legal moves, encodes only those afterstates, and applies softmax over
that candidate list. This keeps the policy naturally masked to legal moves
without inventing a fixed action table for compound dice plays.

The auxiliary outcome head mirrors :class:`OutcomeMLP`, so the same checkpoint
also provides point equity and six outcome probabilities for search and cube use.
"""

from __future__ import annotations

import torch
from torch import nn

from active_oracle_evaluation.backgammon.nn.device import default_torch_device
from active_oracle_evaluation.backgammon.nn.outcome_mlp import CLASS_POINTS, NUM_CLASSES

#: Default hidden-layer widths (the paper's hyperparameter table, AlphaZero and PPO trunk
#: 247->256->256->128).
DEFAULT_HIDDEN = (256, 256, 128)


class PolicyValueMLP(nn.Module):
    """Shared-trunk MLP with legal-afterstate policy and outcome heads."""

    def __init__(self, input_size: int, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
        """Build the shared trunk, both heads and the outcome-to-points buffer."""
        super().__init__()
        self.input_size = int(input_size)
        self.hidden = tuple(int(h) for h in hidden)

        layers: list[nn.Module] = []
        prev = self.input_size
        for width in self.hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(prev, 1)
        self.outcome_head = nn.Linear(prev, NUM_CLASSES)
        self.register_buffer("class_points", torch.tensor(CLASS_POINTS, dtype=torch.float32))
        self.to(default_torch_device())

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Move ``x`` to the model device and return the shared trunk features."""
        x = x.to(next(self.parameters()).device)
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy logits ``(...)`` for encoded candidate afterstates."""
        return self.policy_logits(x)

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalized candidate-move logits ``(...)``."""
        return self.policy_head(self._features(x)).squeeze(-1)

    def outcome_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw six-outcome logits ``(..., 6)``."""
        return self.outcome_head(self._features(x))

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax outcome distribution ``(..., 6)``."""
        return torch.softmax(self.outcome_logits(x), dim=-1)

    def equity(self, x: torch.Tensor) -> torch.Tensor:
        """Expected point-equity ``(...)`` for the player to move."""
        return self.probabilities(x) @ self.class_points
