"""MuZero value-equivalent model for backgammon.

Three networks act in a learned latent space:

- **representation** ``h``: observation (encoded board) -> latent state ``z0``,
- **dynamics** ``g``: ``(z, action) -> (z', reward)``,
- **prediction** ``f``: ``z -> value`` (a six-class outcome distribution).

Backgammon has no compact action id, so the action fed to ``g`` is an embedding of the
encoded chosen afterstate. The per-ply reward is zero; the terminal outcome is carried by
the value head. The dice are not modelled: ``g`` is a deterministic transition.
"""

from __future__ import annotations

import torch
from torch import nn

from active_oracle_evaluation.backgammon.nn.device import default_torch_device
from active_oracle_evaluation.backgammon.nn.outcome_mlp import CLASS_POINTS, NUM_CLASSES

#: Epsilon in the min-max latent scaling of MuZero (Schrittwieser et al., 2020); value
#: chosen for this study.
_LATENT_EPS = 1e-5
#: Default latent width, action-embedding width and hidden widths, chosen for this study.
DEFAULT_LATENT_DIM = 64
DEFAULT_ACTION_DIM = 32
DEFAULT_HIDDEN = (128,)


def _mlp(sizes: list[int]) -> nn.Sequential:
    """Build a ReLU MLP with the given layer widths and a linear output layer."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _normalize_latent(latent: torch.Tensor) -> torch.Tensor:
    """Min-max scale each latent to [0, 1], as in MuZero, to keep states bounded."""
    low = latent.min(dim=-1, keepdim=True).values
    high = latent.max(dim=-1, keepdim=True).values
    return (latent - low) / (high - low + _LATENT_EPS)


class MuZeroNet(nn.Module):
    """Representation + dynamics + prediction networks for a value-equivalent model."""

    def __init__(
        self,
        input_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
    ) -> None:
        """Build the five sub-networks and move the model to the default device."""
        super().__init__()
        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden = tuple(int(h) for h in hidden)
        widths = list(self.hidden)

        self.representation = _mlp([self.input_size, *widths, self.latent_dim])
        self.action_embed = _mlp([self.input_size, *widths, self.action_dim])
        self.dynamics = _mlp([self.latent_dim + self.action_dim, *widths, self.latent_dim])
        self.reward_head = _mlp([self.latent_dim + self.action_dim, *widths, 1])
        self.prediction = _mlp([self.latent_dim, *widths, NUM_CLASSES])

        self.register_buffer("class_points", torch.tensor(CLASS_POINTS, dtype=torch.float32))
        self.to(default_torch_device())

    def _device(self) -> torch.device:
        """Return the device that holds the parameters."""
        return next(self.parameters()).device

    def represent(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode an observation (board features) into an initial latent state."""
        return _normalize_latent(self.representation(obs.to(self._device())))

    def dynamics_step(
        self, latent: torch.Tensor, action_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ``g``: ``(latent, encoded-afterstate-action) -> (next latent, reward)``."""
        action = self.action_embed(action_obs.to(self._device()))
        joined = torch.cat([latent, action], dim=-1)
        next_latent = _normalize_latent(self.dynamics(joined))
        reward = self.reward_head(joined).squeeze(-1)
        return next_latent, reward

    def prediction_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """``f``: latent -> six-class outcome logits."""
        return self.prediction(latent)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        """Expected point-equity of a latent (for the player to move there)."""
        return torch.softmax(self.prediction_logits(latent), dim=-1) @ self.class_points
