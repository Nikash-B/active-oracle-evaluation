"""The scalar-head value network: a small MLP over the binary-slot encoding.

Architecture:

    input (247) -> Linear 128 -> ReLU -> Linear 128 -> ReLU -> Linear 1 -> 3*tanh

The output is a single scalar **point-equity for the player to move** in the
encoded (canonical) board, squashed into ``[-3, +3]`` by ``3 * tanh`` so it lines
up with the reward set ``{+/-1, +/-2, +/-3}``.

Checkpoints are self-describing: alongside the weights they store the
architecture, the encoder identity and size, the training hyperparameters and the
number of self-play games. :func:`load_checkpoint` uses them to rebuild any
registered model class.
"""

from __future__ import annotations

import torch
from torch import nn

from active_oracle_evaluation.backgammon.nn.device import default_torch_device

#: Default hidden-layer widths (the paper's hyperparameter table, scalar-head trunk
#: 247->128->128).
DEFAULT_HIDDEN = (128, 128)
#: Point-equity bound; the head is ``OUTPUT_SCALE * tanh(.)``. Chosen for this study to
#: match the backgammon point range of +/-3.
OUTPUT_SCALE = 3.0


class ValueMLP(nn.Module):
    """Feed-forward afterstate value network producing a bounded scalar equity."""

    def __init__(
        self,
        input_size: int,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        output_scale: float = OUTPUT_SCALE,
    ) -> None:
        """Build the ReLU MLP body with a single scalar output."""
        super().__init__()
        self.input_size = int(input_size)
        self.hidden = tuple(int(h) for h in hidden)
        self.output_scale = float(output_scale)

        layers: list[nn.Module] = []
        prev = self.input_size
        for width in self.hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)
        self.to(default_torch_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features ``(..., input_size)`` to equity ``(..., 1)`` in [-scale, scale]."""
        x = x.to(next(self.parameters()).device)
        return self.output_scale * torch.tanh(self.body(x))


#: Maps a checkpoint ``model_type`` to the class that rebuilds it, as
#: ``(module_path, class_name)`` so the (torch-heavy) model modules are imported
#: lazily and only when actually loading.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "value_mlp": ("active_oracle_evaluation.backgammon.nn.mlp", "ValueMLP"),
    "outcome_mlp": ("active_oracle_evaluation.backgammon.nn.outcome_mlp", "OutcomeMLP"),
    "policy_value_mlp": (
        "active_oracle_evaluation.backgammon.nn.policy_value_mlp",
        "PolicyValueMLP",
    ),
    "muzero": ("active_oracle_evaluation.backgammon.nn.muzero", "MuZeroNet"),
}


def _resolve_model_class(model_type: str) -> type[nn.Module]:
    """Import and return the model class registered for ``model_type``."""
    import importlib

    if model_type not in _MODEL_REGISTRY:  # pragma: no cover - defensive guard
        raise ValueError(f"unknown checkpoint model_type {model_type!r}")
    module, name = _MODEL_REGISTRY[model_type]
    return getattr(importlib.import_module(module), name)


def load_checkpoint(path: str) -> nn.Module:
    """Load ``path`` and rebuild the model.

    Dispatches on the stored ``model_type``. The model is
    rebuilt from its stored architecture and set to eval mode.
    """
    device = default_torch_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model_type = payload["model_type"]
    model_cls = _resolve_model_class(model_type)
    model = model_cls(**payload["arch"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
