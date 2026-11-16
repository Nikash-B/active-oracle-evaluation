"""PyTorch device selection helpers."""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch

# When set to a truthy value, device auto-selection returns CPU even if CUDA/MPS is
# available.
FORCE_CPU_ENV = "BACKGAMMON_FORCE_CPU"


def default_torch_device() -> torch.device:
    """Return the best available torch device: CUDA, then MPS, then CPU.

    The priority order is chosen for this study.

    It returns CPU whenever the ``BACKGAMMON_FORCE_CPU`` env var is set (see
    :data:`FORCE_CPU_ENV`).
    """
    if os.environ.get(FORCE_CPU_ENV):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_dtype_for_capability(capability: tuple[int, int]) -> "torch.dtype":
    """Pick the AMP dtype for a CUDA compute capability.

    Pre-Ampere cards (compute < 8.0) have no native bf16 tensor-core path, so they
    use fp16. Ampere and later use bf16, which needs no gradient loss scaling.
    """
    # Capability threshold 8 (Ampere) chosen for this study from NVIDIA's bf16 support.
    return torch.float16 if capability[0] < 8 else torch.bfloat16


def amp_dtype(device: torch.device) -> "torch.dtype":
    """The autocast dtype to use on ``device``, chosen by CUDA capability.

    Off CUDA the value is irrelevant (autocast is a no-op there); bf16 is returned
    as a harmless default.
    """
    if device.type == "cuda":
        return _amp_dtype_for_capability(torch.cuda.get_device_capability(device))
    return torch.bfloat16


@contextlib.contextmanager
def amp_autocast(device: torch.device) -> Iterator[None]:
    """Mixed-precision autocast on CUDA; a no-op everywhere else.

    The dtype follows :func:`amp_dtype`.
    """
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype(device)):
            yield
    else:
        yield
