"""Pick a torch device that works on both macOS (MPS) and WSL/Linux (CUDA)."""

import os

# Let unsupported MPS ops fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return `prefer` if given (or $HUMPER_DEVICE), else cuda > mps > cpu."""
    name = prefer or os.environ.get("HUMPER_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
