"""Atomic checkpoint I/O and RNG-state capture for reliable resumption.

Colab runtimes can disconnect at any moment, including during a write. Every save
therefore goes to a temporary file that is renamed only once fully written, so
readers never see a truncated checkpoint.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Serialise ``payload`` to ``path`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint produced by this project. Only load files you trust."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required because payloads contain Python metadata
    # (history lists, progress counters, RNG state). Do not load untrusted files.
    return torch.load(path, map_location=map_location, weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy and PyTorch (CPU and CUDA) random states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
