"""Load the DiagnoTech Model 2 bundle for inference.

Expected deployment directory contents (created by notebook 03):
    best_model.pth      weights and metadata (copied from the selected experiment)
    class_mapping.json  fixed class order; must match the checkpoint
    model_config.json   architecture, dropout, classes, validation-tuned thresholds, preprocessing

Every check here fails loudly. A mismatch between the weights, the class order and the
thresholds would produce plausible but wrong predictions, which is the worst failure mode.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diagnotech.checkpointing import load_checkpoint  # noqa: E402
from diagnotech.model import build_densenet121  # noqa: E402
from diagnotech.preprocessing import build_transforms  # noqa: E402

EXPECTED_CLASSES: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "No Finding", "Nodule", "Pneumothorax",
)


@dataclass
class ModelBundle:
    """Everything the predictor needs, loaded and validated once per process."""

    model: nn.Module
    classes: tuple[str, ...]
    thresholds: dict[str, float]
    transform: Callable
    device: torch.device
    model_version: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Deployment file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_bundle(model_dir: str | Path, device: str | torch.device | None = None) -> ModelBundle:
    """Load and cross-validate every deployment artifact, then return a ready-to-run bundle."""
    model_dir = Path(model_dir)
    class_mapping = _read_json(model_dir / "class_mapping.json")
    model_config = _read_json(model_dir / "model_config.json")

    classes = tuple(class_mapping["classes"])
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"class_mapping.json must list exactly {EXPECTED_CLASSES}; found {classes}")
    if tuple(model_config["classes"]) != classes:
        raise ValueError("model_config.json class order differs from class_mapping.json")
    if model_config.get("architecture") != "densenet121":
        raise ValueError(f"Unsupported architecture: {model_config.get('architecture')}")

    try:
        thresholds = {cls: float(model_config["thresholds"][cls]) for cls in classes}
    except KeyError as exc:
        raise ValueError(f"model_config.json has no threshold for class {exc}") from exc

    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = load_checkpoint(model_dir / model_config["weights_file"], map_location="cpu")
    if tuple(checkpoint["class_mapping"]) != classes:
        raise ValueError("Checkpoint class mapping does not match the deployment class mapping")

    model = build_densenet121(
        num_classes=len(classes),
        dropout=float(model_config["dropout"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(dev).eval()

    transform = build_transforms(model_config["preprocessing"], augment=False)
    return ModelBundle(
        model=model,
        classes=classes,
        thresholds=thresholds,
        transform=transform,
        device=dev,
        model_version=str(model_config.get("model_version", "unknown")),
    )
