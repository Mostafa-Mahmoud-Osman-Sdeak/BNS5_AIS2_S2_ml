"""Inference interface for DiagnoTech Model 2, designed to be called from a backend service.

Python usage:
    from inference import DiagnoTechPredictor
    predictor = DiagnoTechPredictor.from_directory("deployment/")
    result = predictor.predict("chest_xray.png")          # path, bytes, or PIL image
    results = predictor.predict_batch([path_a, path_b])

CLI smoke test:
    python inference.py --model-dir deployment/ chest_xray.png

Preprocessing is the same function used in training (diagnotech.preprocessing.build_transforms),
and the class order comes from class_mapping.json, so no separate code path can drift.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model_loader import ModelBundle, load_model_bundle  # noqa: E402

ImageInput = Union[str, Path, bytes, bytearray, Image.Image]

RESEARCH_DISCLAIMER = (
    "Research prototype for decision support only. Not a validated diagnostic device. "
    "All outputs must be reviewed by a qualified clinician."
)


class DiagnoTechPredictor:
    """Stateless wrapper around a loaded bundle. Safe to share across requests in one process."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_directory(cls, model_dir: str | Path, device: str | torch.device | None = None) -> "DiagnoTechPredictor":
        return cls(load_model_bundle(model_dir, device))

    @staticmethod
    def _load_rgb(source: ImageInput) -> Image.Image:
        try:
            if isinstance(source, Image.Image):
                return source.convert("RGB")
            if isinstance(source, (bytes, bytearray)):
                with Image.open(io.BytesIO(bytes(source))) as img:
                    return img.convert("RGB")
            with Image.open(source) as img:
                return img.convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Input is not a readable image: {exc}") from exc

    @torch.inference_mode()
    def predict_batch(self, sources: Sequence[ImageInput]) -> list[dict]:
        """Run one forward pass over a list of images and return one result dict per image."""
        if len(sources) == 0:
            return []
        tensors = [self.bundle.transform(self._load_rgb(src)) for src in sources]
        batch = torch.stack(tensors).to(self.bundle.device)
        logits = self.bundle.model(batch)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        return [self._format(row) for row in probs]

    def predict(self, source: ImageInput) -> dict:
        return self.predict_batch([source])[0]

    def _format(self, probs: np.ndarray) -> dict:
        classes = self.bundle.classes
        thresholds = self.bundle.thresholds
        order = np.argsort(-probs, kind="stable")

        probabilities = {classes[i]: round(float(probs[i]), 6) for i in order}
        top_3 = [
            {"rank": rank, "label": classes[i], "probability": round(float(probs[i]), 6)}
            for rank, i in enumerate(order[:3], start=1)
        ]
        predicted = [classes[i] for i in order if probs[i] >= thresholds[classes[i]]]

        return {
            "predicted_labels": predicted,
            "top_3": top_3,
            "probabilities": probabilities,
            "thresholds": dict(thresholds),
            "model_version": self.bundle.model_version,
            "disclaimer": RESEARCH_DISCLAIMER,
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="DiagnoTech Model 2 inference smoke test")
    parser.add_argument("--model-dir", required=True, help="Directory with best_model.pth, class_mapping.json, model_config.json")
    parser.add_argument("images", nargs="+", help="One or more chest X-ray image paths")
    args = parser.parse_args()

    predictor = DiagnoTechPredictor.from_directory(args.model_dir)
    results = predictor.predict_batch(args.images)
    for path, result in zip(args.images, results):
        print(json.dumps({"image": path, **result}, indent=2))


if __name__ == "__main__":
    _main()
