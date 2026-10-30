"""Checkpoint loading and combined classification/segmentation inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import CLASSIFIER_THRESHOLD, SEGMENTATION_THRESHOLD
from .gradcam import GradCAM
from .models import ResNet50UNet, build_classifier
from .preprocessing import prepare_image_tensor


@dataclass(frozen=True)
class InferenceResult:
    malignancy_probability: float
    predicted_label: int
    predicted_class: str
    gradcam: np.ndarray
    segmentation_probability: np.ndarray
    segmentation_mask: np.ndarray


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognized model state dictionary")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # The supplied checkpoints contain NumPy scalar metadata alongside tensors.
    # Keep weights-only loading enabled and allow only the small set of NumPy
    # scalar/dtype classes required to decode that trusted metadata.
    numpy_safe_globals = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    return _checkpoint_state(checkpoint)


class BreastCancerPipeline:
    """Run the locked classifier, Grad-CAM, and segmentation models."""

    def __init__(
        self,
        classifier_checkpoint: str | Path,
        segmentation_checkpoint: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classifier: nn.Module = build_classifier(pretrained=False).to(self.device)
        self.classifier.load_state_dict(
            _load_checkpoint(Path(classifier_checkpoint), self.device)
        )
        self.segmenter: nn.Module = ResNet50UNet().to(self.device)
        self.segmenter.load_state_dict(
            _load_checkpoint(Path(segmentation_checkpoint), self.device)
        )
        self.classifier.eval()
        self.segmenter.eval()

    def predict(self, image: np.ndarray) -> InferenceResult:
        tensor = prepare_image_tensor(image, self.device)
        with torch.no_grad():
            probability = torch.sigmoid(self.classifier(tensor).squeeze()).item()
            segmentation_probability = torch.sigmoid(self.segmenter(tensor))[0, 0]

        predicted_label = int(probability >= CLASSIFIER_THRESHOLD)
        cam = GradCAM(self.classifier, self.classifier.layer4[-1]).generate(
            tensor,
            predicted_label,
        )
        probability_map = segmentation_probability.cpu().numpy()
        binary_mask = (probability_map >= SEGMENTATION_THRESHOLD).astype(np.uint8)
        return InferenceResult(
            malignancy_probability=float(probability),
            predicted_label=predicted_label,
            predicted_class="malignant" if predicted_label else "non-malignant",
            gradcam=cam,
            segmentation_probability=probability_map,
            segmentation_mask=binary_mask,
        )
