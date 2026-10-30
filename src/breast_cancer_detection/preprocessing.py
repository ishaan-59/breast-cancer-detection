"""Input validation and normalization for model-ready mammograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD, TARGET_HEIGHT, TARGET_WIDTH


def load_model_ready_image(path: str | Path) -> np.ndarray:
    """Load a prepared uint16 NPY image; raw DICOM input is not accepted."""
    image = np.load(Path(path))
    validate_model_ready_image(image)
    return image


def validate_model_ready_image(image: np.ndarray) -> None:
    expected_shape = (TARGET_HEIGHT, TARGET_WIDTH)
    if image.shape != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {image.shape}")
    if image.dtype != np.uint16:
        raise TypeError(f"Expected uint16 image, got {image.dtype}")


def prepare_image_tensor(
    image: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Apply the exact scaling and ImageNet normalization used in training."""
    validate_model_ready_image(image)
    image_float = image.astype(np.float32) / 65535.0
    tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)
    tensor = tensor.repeat(1, 3, 1, 1).to(device)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    return (tensor - mean) / std
