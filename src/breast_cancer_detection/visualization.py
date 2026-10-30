"""Visualization utilities for combined inference results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .inference import InferenceResult


def save_inference_figure(
    image: np.ndarray,
    result: InferenceResult,
    output_path: str | Path,
) -> None:
    display_image = image.astype(np.float32) / 65535.0
    figure, axes = plt.subplots(1, 4, figsize=(20, 7))
    axes[0].imshow(display_image, cmap="gray")
    axes[0].set_title("Model-ready mammogram")
    axes[1].imshow(display_image, cmap="gray")
    axes[1].imshow(result.gradcam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM ({result.predicted_class})")
    axes[2].imshow(result.segmentation_probability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Segmentation probability")
    axes[3].imshow(display_image, cmap="gray")
    axes[3].imshow(result.segmentation_mask, cmap="spring", alpha=0.45)
    axes[3].set_title("Thresholded lesion mask")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Malignancy probability: {result.malignancy_probability:.4f}",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
