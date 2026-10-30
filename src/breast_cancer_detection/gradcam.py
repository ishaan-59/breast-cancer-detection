"""Grad-CAM for qualitative classifier-attention inspection."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import TARGET_HEIGHT, TARGET_WIDTH


class GradCAM:
    """Generate a CAM and immediately remove the temporary forward hook."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, predicted_label: int) -> np.ndarray:
        activations: torch.Tensor | None = None
        gradients: torch.Tensor | None = None

        def forward_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            nonlocal activations, gradients
            activations = output

            def save_gradient(gradient: torch.Tensor) -> None:
                nonlocal gradients
                gradients = gradient

            output.register_hook(save_gradient)

        handle = self.target_layer.register_forward_hook(forward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logit = self.model(input_tensor).squeeze()
                target_score = logit if int(predicted_label) == 1 else -logit
                target_score.backward()

            if activations is None or gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture tensors")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=(TARGET_HEIGHT, TARGET_WIDTH),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.detach().cpu().numpy()
        finally:
            handle.remove()
