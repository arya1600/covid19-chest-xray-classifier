"""Grad-CAM for BaselineCNN: hook the last conv block's output, weight its
channels by the gradient of the target class score, and produce a
coarse spatial "where did the model look" heatmap.

Used for two purposes in this project: (1) sanity-checking a model's own
predictions (does a correct prediction actually attend to the lungs?),
and (2) as qualitative, visual evidence for the source-confound question
alongside the quantitative Experiment D comparison — if a model with a
suspiciously perfect confusion matrix is attending to image borders or
non-anatomical regions rather than lung fields, that corroborates the
shortcut-learning concern; if it attends to lungs, that's evidence
against it. Grad-CAM is a coarse, sometimes-misleading localization
method — it is one piece of evidence, not ground truth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib.cm as mpl_cm
except ImportError:  # pragma: no cover
    mpl_cm = None


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """input_tensor: (1, C, H, W). Returns (cam in [0,1] as (h, w) array,
        the target class actually used).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1))  # (1, h, w)
        cam = cam.squeeze(0).cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class


def overlay_heatmap(base_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """`base_image`: the (grayscale, already resized/padded) image shown to
    the model. `cam`: Grad-CAM output at the feature-map resolution — it's
    upsampled to `base_image`'s size before blending.
    """
    if mpl_cm is None:
        raise ImportError("matplotlib is required for overlay_heatmap")

    size = base_image.size  # (W, H)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

    heat_rgba = mpl_cm.jet(cam_resized)  # (H, W, 4) float in [0,1]
    heat_rgb = heat_rgba[:, :, :3]

    base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    blended = np.clip((1 - alpha) * base_rgb + alpha * heat_rgb, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
