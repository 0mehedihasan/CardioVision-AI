"""
CardioVision AI — server-side rendering of segmentation results.

Turns mask / saliency / input arrays into base64-encoded PNG data URLs so
the frontend can display them with a plain <img> tag.

The colour ramp and PNG encoder live in ``rendering.primitives``, shared with
the ECG renderer. They are re-exported here because this module was their
original home and callers still import them from it.
"""

from __future__ import annotations

import numpy as np

from cardiovision.config import ECHO_CLASS_COLORS, ECHO_NUM_CLASSES
from cardiovision.rendering.primitives import apply_jet, to_png_data_url

__all__ = [
    "apply_jet",
    "to_png_data_url",
    "colorize_mask",
    "render_analysis_images",
    "encode_mask_payload",
]


# ============================================================
# HELPERS
# ============================================================

def _to_uint8_gray(values: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary float array to 0–255 uint8."""
    array = np.nan_to_num(values.astype(np.float32), nan=0.0)

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum - minimum < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - minimum) / (maximum - minimum)
    return (scaled * 255.0).round().astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Class-index mask -> (H, W, 3) uint8 RGB using the shared palette."""
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for class_index in range(ECHO_NUM_CLASSES):
        color = ECHO_CLASS_COLORS.get(class_index, (255, 255, 255))
        rgb[mask == class_index] = color

    return rgb


def _blend(
    base: np.ndarray,
    layer: np.ndarray,
    alpha: float,
    where: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-blend `layer` over `base`, optionally restricted to a mask."""
    base_f = base.astype(np.float32)
    layer_f = layer.astype(np.float32)

    blended = base_f * (1.0 - alpha) + layer_f * alpha

    if where is not None:
        result = base_f.copy()
        result[where] = blended[where]
        blended = result

    return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# RESULT PANELS
# ============================================================

def render_analysis_images(
    normalised_input: np.ndarray,
    prediction: np.ndarray,
    saliency: np.ndarray,
    mask_alpha: float = 0.45,
    saliency_alpha: float = 0.45,
) -> dict[str, str]:
    """
    Produce the display panels, mirroring the five-panel figure at the end
    of 02_Echo_Training.ipynb.
    """
    gray = _to_uint8_gray(normalised_input)
    gray_rgb = _gray_to_rgb(gray)

    mask_rgb = colorize_mask(prediction)
    foreground = prediction > 0

    overlay = _blend(gray_rgb, mask_rgb, mask_alpha, where=foreground)

    saliency_rgb = apply_jet(saliency)
    saliency_overlay = _blend(gray_rgb, saliency_rgb, saliency_alpha)

    combined = _blend(gray_rgb, mask_rgb, 0.35, where=foreground)
    combined = _blend(combined, saliency_rgb, 0.40)

    return {
        "original": to_png_data_url(gray_rgb),
        "mask": to_png_data_url(mask_rgb),
        "overlay": to_png_data_url(overlay),
        "saliency": to_png_data_url(saliency_rgb),
        "saliency_overlay": to_png_data_url(saliency_overlay),
        "combined": to_png_data_url(combined),
    }


def encode_mask_payload(prediction: np.ndarray) -> dict[str, object]:
    """
    Serialise the raw class mask for client-side canvas rendering.

    Sent as a flat row-major list, which is ~40% smaller than a nested
    array in JSON and trivial to index in JavaScript as `data[y * width + x]`.
    """
    height, width = prediction.shape

    return {
        "width": int(width),
        "height": int(height),
        "num_classes": ECHO_NUM_CLASSES,
        "layout": "row-major flat array; index = y * width + x",
        "class_colors": {
            str(index): list(color)
            for index, color in ECHO_CLASS_COLORS.items()
        },
        "data": prediction.astype(np.uint8).flatten().tolist(),
    }
