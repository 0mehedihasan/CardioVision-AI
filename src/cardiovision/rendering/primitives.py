"""
Low-level drawing primitives shared by the modality renderers.

The colour ramp and the two data-URL encoders live here because both the echo
renderer and the ECG renderer need them, and neither owns them. Everything in
this module is modality-agnostic: it knows about arrays, colours and bytes, and
nothing about hearts.

matplotlib is deliberately absent. The jet ramp is a handful of interpolated
anchors, PNG encoding is Pillow, and vector output is generated as SVG text, so
matplotlib stays a notebook-only dependency rather than a runtime one.
"""

from __future__ import annotations

import base64
from io import BytesIO
from xml.sax.saxutils import escape as _xml_escape

import numpy as np


# ============================================================
# JET COLORMAP
# ============================================================
#
# Anchor points of the classic MATLAB/matplotlib "jet" colormap, chosen so the
# saliency overlays look the same as the notebooks' plt.imshow(cmap="jet").

_JET_ANCHORS = (
    (0.000, (0, 0, 131)),
    (0.125, (0, 0, 255)),
    (0.375, (0, 255, 255)),
    (0.625, (255, 255, 0)),
    (0.875, (255, 0, 0)),
    (1.000, (128, 0, 0)),
)


def _build_jet_lut() -> np.ndarray:
    """256x3 uint8 lookup table."""
    positions = np.array([anchor[0] for anchor in _JET_ANCHORS])
    colors = np.array([anchor[1] for anchor in _JET_ANCHORS], dtype=np.float64)

    ramp = np.linspace(0.0, 1.0, 256)

    lut = np.stack(
        [np.interp(ramp, positions, colors[:, channel]) for channel in range(3)],
        axis=1,
    )

    return np.clip(lut, 0, 255).astype(np.uint8)


_JET_LUT = _build_jet_lut()


def apply_jet(values: np.ndarray) -> np.ndarray:
    """Map a float array in [0, 1] to an (..., 3) uint8 RGB image."""
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    indices = (clipped * 255.0).round().astype(np.uint8)
    return _JET_LUT[indices]


def jet_hex(value: float) -> str:
    """One point on the jet ramp as ``#rrggbb``, for SVG fills."""
    red, green, blue = _JET_LUT[int(round(float(np.clip(value, 0.0, 1.0)) * 255))]
    return f"#{red:02x}{green:02x}{blue:02x}"


# ============================================================
# ENCODING
# ============================================================


def to_png_data_url(rgb: np.ndarray, scale: int = 2) -> str:
    """
    Encode an RGB or RGBA array as a base64 PNG data URL.

    `scale` applies nearest-neighbour upscaling so 256x256 output still looks
    crisp in the UI without the browser blurring class boundaries.
    """
    try:
        from PIL import Image
    except ImportError as error:                       # pragma: no cover
        raise RuntimeError(
            "Pillow is required to render PNG output. "
            "Install it with: pip install Pillow"
        ) from error

    array = rgb
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)

    mode = "RGBA" if array.shape[2] == 4 else "RGB"
    image = Image.fromarray(array.astype(np.uint8), mode=mode)

    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=Image.NEAREST,
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_svg_data_url(svg: str) -> str:
    """
    Encode SVG markup as a base64 data URL.

    Base64 rather than percent-encoding: the markup contains quotes, hashes and
    angle brackets on every line, and a mis-escaped '#' silently truncates the
    URL in some browsers. Base64 costs a third more bytes and cannot be got
    wrong.
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def xml_text(value: object) -> str:
    """
    Escape a value for use as SVG text content or an attribute.

    Record names, lead labels and units all originate in uploaded files, so
    none of them can be interpolated into markup unescaped.
    """
    return _xml_escape(str(value), {'"': "&quot;", "'": "&apos;"})
