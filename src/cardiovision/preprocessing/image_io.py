"""
CardioVision AI — medical image loading and preprocessing.

Accepts PNG / JPEG, NIfTI (.nii, .nii.gz) and DICOM (.dcm), normalises
everything to a single 2D grayscale float array, and applies the exact
preprocessing used during training.

CRITICAL: `preprocess_to_tensor` is a byte-for-byte reimplementation of
`preprocess_image` in notebooks/02_Echo_Training.ipynb. If you change it,
the trained checkpoint's predictions become invalid. Do not "improve" it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cardiovision.config import ECHO_IMAGE_SIZE


class UnsupportedImageError(ValueError):
    """Raised when a file cannot be interpreted as a 2D echo image."""


@dataclass
class LoadedImage:
    """A modality-agnostic 2D grayscale image plus provenance."""

    pixels: np.ndarray                       # 2D float32, native intensities
    image_format: str                        # "png" | "jpeg" | "nifti" | "dicom"
    original_shape: tuple[int, int]
    pixel_spacing_mm: Optional[tuple[float, float]] = None   # (row, col)
    frame_index: Optional[int] = None
    frame_count: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    # Geometry the caller explicitly asked for, recorded so the response can
    # state what was done rather than leaving the transform invisible.
    rotation_applied: int = 0                # 0 | 90 | 180 | 270, CCW
    flip_applied: bool = False               # horizontal mirror
    oriented_shape: Optional[tuple[int, int]] = None   # shape after rotation

    @property
    def has_spatial_calibration(self) -> bool:
        return self.pixel_spacing_mm is not None

    @property
    def was_reoriented(self) -> bool:
        return bool(self.rotation_applied) or self.flip_applied


# ============================================================
# FORMAT DETECTION
# ============================================================

def detect_format(filename: str, data: bytes) -> str:
    """
    Detect format from magic bytes first, filename second. Magic bytes are
    authoritative because browsers and clinical export tools both lie about
    extensions regularly.
    """
    name = (filename or "").lower()

    # DICOM: "DICM" at byte offset 128.
    if len(data) > 132 and data[128:132] == b"DICM":
        return "dicom"

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"

    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"

    # gzip magic — assume .nii.gz
    if data[:2] == b"\x1f\x8b":
        return "nifti"

    # Uncompressed NIfTI-1: "n+1\0" or "ni1\0" at offset 344.
    if len(data) > 348 and data[344:348] in (b"n+1\x00", b"ni1\x00"):
        return "nifti"

    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if name.endswith((".dcm", ".dicom")):
        return "dicom"
    if name.endswith(".png"):
        return "png"
    if name.endswith((".jpg", ".jpeg")):
        return "jpeg"

    raise UnsupportedImageError(
        "Could not determine the image format. Supported formats are "
        "PNG, JPEG, NIfTI (.nii / .nii.gz) and DICOM (.dcm)."
    )


# ============================================================
# PNG / JPEG
# ============================================================

def _load_pillow(data: bytes, image_format: str) -> LoadedImage:
    try:
        from PIL import Image
    except ImportError as error:                       # pragma: no cover
        raise UnsupportedImageError(
            "Pillow is required to read PNG/JPEG files. "
            "Install it with: pip install Pillow"
        ) from error

    notes: list[str] = []

    with Image.open(BytesIO(data)) as image:
        if image.mode not in ("L", "I;16", "I", "F"):
            notes.append(
                f"Converted from {image.mode} to grayscale before inference."
            )
            image = image.convert("L")

        pixels = np.asarray(image).astype(np.float32)

    if pixels.ndim != 2:
        raise UnsupportedImageError(
            f"Expected a 2D grayscale image, got shape {pixels.shape}."
        )

    notes.append(
        "PNG/JPEG carries no pixel spacing, so structure sizes are reported "
        "as area fractions only, not in cm²."
    )

    return LoadedImage(
        pixels=pixels,
        image_format=image_format,
        original_shape=(pixels.shape[0], pixels.shape[1]),
        pixel_spacing_mm=None,
        notes=notes,
    )


# ============================================================
# NIfTI
# ============================================================

def _load_nifti(data: bytes, filename: str) -> LoadedImage:
    try:
        import nibabel as nib
    except ImportError as error:                       # pragma: no cover
        raise UnsupportedImageError(
            "nibabel is required to read NIfTI files. "
            "Install it with: pip install nibabel"
        ) from error

    notes: list[str] = []

    suffix = ".nii.gz" if (data[:2] == b"\x1f\x8b"
                           or (filename or "").lower().endswith(".nii.gz")) else ".nii"

    # nibabel needs a real path for gzipped streams, so round-trip via a
    # temp file. Cleaned up in `finally` even if parsing raises.
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)

        image = nib.load(temp_path)

        # get_fdata() is used here *deliberately*, with no transpose or
        # reorientation, because that is exactly what the training notebook
        # did. Any axis fix-up here would silently rotate the input relative
        # to what the model learned.
        pixels = np.asanyarray(image.dataobj).astype(np.float32)
        zooms = image.header.get_zooms()
    finally:
        try:
            os.unlink(temp_path)
        except OSError:                                # pragma: no cover
            pass

    # Track which ORIGINAL axes survive the reshaping below, so the header's
    # zooms stay attached to the axes they actually describe. Reading
    # zooms[0], zooms[1] blindly picks up the spacing of a dropped singleton
    # axis or of the time axis, which yields confident but wrong cm² areas —
    # the worst failure mode this loader has.
    axes = [index for index, size in enumerate(pixels.shape) if size != 1]
    pixels = np.squeeze(pixels)

    if pixels.ndim == 3:
        # A 3D volume or a sequence. Take the middle slice along the
        # shortest axis, which for CAMUS-style 2D+t data is the time axis.
        short_axis = int(np.argmin(pixels.shape))
        middle = pixels.shape[short_axis] // 2
        pixels = np.take(pixels, middle, axis=short_axis)
        dropped = axes.pop(short_axis)
        notes.append(
            f"Input was 3D; used middle slice {middle} along original "
            f"axis {dropped} (shortest axis, typically time)."
        )

    if pixels.ndim != 2:
        raise UnsupportedImageError(
            f"Expected a 2D NIfTI image, got shape {pixels.shape} after squeeze."
        )

    spacing: Optional[tuple[float, float]] = None
    if zooms is not None and len(axes) == 2 and len(zooms) > max(axes):
        row_mm, col_mm = float(zooms[axes[0]]), float(zooms[axes[1]])
        if row_mm > 0 and col_mm > 0:
            spacing = (row_mm, col_mm)
            notes.append(
                f"Pixel spacing from NIfTI header axes {axes[0]},{axes[1]}: "
                f"{row_mm:.3f} × {col_mm:.3f} mm."
            )

    return LoadedImage(
        pixels=pixels,
        image_format="nifti",
        original_shape=(pixels.shape[0], pixels.shape[1]),
        pixel_spacing_mm=spacing,
        notes=notes,
    )


# ============================================================
# DICOM
# ============================================================

def _dicom_spacing(dataset) -> tuple[Optional[tuple[float, float]], list[str]]:
    """
    Resolve pixel spacing in mm/px, preferring ultrasound region
    calibration, which is what echo machines actually populate.
    """
    notes: list[str] = []

    regions = getattr(dataset, "SequenceOfUltrasoundRegions", None)
    if regions:
        region = regions[0]
        delta_x = getattr(region, "PhysicalDeltaX", None)
        delta_y = getattr(region, "PhysicalDeltaY", None)

        if delta_x and delta_y:
            # Unit 3 == cm. Absent tags are NOT assumed to be cm: guessing the
            # unit here would turn an unknown scale into an authoritative cm²
            # figure. No units tag means no absolute calibration.
            units_x = getattr(region, "PhysicalUnitsXDirection", None)
            units_y = getattr(region, "PhysicalUnitsYDirection", None)

            if units_x == 3 and units_y == 3:
                row_mm = abs(float(delta_y)) * 10.0
                col_mm = abs(float(delta_x)) * 10.0
                notes.append(
                    "Pixel spacing from SequenceOfUltrasoundRegions: "
                    f"{row_mm:.3f} × {col_mm:.3f} mm."
                )
                return (row_mm, col_mm), notes

            if units_x is None or units_y is None:
                notes.append(
                    "Ultrasound region spacing present but the physical units "
                    "tag is missing, so the scale is unknown and areas are not "
                    "reported in cm²."
                )
            else:
                notes.append(
                    "Ultrasound region spacing present but not in cm; ignored "
                    "so areas are not silently wrong."
                )

    for attribute in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(dataset, attribute, None)
        if value is not None and len(value) >= 2:
            row_mm, col_mm = float(value[0]), float(value[1])
            if row_mm > 0 and col_mm > 0:
                notes.append(
                    f"Pixel spacing from {attribute}: "
                    f"{row_mm:.3f} × {col_mm:.3f} mm."
                )
                return (row_mm, col_mm), notes

    notes.append(
        "No pixel spacing in the DICOM header, so structure sizes are "
        "reported as area fractions only, not in cm²."
    )
    return None, notes


def _load_dicom(data: bytes, frame: Optional[int]) -> LoadedImage:
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import convert_color_space
    except ImportError as error:                       # pragma: no cover
        raise UnsupportedImageError(
            "pydicom is required to read DICOM files. "
            "Install it with: pip install pydicom pylibjpeg pylibjpeg-libjpeg"
        ) from error

    notes: list[str] = []

    dataset = pydicom.dcmread(BytesIO(data), force=True)

    try:
        array = dataset.pixel_array
    except Exception as error:
        raise UnsupportedImageError(
            "Could not decode DICOM pixel data. Compressed transfer syntaxes "
            "need the pylibjpeg extras: "
            "pip install pylibjpeg pylibjpeg-libjpeg. "
            f"Underlying error: {error}"
        ) from error

    array = np.asarray(array)

    photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "")
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    frame_count = int(getattr(dataset, "NumberOfFrames", 1) or 1)

    # ---- Multi-frame cine loop: pick one frame -------------------
    selected_frame: Optional[int] = None

    # The ndim guard matters: a header can claim NumberOfFrames > 1 while the
    # pixel data decodes to a single 2D frame. Slicing that would silently
    # take one ROW of the image and segment a 1-D strip.
    frames_in_array = array.ndim >= 3 and not (
        array.ndim == 3 and array.shape[-1] in (3, 4)
    )

    if frame_count > 1 and frames_in_array:
        available = int(array.shape[0])

        if available != frame_count:
            notes.append(
                f"Header declares {frame_count} frames but the decoded pixel "
                f"data holds {available}; using the decoded count."
            )
            frame_count = available

        if frame is None:
            selected_frame = frame_count // 2
            notes.append(
                f"Multi-frame cine loop with {frame_count} frames; "
                f"defaulted to the middle frame ({selected_frame}). "
                "Pass ?frame=N to choose a specific frame."
            )
        else:
            if not 0 <= frame < frame_count:
                raise UnsupportedImageError(
                    f"frame={frame} is out of range; this study has "
                    f"{frame_count} frames (0–{frame_count - 1})."
                )
            selected_frame = frame
            notes.append(
                f"Using frame {selected_frame} of {frame_count}."
            )

        array = array[selected_frame]

    elif frame_count > 1:
        notes.append(
            f"Header declares {frame_count} frames but the pixel data decoded "
            "as a single image; treated as one frame."
        )
        frame_count = 1

    # ---- Colour handling ----------------------------------------
    # Ultrasound is frequently stored as YBR_FULL_422 RGB even when the
    # underlying image is grayscale.
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        if photometric.startswith("YBR"):
            try:
                array = convert_color_space(
                    array, photometric, "RGB", per_frame=False
                )
                notes.append(f"Converted {photometric} to RGB.")
            except Exception:
                notes.append(
                    f"Could not convert {photometric} to RGB; "
                    "used raw channel values."
                )

        rgb = array[..., :3].astype(np.float32)
        # Rec. 601 luma — standard grayscale conversion.
        array = (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )
        notes.append("Converted colour ultrasound frame to grayscale.")
    elif samples > 1:                                  # pragma: no cover
        notes.append(
            f"SamplesPerPixel={samples} but pixel array is not channel-last; "
            "used raw values."
        )

    array = np.squeeze(np.asarray(array)).astype(np.float32)

    if array.ndim == 3:
        short_axis = int(np.argmin(array.shape))
        middle = array.shape[short_axis] // 2
        array = np.take(array, middle, axis=short_axis)
        notes.append(
            f"Pixel data still 3D; used middle slice {middle} "
            f"along axis {short_axis}."
        )

    if array.ndim != 2:
        raise UnsupportedImageError(
            f"Expected a 2D DICOM frame, got shape {array.shape}."
        )

    # ---- Rescale slope / intercept ------------------------------
    slope = getattr(dataset, "RescaleSlope", None)
    intercept = getattr(dataset, "RescaleIntercept", None)
    if slope is not None and intercept is not None:
        array = array * float(slope) + float(intercept)
        notes.append("Applied RescaleSlope/RescaleIntercept.")

    if photometric == "MONOCHROME1":
        # MONOCHROME1 is inverted: high value == dark.
        array = array.max() - array
        notes.append("Inverted MONOCHROME1 photometric interpretation.")

    spacing, spacing_notes = _dicom_spacing(dataset)
    notes.extend(spacing_notes)

    return LoadedImage(
        pixels=array,
        image_format="dicom",
        original_shape=(array.shape[0], array.shape[1]),
        pixel_spacing_mm=spacing,
        frame_index=selected_frame,
        frame_count=frame_count if frame_count > 1 else None,
        notes=notes,
    )


# ============================================================
# PUBLIC LOADER
# ============================================================

# The model was trained on CAMUS NIfTI arrays used with no reorientation, in
# which the ultrasound sector's apex points LEFT and the beam opens to the
# right. This is verifiable in the notebook's own saved figure
# (models/echo/__results___0_129.png).
#
# A conventional echocardiogram is displayed apex-UP with the beam opening
# downward, so a screenshot or a DICOM frame is typically rotated 90° from
# the distribution the model learned. We do NOT guess a correction: rotating
# silently would be a hidden transformation with no audit trail, and the
# correct rotation depends on how the operator exported the image. Instead the
# caller passes an explicit rotation and the UI offers it as a control.
ORIENTATION_NOTE = (
    "This model was trained on images whose ultrasound sector apex points "
    "LEFT (beam opening rightward). Conventional echo displays are apex-up, "
    "which is a 90° rotation away from that. If the segmentation looks "
    "anatomically wrong, re-run with a rotation."
)

DISPLAY_ORIENTED_FORMATS = ("png", "jpeg", "dicom")


def load_echo_image(
    filename: str,
    data: bytes,
    frame: Optional[int] = None,
    rotate: int = 0,
    flip: bool = False,
) -> LoadedImage:
    """
    Load an echocardiography image from raw bytes into a 2D float array.

    `frame`  selects a frame from a multi-frame DICOM cine loop; it is
             ignored for single-frame formats.
    `rotate` counter-clockwise rotation in degrees, one of 0/90/180/270,
             applied after loading. Default 0 — nothing is rotated implicitly.
    `flip`   mirror horizontally, applied after the rotation.
    """
    if not data:
        raise UnsupportedImageError("The uploaded file is empty.")

    if rotate not in (0, 90, 180, 270):
        raise UnsupportedImageError(
            f"rotate must be 0, 90, 180 or 270 degrees, got {rotate}."
        )

    image_format = detect_format(filename, data)

    if image_format in ("png", "jpeg"):
        loaded = _load_pillow(data, image_format)
    elif image_format == "nifti":
        loaded = _load_nifti(data, filename)
    elif image_format == "dicom":
        loaded = _load_dicom(data, frame)
    else:                                              # pragma: no cover
        raise UnsupportedImageError(f"Unsupported format: {image_format}")

    if min(loaded.original_shape) < 16:
        raise UnsupportedImageError(
            f"Image is too small to segment: {loaded.original_shape}."
        )

    if not np.isfinite(loaded.pixels).any():
        raise UnsupportedImageError(
            "Image contains no finite pixel values."
        )

    # Guard against NaN/inf reaching the percentile computation.
    if not np.isfinite(loaded.pixels).all():
        loaded.pixels = np.nan_to_num(
            loaded.pixels, nan=0.0, posinf=0.0, neginf=0.0
        )
        loaded.notes.append("Replaced non-finite pixel values with 0.")

    # ---- explicit geometry adjustments ---------------------------
    if rotate:
        loaded.pixels = np.rot90(loaded.pixels, k=rotate // 90)
        loaded.notes.append(f"Rotated {rotate}° counter-clockwise as requested.")

        if rotate in (90, 270) and loaded.pixel_spacing_mm:
            # Row and column spacing swap with a quarter turn, otherwise the
            # cm² figure would be computed against the wrong axes.
            row_mm, col_mm = loaded.pixel_spacing_mm
            loaded.pixel_spacing_mm = (col_mm, row_mm)

    if flip:
        loaded.pixels = np.fliplr(loaded.pixels)
        loaded.notes.append("Mirrored horizontally as requested.")

    loaded.pixels = np.ascontiguousarray(loaded.pixels)
    loaded.rotation_applied = rotate
    loaded.flip_applied = flip
    loaded.oriented_shape = (loaded.pixels.shape[0], loaded.pixels.shape[1])

    if image_format in DISPLAY_ORIENTED_FORMATS and not rotate:
        loaded.notes.append(ORIENTATION_NOTE)

    return loaded


# ============================================================
# PREPROCESSING — MUST MATCH THE TRAINING NOTEBOOK
# ============================================================

def preprocess_to_tensor(pixels: np.ndarray) -> torch.Tensor:
    """
    Reproduce `preprocess_image` from 02_Echo_Training.ipynb exactly:

        1. cast to float32
        2. clip to the 1st–99th intensity percentile
        3. min-max normalise to [0, 1] with a 1e-8 epsilon
        4. bilinear resize to 256×256, align_corners=False

    Returns a (1, 1, 256, 256) tensor on the CPU.
    """
    image = pixels.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)
    image = np.clip(image, low, high)

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
    tensor = tensor[None, None, ...]

    tensor = F.interpolate(
        tensor,
        size=(ECHO_IMAGE_SIZE, ECHO_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return tensor
