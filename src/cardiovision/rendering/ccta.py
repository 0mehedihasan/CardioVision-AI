"""
CardioVision AI — CCTA result rendering.

A 3-D mask cannot be shown as one picture. This module produces the smallest
set of views that lets a reader check the result honestly:

* three orthogonal slices through the centre of the predicted lumen, with the
  mask overlaid on the CT — the plane where there is something to look at,
  rather than the geometric middle of the array where there usually is not;
* a maximum-intensity projection of the probability map along each axis, which
  is the only view that shows the whole predicted tree at once and therefore
  the only one where fragmentation is visible;
* the Grad-CAM patch, over the CT it was computed from.

Every panel is labelled with the plane and the index it came from, because a
single slice through a volume invites being read as the whole result. Slice
indices are in the analysed (1 mm isotropic) grid, not the source grid.

Rendering never raises. A figure that cannot be produced comes back missing
with a note beside it, because losing the mask because a colour map failed
would be a strictly worse outcome than losing the picture.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cardiovision.config import CCTA_CLASS_COLORS
from cardiovision.rendering.primitives import apply_jet, to_png_data_url

__all__ = [
    "render_ccta_images",
    "slice_labels",
]


# Which array axis each named plane cuts across. The volume is in the source
# file's own axis order (see preprocessing.ccta_io), so these are labelled by
# array axis rather than by anatomical plane — calling axis 2 "axial" would be
# a claim about patient orientation that the loader does not verify.
_PLANES = (
    ("axis0", 0),
    ("axis1", 1),
    ("axis2", 2),
)

_MASK_ALPHA = 0.50
_CAM_ALPHA = 0.45

# Window used for display only. The model sees the full [-1000, 1000] HU range;
# a coronary reader sees nothing useful in it, because bone and contrast both
# saturate. This is roughly a mediastinal window mapped into the normalised
# [-1, 1] scale: HU -150 to 350.
_DISPLAY_LOW = -150.0 / 1000.0
_DISPLAY_HIGH = 350.0 / 1000.0


def _to_gray(plane: np.ndarray) -> np.ndarray:
    """Normalised CT slice to 0-255 uint8 through a fixed display window."""
    values = np.nan_to_num(plane.astype(np.float32), nan=_DISPLAY_LOW)
    values = np.clip(values, _DISPLAY_LOW, _DISPLAY_HIGH)
    span = _DISPLAY_HIGH - _DISPLAY_LOW

    scaled = (values - _DISPLAY_LOW) / span if span > 0 else values * 0.0
    return (scaled * 255.0).round().astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def _blend(
    base: np.ndarray,
    layer: np.ndarray,
    alpha: float,
    where: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_f = base.astype(np.float32)
    layer_f = layer.astype(np.float32)
    blended = base_f * (1.0 - alpha) + layer_f * alpha

    if where is not None:
        result = base_f.copy()
        result[where] = blended[where]
        blended = result

    return np.clip(blended, 0, 255).astype(np.uint8)


def _colorize_mask(mask_plane: np.ndarray) -> np.ndarray:
    height, width = mask_plane.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[mask_plane > 0] = CCTA_CLASS_COLORS[1]
    return rgb


def _best_index(mask: np.ndarray, probability: np.ndarray, axis: int) -> int:
    """
    The most informative slice along one axis.

    The slice with the most predicted lumen; failing that, the slice with the
    highest total probability. Never the geometric centre by default — on a mask
    occupying a tenth of a percent of the volume the centre slice is usually
    empty, and an empty panel reads as "the model found nothing".
    """
    others = tuple(a for a in range(3) if a != axis)

    per_slice = mask.sum(axis=others)
    if per_slice.max() > 0:
        return int(np.argmax(per_slice))

    per_slice = probability.sum(axis=others)
    if per_slice.size and float(per_slice.max()) > 0.0:
        return int(np.argmax(per_slice))

    return int(mask.shape[axis] // 2)


def _take(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def slice_labels(
    mask: np.ndarray,
    probability: np.ndarray,
) -> dict[str, int]:
    """The slice index chosen for each plane, so the caller can report it."""
    return {
        name: _best_index(mask, probability, axis) for name, axis in _PLANES
    }


def render_ccta_images(
    volume: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    analysed: Optional[np.ndarray] = None,
    gradcam: Optional[np.ndarray] = None,
    gradcam_origin: Optional[tuple[int, int, int]] = None,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """
    Render the CCTA panels.

    Returns the images as data URLs, a metadata dict describing what each panel
    is a view of, and any notes about panels that could not be produced.
    """
    images: dict[str, str] = {}
    notes: list[str] = []
    meta: dict[str, Any] = {
        "planes": {},
        "display_window_hu": [-150, 350],
        "display_note": (
            "Slices are windowed to -150..350 HU for display. The model saw the "
            "full -1000..1000 HU range."
        ),
        "axis_note": (
            "Planes are named by array axis, not by anatomical plane. The "
            "loader does not reorient the volume, so which axis is axial "
            "depends on how the study was stored."
        ),
    }

    try:
        indices = slice_labels(mask, probability)
    except Exception as error:      # pragma: no cover - defensive
        notes.append(f"Slice selection failed ({error}); no slice views were rendered.")
        indices = {}

    for name, axis in _PLANES:
        if name not in indices:
            continue

        index = indices[name]

        try:
            ct_plane = _take(volume, axis, index)
            mask_plane = _take(mask, axis, index)
            probability_plane = _take(probability, axis, index)

            gray = _gray_to_rgb(_to_gray(ct_plane))
            lumen = mask_plane > 0

            overlay = _blend(gray, _colorize_mask(mask_plane), _MASK_ALPHA,
                             where=lumen)

            images[f"{name}_ct"] = to_png_data_url(gray)
            images[f"{name}_overlay"] = to_png_data_url(overlay)
            images[f"{name}_probability"] = to_png_data_url(
                apply_jet(np.clip(probability_plane, 0.0, 1.0))
            )

            plane_meta: dict[str, Any] = {
                "axis": axis,
                "index": int(index),
                "of": int(mask.shape[axis]),
                "lumen_voxels_in_slice": int(lumen.sum()),
                "selected_by": (
                    "most predicted lumen" if mask.sum() > 0
                    else "highest total probability"
                ),
            }

            if analysed is not None:
                analysed_plane = _take(analysed, axis, index)
                covered = float(analysed_plane.mean()) if analysed_plane.size else 0.0
                plane_meta["analysed_fraction_of_slice"] = round(covered, 4)
                if covered < 0.999:
                    plane_meta["warning"] = (
                        f"{(1.0 - covered) * 100:.0f}% of this slice was outside "
                        "the analysed region. Black there means not looked at, "
                        "not absent."
                    )

            meta["planes"][name] = plane_meta
        except Exception as error:      # pragma: no cover - defensive
            notes.append(
                f"The {name} slice views could not be rendered ({error}). The "
                "measurements are unaffected."
            )

    # ---- projections ------------------------------------------
    #
    # The one view where a fragmented mask is obvious. Worth more than any
    # single slice for judging whether this model's output is usable.

    for name, axis in _PLANES:
        try:
            projection = probability.max(axis=axis)
            images[f"{name}_mip"] = to_png_data_url(
                apply_jet(np.clip(projection, 0.0, 1.0))
            )
        except Exception as error:      # pragma: no cover - defensive
            notes.append(
                f"The {name} maximum-intensity projection could not be "
                f"rendered ({error})."
            )

    if images:
        meta["projection_note"] = (
            "Maximum-intensity projections collapse the whole volume onto one "
            "plane. Use them to judge whether the predicted lumen forms a few "
            "connected vessels or many disconnected fragments."
        )

    # ---- Grad-CAM ---------------------------------------------

    if gradcam is not None and gradcam_origin is not None:
        try:
            z, y, x = (int(v) for v in gradcam_origin)
            dz, dy, dx = (int(d) for d in gradcam.shape)

            patch_ct = volume[z:z + dz, y:y + dy, x:x + dx]
            patch_mask = mask[z:z + dz, y:y + dy, x:x + dx]

            if patch_ct.shape != gradcam.shape:
                raise ValueError(
                    f"CAM shape {gradcam.shape} does not fit the volume at "
                    f"offset {(z, y, x)} (got {patch_ct.shape})"
                )

            # The CAM slice with the most attention, so the panel shows the
            # thing the CAM is pointing at.
            per_slice = gradcam.sum(axis=(1, 2))
            local = int(np.argmax(per_slice)) if per_slice.size else dz // 2

            gray = _gray_to_rgb(_to_gray(patch_ct[local]))
            heat = apply_jet(np.clip(gradcam[local], 0.0, 1.0))

            images["gradcam"] = to_png_data_url(heat)
            images["gradcam_overlay"] = to_png_data_url(
                _blend(gray, heat, _CAM_ALPHA)
            )
            images["gradcam_ct"] = to_png_data_url(gray)
            images["gradcam_mask_overlay"] = to_png_data_url(
                _blend(gray, _colorize_mask(patch_mask[local]), _MASK_ALPHA,
                       where=patch_mask[local] > 0)
            )

            meta["gradcam"] = {
                "origin": [z, y, x],
                "shape": [dz, dy, dx],
                "slice_index_in_patch": local,
                "slice_index_in_volume": z + local,
                "scope": (
                    "One 96x96x96 patch, not the whole volume. Attention "
                    "outside this patch was not computed."
                ),
            }
        except Exception as error:      # pragma: no cover - defensive
            notes.append(
                f"The Grad-CAM panels could not be rendered ({error}). The "
                "segmentation itself is unaffected."
            )

    return images, meta, notes
