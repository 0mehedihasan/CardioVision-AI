"""
CardioVision AI — CCTA volume loading and preprocessing.

Reads a whole 3-D cardiac CT volume and puts it into exactly the state the
trained ``Small3DUNet`` saw during training: 1 mm isotropic voxels, Hounsfield
units clipped to [-1000, 1000], rescaled to [-1, 1].

Why this is a separate module from ``image_io``
-----------------------------------------------
``image_io`` deliberately reduces everything to one 2-D frame, because that is
what the echo model consumes. Feeding a CCTA study through it would silently
discard the other 500 slices. Volumes need their own path, their own spacing
handling and their own size guards.

The arithmetic below mirrors notebooks/01_CCTA_Training.ipynb step for step.
Deviating from it — a different HU window, a different interpolation order, a
different target spacing — produces a plausible-looking mask computed from
inputs the weights have never seen, which is worse than an error.

Two things this module will not do
---------------------------------
* Guess voxel spacing. Resampling to 1 mm is impossible without knowing the
  current spacing, and a wrong guess rescales the anatomy. Missing spacing is
  an error with an explanation, not a default.
* Accept a 2-D image. A single slice is not a volume; the model's first pooling
  layer would fail on it, and the failure would be an opaque shape error rather
  than "this needs a volume".
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from cardiovision.config import (
    CCTA_HU_MAX,
    CCTA_HU_MIN,
    CCTA_RESAMPLE_ORDER,
    CCTA_TARGET_SPACING,
    MAX_CCTA_VOXELS,
)


class UnsupportedVolumeError(ValueError):
    """Raised when an upload cannot be read as a 3-D CT volume."""


# ============================================================
# RESULT
# ============================================================


@dataclass
class LoadedVolume:
    """
    A CCTA study, resampled and normalised, ready for the model.

    ``volume`` is float32 in [-1, 1] in nibabel's native array order — the axes
    are whatever the source file's affine says they are, not a canonical
    RAS/LPS ordering. The training pipeline did not reorient either, so this
    matches. ``affine`` is carried alongside so a mask can be written back out
    into the same physical space.
    """

    volume: np.ndarray
    affine: Optional[np.ndarray]
    source_format: str
    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    spacing_mm: tuple[float, float, float]
    resampled: bool
    hu_min_observed: float
    hu_max_observed: float
    slices: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    @property
    def voxels(self) -> int:
        return int(self.volume.size)

    def summary(self) -> dict[str, Any]:
        """
        What the API reports about the input.

        No filesystem paths, no series UIDs, no patient identifiers — geometry
        only. Everything here is derived from the pixel data and the spacing
        headers the resampling step needed anyway.
        """
        return {
            "format": self.source_format,
            "original_shape": list(self.original_shape),
            "original_spacing_mm": [
                round(float(v), 4) for v in self.original_spacing_mm
            ],
            "analysed_shape": list(self.volume.shape),
            "analysed_spacing_mm": [round(float(v), 4) for v in self.spacing_mm],
            "resampled": self.resampled,
            "voxels": self.voxels,
            "source_slices": self.slices,
            "hu_range_observed": [
                round(float(self.hu_min_observed), 1),
                round(float(self.hu_max_observed), 1),
            ],
            "hu_window": [CCTA_HU_MIN, CCTA_HU_MAX],
            "notes": list(self.notes),
        }


# ============================================================
# FORMAT DETECTION
# ============================================================


def _suffix_of(filename: str) -> str:
    lowered = (filename or "").lower()

    if lowered.endswith(".nii.gz"):
        return ".nii.gz"

    _, _, tail = lowered.rpartition(".")
    return f".{tail}" if tail and tail != lowered else ""


def detect_volume_format(filename: str, data: bytes) -> str:
    """
    Decide how to read the bytes, preferring content over filename.

    A file called ``study.nii.gz`` that is really a zip of DICOMs is a common
    result of a well-meaning export step, so magic bytes win.
    """
    if data[:4] == b"PK\x03\x04":
        return "zip"

    # NIfTI-1: the first 4 bytes are the header size, 348, little- or
    # big-endian. Gzipped NIfTI starts with the gzip magic instead.
    if data[:2] == b"\x1f\x8b":
        return "nifti"

    if len(data) >= 4 and data[:4] in (
        (348).to_bytes(4, "little"),
        (348).to_bytes(4, "big"),
    ):
        return "nifti"

    # DICOM's "DICM" preamble sits at byte 128.
    if len(data) > 132 and data[128:132] == b"DICM":
        return "dicom"

    suffix = _suffix_of(filename)

    if suffix in (".nii", ".nii.gz", ".gz"):
        return "nifti"
    if suffix == ".zip":
        return "zip"
    if suffix in (".dcm", ".dicom"):
        return "dicom"

    raise UnsupportedVolumeError(
        "Could not tell what kind of file this is. CCTA analysis accepts a "
        "NIfTI volume (.nii or .nii.gz) or a .zip containing one DICOM series."
    )


# ============================================================
# GEOMETRY
# ============================================================


def resampled_shape(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    target: tuple[float, float, float] = CCTA_TARGET_SPACING,
) -> tuple[int, int, int]:
    """
    Shape after resampling to ``target`` spacing.

    ``round`` rather than ``ceil``, matching the notebook. The difference is one
    voxel at the edge, but a mismatch here against the training pipeline would
    put every subsequent patch grid half a voxel out of step.
    """
    return tuple(
        max(1, int(round(shape[i] * spacing[i] / target[i]))) for i in range(3)
    )


def _target_affine(
    affine: np.ndarray,
    target: tuple[float, float, float],
) -> np.ndarray:
    """
    Rebuild the affine at the target spacing, preserving orientation.

    The direction cosines are the affine's columns normalised to unit length;
    scaling those by the target spacing keeps the volume pointing the same way
    in physical space while changing only how finely it is sampled. Rewriting
    the diagonal instead — the obvious shortcut — silently reorients any volume
    that was not stored axis-aligned.
    """
    out = np.array(affine, dtype=np.float64, copy=True)

    for axis in range(3):
        column = out[:3, axis]
        norm = float(np.linalg.norm(column))
        if norm <= 0:
            # Degenerate column: fall back to a unit vector along this axis so
            # the affine stays invertible instead of producing a singular
            # transform that nibabel would reject with an opaque error.
            direction = np.zeros(3, dtype=np.float64)
            direction[axis] = 1.0
        else:
            direction = column / norm
        out[:3, axis] = direction * target[axis]

    return out


def _check_size(shape: tuple[int, ...], stage: str) -> None:
    voxels = 1
    for dim in shape:
        voxels *= int(dim)

    if voxels > MAX_CCTA_VOXELS:
        raise UnsupportedVolumeError(
            f"This volume is too large to analyse: {stage} shape "
            f"{'x'.join(str(d) for d in shape)} is {voxels:,} voxels, above "
            f"the {MAX_CCTA_VOXELS:,} limit. The model was trained on cardiac "
            "field-of-view studies; a whole-body scan should be cropped to the "
            "heart before analysis."
        )


# ============================================================
# NIfTI
# ============================================================


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as error:      # pragma: no cover - environment guard
        raise UnsupportedVolumeError(
            "nibabel is not installed, so NIfTI volumes cannot be read. "
            "Install it with: pip install nibabel"
        ) from error
    return nib


def _load_nifti_volume(data: bytes, filename: str) -> LoadedVolume:
    nib = _require_nibabel()

    notes: list[str] = []
    suffix = ".nii.gz" if _suffix_of(filename) in (".nii.gz", ".gz") else ".nii"

    # nibabel wants a path for gzipped files. Written to a temporary file and
    # deleted in the finally block; the array is copied out first so nothing
    # depends on the file still existing.
    handle, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(data)

        try:
            image = nib.load(path)
        except Exception as error:
            raise UnsupportedVolumeError(
                f"This file could not be read as NIfTI: {error}"
            ) from error

        shape = tuple(int(d) for d in image.shape)

        if len(shape) < 3:
            raise UnsupportedVolumeError(
                f"CCTA analysis needs a 3-D volume; this file is {len(shape)}-D "
                f"with shape {shape}. A single slice or a 2-D image cannot be "
                "segmented by this model."
            )

        if len(shape) > 3:
            trailing = shape[3:]
            if any(int(d) != 1 for d in trailing):
                raise UnsupportedVolumeError(
                    f"This file has shape {shape}. CCTA analysis expects one "
                    "volume, not a time series or a multi-channel stack. Split "
                    "it and submit a single phase."
                )
            notes.append(
                f"Dropped trailing singleton axes {trailing} to get a 3-D "
                "volume."
            )
            image = image.slicer[:, :, :, 0]
            shape = tuple(int(d) for d in image.shape[:3])

        zooms = tuple(float(z) for z in image.header.get_zooms()[:3])

        if not all(np.isfinite(zooms)) or min(zooms) <= 0:
            raise UnsupportedVolumeError(
                f"This volume reports voxel spacing {zooms}, which cannot be "
                "used. Resampling to 1 mm requires valid spacing, and guessing "
                "it would rescale the anatomy. Re-export the volume with a "
                "correct header."
            )

        _check_size(shape, "source")
        target_shape = resampled_shape(shape, zooms)
        _check_size(target_shape, "resampled")

        affine = np.array(image.affine, dtype=np.float64)
        needs_resample = target_shape != shape

        if needs_resample:
            from nibabel.processing import resample_from_to

            reference = (target_shape, _target_affine(affine, CCTA_TARGET_SPACING))

            try:
                resampled = resample_from_to(
                    image, reference, order=CCTA_RESAMPLE_ORDER
                )
            except Exception as error:
                raise UnsupportedVolumeError(
                    f"Resampling this volume to 1 mm isotropic failed: {error}"
                ) from error

            array = np.asarray(resampled.dataobj, dtype=np.float32)
            out_affine = np.array(resampled.affine, dtype=np.float64)
            notes.append(
                f"Resampled from {zooms[0]:.3g}/{zooms[1]:.3g}/{zooms[2]:.3g} mm "
                f"to 1/1/1 mm, {'x'.join(str(d) for d in shape)} -> "
                f"{'x'.join(str(d) for d in target_shape)}, matching the "
                "training pipeline."
            )
        else:
            array = np.asarray(image.dataobj, dtype=np.float32)
            out_affine = affine
            notes.append("Already 1 mm isotropic; no resampling needed.")

        hu_min = float(np.nanmin(array)) if array.size else 0.0
        hu_max = float(np.nanmax(array)) if array.size else 0.0

        if hu_min >= -10.0:
            notes.append(
                f"The lowest value in this volume is {hu_min:.0f}. Real CT air "
                "is near -1000 HU, so these voxels may already be rescaled or "
                "windowed rather than raw Hounsfield units. The model expects "
                "raw HU, and a shifted intensity scale will degrade the mask."
            )

        return LoadedVolume(
            volume=_window_and_scale(array),
            affine=out_affine,
            source_format="nifti",
            original_shape=shape,
            original_spacing_mm=zooms,
            spacing_mm=tuple(float(v) for v in CCTA_TARGET_SPACING),
            resampled=needs_resample,
            hu_min_observed=hu_min,
            hu_max_observed=hu_max,
            slices=int(shape[2]),
            notes=notes,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:      # pragma: no cover - best effort cleanup
            pass


# ============================================================
# DICOM SERIES
# ============================================================


def _require_pydicom():
    try:
        import pydicom
    except ImportError as error:      # pragma: no cover - environment guard
        raise UnsupportedVolumeError(
            "pydicom is not installed, so DICOM series cannot be read. "
            "Install it with: pip install pydicom"
        ) from error
    return pydicom


_DICOM_SKIP_PREFIXES = ("__MACOSX/", "._")


def _zip_members(data: bytes) -> list[tuple[str, bytes]]:
    """
    Every plausible DICOM file in a zip, read into memory.

    Only regular files are read and only by name from the archive's own index —
    nothing is written to disk and no path from the archive is ever joined onto
    a filesystem path, so a crafted entry like ``../../etc/passwd`` cannot
    escape anywhere. There is nowhere for it to go.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise UnsupportedVolumeError(
            f"This .zip could not be opened: {error}"
        ) from error

    members: list[tuple[str, bytes]] = []

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            name = info.filename
            base = name.rsplit("/", 1)[-1]

            if name.startswith(_DICOM_SKIP_PREFIXES) or base.startswith("._"):
                continue
            if base in ("DICOMDIR", "DIRFILE") or base.startswith("."):
                continue
            # A whole series of empty entries is not worth an error; skip and
            # let the "no readable slices" message cover it.
            if info.file_size == 0:
                continue

            try:
                members.append((base, archive.read(info)))
            except (RuntimeError, zipfile.BadZipFile, OSError):
                # Encrypted or corrupt single entry: skip it and report the
                # shortfall in the slice count rather than failing the upload.
                continue

    if not members:
        raise UnsupportedVolumeError(
            "This .zip contains no readable files. It should hold the DICOM "
            "slices of one CT series."
        )

    return members


def _load_dicom_series(data: bytes, filename: str) -> LoadedVolume:
    pydicom = _require_pydicom()

    notes: list[str] = []

    if detect_volume_format(filename, data) == "zip":
        members = _zip_members(data)
    else:
        members = [(filename or "slice.dcm", data)]

    parsed: list[Any] = []
    skipped = 0

    for _name, payload in members:
        try:
            dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
        except Exception:
            skipped += 1
            continue

        # A slice with no pixel data is a structured report, a presentation
        # state or a directory record, not an image.
        if "PixelData" not in dataset:
            skipped += 1
            continue

        parsed.append(dataset)

    if not parsed:
        raise UnsupportedVolumeError(
            "No DICOM image slices were found. CCTA analysis needs the axial "
            "slices of one CT series; a DICOMDIR or a structured report alone "
            "is not enough."
        )

    if skipped:
        notes.append(
            f"Ignored {skipped} file(s) in the archive with no image data."
        )

    if len(parsed) < 16:
        raise UnsupportedVolumeError(
            f"Only {len(parsed)} image slice(s) were found. The model needs at "
            "least a 96-voxel-deep volume after resampling, so a partial "
            "series cannot be analysed. Submit the full series."
        )

    # One series only. Mixing series stacks unrelated geometry into one array.
    series = {str(getattr(d, "SeriesInstanceUID", "")) for d in parsed}
    if len(series) > 1:
        largest = max(
            series,
            key=lambda uid: sum(
                1 for d in parsed if str(getattr(d, "SeriesInstanceUID", "")) == uid
            ),
        )
        kept = [
            d for d in parsed if str(getattr(d, "SeriesInstanceUID", "")) == largest
        ]
        notes.append(
            f"The archive held {len(series)} series; analysed the largest "
            f"({len(kept)} of {len(parsed)} slices). Submit one series to "
            "choose explicitly."
        )
        parsed = kept

    parsed, ordering_note = _sort_slices(parsed)
    if ordering_note:
        notes.append(ordering_note)

    rows = int(getattr(parsed[0], "Rows", 0))
    cols = int(getattr(parsed[0], "Columns", 0))

    if rows <= 0 or cols <= 0:
        raise UnsupportedVolumeError(
            "The first slice reports no image dimensions, so this series "
            "cannot be assembled into a volume."
        )

    inconsistent = [
        d for d in parsed
        if int(getattr(d, "Rows", 0)) != rows
        or int(getattr(d, "Columns", 0)) != cols
    ]
    if inconsistent:
        raise UnsupportedVolumeError(
            f"Slice dimensions vary within this series ({len(inconsistent)} of "
            f"{len(parsed)} differ from {rows}x{cols}). A volume cannot be "
            "assembled from slices of different sizes."
        )

    in_plane = _dicom_in_plane_spacing(parsed[0])
    slice_mm, slice_note = _dicom_slice_spacing(parsed)
    if slice_note:
        notes.append(slice_note)

    _check_size((rows, cols, len(parsed)), "source")

    # Assembled as (rows, cols, slices) so the axis order matches what nibabel
    # hands back for a NIfTI volume, keeping one downstream convention.
    stack = np.empty((rows, cols, len(parsed)), dtype=np.float32)

    for index, dataset in enumerate(parsed):
        try:
            frame = dataset.pixel_array
        except Exception as error:
            raise UnsupportedVolumeError(
                f"Slice {index + 1} could not be decoded: {error}. Compressed "
                "transfer syntaxes need pylibjpeg or gdcm installed."
            ) from error

        if frame.ndim != 2:
            raise UnsupportedVolumeError(
                f"Slice {index + 1} has shape {frame.shape}. Multi-frame "
                "objects inside a series are not supported; submit single-frame "
                "axial slices."
            )

        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)

        stack[:, :, index] = frame.astype(np.float32) * slope + intercept

    if not any(
        "RescaleIntercept" in d for d in parsed[:1]
    ):
        notes.append(
            "No RescaleIntercept was present, so stored values were used as "
            "Hounsfield units directly. If this series was not stored in HU, "
            "the intensity window will be wrong."
        )

    spacing = (in_plane[0], in_plane[1], slice_mm)
    shape = (rows, cols, len(parsed))

    target_shape = resampled_shape(shape, spacing)
    _check_size(target_shape, "resampled")

    hu_min = float(np.nanmin(stack)) if stack.size else 0.0
    hu_max = float(np.nanmax(stack)) if stack.size else 0.0

    needs_resample = target_shape != shape
    if needs_resample:
        stack = _resample_array(stack, shape, target_shape)
        notes.append(
            f"Resampled from {spacing[0]:.3g}/{spacing[1]:.3g}/{spacing[2]:.3g} "
            f"mm to 1/1/1 mm, {'x'.join(str(d) for d in shape)} -> "
            f"{'x'.join(str(d) for d in target_shape)}."
        )
    else:
        notes.append("Already 1 mm isotropic; no resampling needed.")

    notes.append(
        "Assembled from a DICOM series. The training data was NIfTI, and the "
        "axis order here follows slice order rather than a reconstructed "
        "patient-space affine, so submit NIfTI when you have it."
    )

    return LoadedVolume(
        volume=_window_and_scale(stack),
        affine=None,
        source_format="dicom-series",
        original_shape=shape,
        original_spacing_mm=spacing,
        spacing_mm=tuple(float(v) for v in CCTA_TARGET_SPACING),
        resampled=needs_resample,
        hu_min_observed=hu_min,
        hu_max_observed=hu_max,
        slices=len(parsed),
        notes=notes,
    )


def _sort_slices(datasets: list[Any]) -> tuple[list[Any], Optional[str]]:
    """
    Order slices along the acquisition axis.

    ImagePositionPatient projected onto the slice normal is the only ordering
    that is correct for an obliquely-acquired series. InstanceNumber is the
    fallback, and it being used at all is worth reporting: it is usually right
    and occasionally reversed.
    """
    positions = []
    for dataset in datasets:
        ipp = getattr(dataset, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            positions = []
            break
        try:
            positions.append([float(v) for v in ipp[:3]])
        except (TypeError, ValueError):
            positions = []
            break

    if positions:
        normal = _slice_normal(datasets[0])
        keyed = sorted(
            zip(datasets, positions),
            key=lambda pair: float(np.dot(normal, pair[1])),
        )
        return [item[0] for item in keyed], None

    numbers = [
        int(getattr(dataset, "InstanceNumber", 0) or 0) for dataset in datasets
    ]
    if len(set(numbers)) == len(numbers) and any(numbers):
        ordered = [d for _, d in sorted(zip(numbers, datasets), key=lambda p: p[0])]
        return ordered, (
            "Slices carried no ImagePositionPatient, so they were ordered by "
            "InstanceNumber. If the series was exported with reversed instance "
            "numbers the volume is flipped head-to-foot; the mask would still "
            "be computed, but the slice indices in the figures would be "
            "mirrored."
        )

    return datasets, (
        "Slices carried neither ImagePositionPatient nor unique instance "
        "numbers, so archive order was used. Slice ordering is unverified and "
        "the depth axis may be scrambled."
    )


def _slice_normal(dataset: Any) -> np.ndarray:
    orientation = getattr(dataset, "ImageOrientationPatient", None)

    if orientation is not None and len(orientation) >= 6:
        try:
            row = np.array([float(v) for v in orientation[:3]], dtype=np.float64)
            col = np.array([float(v) for v in orientation[3:6]], dtype=np.float64)
            normal = np.cross(row, col)
            norm = float(np.linalg.norm(normal))
            if norm > 0:
                return normal / norm
        except (TypeError, ValueError):
            pass

    # No orientation: assume axial, which is what a cardiac CT series is.
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _dicom_in_plane_spacing(dataset: Any) -> tuple[float, float]:
    spacing = getattr(dataset, "PixelSpacing", None)

    if spacing is not None and len(spacing) >= 2:
        try:
            row_mm = float(spacing[0])
            col_mm = float(spacing[1])
            if row_mm > 0 and col_mm > 0:
                return (row_mm, col_mm)
        except (TypeError, ValueError):
            pass

    raise UnsupportedVolumeError(
        "This series carries no usable PixelSpacing. Resampling to 1 mm needs "
        "it, and assuming a value would rescale the anatomy."
    )


def _dicom_slice_spacing(datasets: list[Any]) -> tuple[float, Optional[str]]:
    """
    Slice thickness, measured from the positions rather than trusted.

    SliceThickness and SpacingBetweenSlices disagree whenever a series was
    reconstructed with overlap, and the position difference is what actually
    describes the sampled grid.
    """
    normal = _slice_normal(datasets[0])
    projections: list[float] = []

    for dataset in datasets:
        ipp = getattr(dataset, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            projections = []
            break
        try:
            point = np.array([float(v) for v in ipp[:3]], dtype=np.float64)
        except (TypeError, ValueError):
            projections = []
            break
        projections.append(float(np.dot(normal, point)))

    if len(projections) >= 2:
        gaps = np.abs(np.diff(np.array(projections, dtype=np.float64)))
        gaps = gaps[gaps > 1e-6]
        if gaps.size:
            measured = float(np.median(gaps))
            spread = float(gaps.max() - gaps.min())
            note = None
            if spread > 0.05 * max(measured, 1e-6):
                note = (
                    f"Slice spacing is uneven: gaps range "
                    f"{gaps.min():.3g}-{gaps.max():.3g} mm. The median "
                    f"{measured:.3g} mm was used for resampling, so anatomy in "
                    "the irregular region is slightly stretched or compressed."
                )
            return measured, note

    for attribute in ("SpacingBetweenSlices", "SliceThickness"):
        value = getattr(datasets[0], attribute, None)
        try:
            thickness = float(value)
        except (TypeError, ValueError):
            continue
        if thickness > 0:
            return thickness, (
                f"Slice positions were unusable, so {attribute} "
                f"({thickness:.3g} mm) was used for the depth spacing."
            )

    raise UnsupportedVolumeError(
        "This series carries no slice positions, no SpacingBetweenSlices and "
        "no SliceThickness, so the depth spacing is unknown. Resampling to "
        "1 mm is not possible without it."
    )


def _resample_array(
    array: np.ndarray,
    shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trilinear resample of a bare array, for the DICOM path with no affine."""
    try:
        from scipy.ndimage import zoom
    except ImportError as error:      # pragma: no cover - environment guard
        raise UnsupportedVolumeError(
            "scipy is not installed, so this series cannot be resampled to "
            "1 mm. Install it with: pip install scipy — or submit a NIfTI "
            "volume that is already 1 mm isotropic."
        ) from error

    factors = tuple(target_shape[i] / shape[i] for i in range(3))

    resized = zoom(array, factors, order=CCTA_RESAMPLE_ORDER, prefilter=False)

    # zoom's output shape is computed by rounding and can land one voxel out.
    # Crop or edge-pad to the shape the patch grid was computed for.
    if tuple(resized.shape) != tuple(target_shape):
        fixed = np.full(target_shape, float(CCTA_HU_MIN), dtype=np.float32)
        cut = tuple(
            slice(0, min(resized.shape[i], target_shape[i])) for i in range(3)
        )
        fixed[cut] = resized[cut]
        resized = fixed

    return np.ascontiguousarray(resized, dtype=np.float32)


# ============================================================
# INTENSITY
# ============================================================


def _window_and_scale(array: np.ndarray) -> np.ndarray:
    """
    Hounsfield units to the [-1, 1] range the weights were trained on.

    In place on a float32 copy, because a 400 MB volume does not need four
    simultaneous copies of itself. The NaN substitution comes first: clipping
    NaN leaves NaN, and one NaN voxel propagates through the convolutions until
    the whole patch is NaN.
    """
    working = np.asarray(array, dtype=np.float32)
    if working.base is None and working.dtype == np.float32:
        working = working.copy()

    np.nan_to_num(
        working,
        copy=False,
        nan=float(CCTA_HU_MIN),
        posinf=float(CCTA_HU_MAX),
        neginf=float(CCTA_HU_MIN),
    )
    np.clip(working, CCTA_HU_MIN, CCTA_HU_MAX, out=working)

    span = float(CCTA_HU_MAX - CCTA_HU_MIN)
    working -= float(CCTA_HU_MIN)
    working /= span
    working *= 2.0
    working -= 1.0

    return working


# ============================================================
# ENTRY POINT
# ============================================================


def load_ccta_volume(data: bytes, filename: str) -> LoadedVolume:
    """
    Read an uploaded CCTA study into a model-ready volume.

    Raises ``UnsupportedVolumeError`` with an explanation for anything that
    cannot be turned into a 1 mm isotropic 3-D array. Every failure here is a
    400-class problem — a wrong file, a 2-D image, missing spacing — and the
    caller turns it into a message rather than a stack trace.
    """
    if not data:
        raise UnsupportedVolumeError("The uploaded file is empty.")

    kind = detect_volume_format(filename, data)

    if kind == "nifti":
        return _load_nifti_volume(data, filename)

    if kind in ("zip", "dicom"):
        return _load_dicom_series(data, filename)

    raise UnsupportedVolumeError(
        f"Unsupported CCTA input: {kind}. Send a NIfTI volume (.nii, .nii.gz) "
        "or a .zip of one DICOM series."
    )
