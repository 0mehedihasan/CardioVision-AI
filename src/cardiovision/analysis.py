"""
The shared analysis core: bytes in, response payload out.

This module holds the orchestration each modality needs — decode, run the model,
render, assemble the payload, archive the upload — with no HTTP in it. It exists
because CardioVision has more than one UI client: the FastAPI routers serve the
React frontend, ``streamlit_app.py`` drives the same pipeline in-process, and a
second copy of this orchestration in either place would be a second set of
medical claims to keep honest.

The payload shape is not incidental. ``services/case_context.py`` and
``fusion/evidence.py`` read these dictionaries by key, so the payload *is* the
contract between analysis and reporting. That is why it is built here once
rather than per client.

Layering: this sits above ``preprocessing``, ``inference``, ``rendering`` and
``services``, and below ``api``. It must not import ``api``, and it must not
raise ``HTTPException`` — :class:`AnalysisError` carries the status an HTTP
client should map it to, which keeps FastAPI out of a module Streamlit imports.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from cardiovision.config import (
    ALLOWED_CCTA_SUFFIXES,
    ALLOWED_ECG_SUFFIXES,
    CCTA_INFERENCE_OVERLAP,
    CCTA_MAX_WINDOWS,
    CCTA_PATCH_SIZE,
    CCTA_PRESENCE_THRESHOLD_VOXELS,
    CCTA_WEAK_NOTES,
    DEVICE,
    ECG_BANDPASS_HIGH_HZ,
    ECG_BANDPASS_LOW_HZ,
    ECG_CLASS_NAMES,
    ECG_CLIP_RANGE,
    ECG_DURATION_SECONDS,
    ECG_INPUT_LENGTH,
    ECG_NORMALIZATION,
    ECG_TARGET_FS,
    ECG_THRESHOLD,
    ECG_WEAK_CLASSES,
    ECHO_CLASS_NAMES,
    ECHO_PRESENCE_THRESHOLD_PX,
    ECHO_TRAINING_ORIENTATION,
)
from cardiovision.inference.ccta import CctaModelUnavailable, ccta_segmenter
from cardiovision.inference.ecg import EcgModelUnavailable, ecg_classifier
from cardiovision.inference.echo import EchoModelUnavailable, echo_segmenter
from cardiovision.preprocessing.ccta_io import (
    UnsupportedVolumeError,
    load_ccta_volume,
)
from cardiovision.preprocessing.ecg_io import UnsupportedEcgError, load_ecg
from cardiovision.preprocessing.image_io import (
    DISPLAY_ORIENTED_FORMATS,
    ORIENTATION_NOTE,
    UnsupportedImageError,
    load_echo_image,
)
from cardiovision.rendering.ccta import render_ccta_images
from cardiovision.rendering.ecg import render_ecg_images
from cardiovision.rendering.echo import encode_mask_payload, render_analysis_images
from cardiovision.services.database import store

__all__ = [
    "AnalysisError",
    "analyze_ccta",
    "analyze_ecg",
    "analyze_echo",
    "ensure_ccta_model",
    "ensure_ecg_model",
    "ensure_echo_model",
]


class AnalysisError(RuntimeError):
    """
    A failure a caller can report, carrying the HTTP status it maps to.

    The status lives on the exception rather than in the router so that the
    distinctions the endpoints already made — 415 for a format that is not
    supported, 413 for a volume too large to hold, 422 for a bad parameter, 503
    for a model that never loaded — survive being called from a client that
    speaks no HTTP.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _ensure_loaded(model: Any, label: str) -> None:
    if not model.is_loaded:
        raise AnalysisError(
            model.load_error or f"The {label} is not loaded.",
            status_code=503,
        )


def ensure_echo_model() -> None:
    """Raise 503 unless the echo segmenter is loaded."""
    _ensure_loaded(echo_segmenter, "echo segmentation model")


def ensure_ccta_model() -> None:
    """Raise 503 unless the CCTA segmenter is loaded."""
    _ensure_loaded(ccta_segmenter, "CCTA segmentation model")


def ensure_ecg_model() -> None:
    """Raise 503 unless the ECG classifier is loaded."""
    _ensure_loaded(ecg_classifier, "ECG classification model")


def _archive(
    *,
    payload: dict[str, Any],
    case_id: Optional[str],
    filename: Optional[str],
    fallback_name: str,
    data: bytes,
    failure_note: str,
) -> None:
    """
    Keep the original bytes with the case, if one is open.

    Failing to archive must not fail the analysis the operator just waited for —
    the result in hand is the valuable part — so the failure becomes a note that
    says re-analysis will need the file again.
    """
    if not (case_id and store.is_ready):
        return

    try:
        payload["source_filename"] = store.store_source_file(
            case_id=case_id,
            filename=filename or fallback_name,
            data=data,
        )
    except Exception as error:                          # pragma: no cover
        print(f"[warning] Could not archive the source file: {error}")
        payload["notes"] = payload["notes"] + [failure_note]


# ============================================================
# ECHO
# ============================================================


def analyze_echo(
    *,
    data: bytes,
    filename: Optional[str],
    frame: Optional[int] = None,
    rotate: int = 0,
    flip: bool = False,
    include_mask: bool = True,
    case_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Segment an echocardiography image into background, LV cavity, myocardium
    and left atrium, and assemble the full response payload.
    """
    ensure_echo_model()

    if rotate not in (0, 90, 180, 270):
        raise AnalysisError(
            f"rotate must be 0, 90, 180 or 270 degrees, got {rotate}.",
            status_code=422,
        )

    if not data:
        raise AnalysisError("The uploaded file is empty.", status_code=400)

    # ---- decode -------------------------------------------------
    try:
        loaded = load_echo_image(
            filename=filename or "",
            data=data,
            frame=frame,
            rotate=rotate,
            flip=flip,
        )
    except UnsupportedImageError as error:
        raise AnalysisError(str(error), status_code=415) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"Could not read the uploaded image: {error}",
            status_code=400,
        ) from error

    # ---- segment ------------------------------------------------
    try:
        analysis = echo_segmenter.analyze(
            pixels=loaded.pixels,
            # The shape of the array actually being segmented, not the shape
            # on disk. A quarter turn swaps rows and columns and the loader
            # swaps the spacing to match, so pairing rotated spacing with
            # the pre-rotation shape would compute the wrong field area.
            original_shape=(loaded.pixels.shape[0], loaded.pixels.shape[1]),
            pixel_spacing_mm=loaded.pixel_spacing_mm,
        )
    except EchoModelUnavailable as error:
        raise AnalysisError(str(error), status_code=503) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"Echo segmentation failed: {error}",
            status_code=500,
        ) from error

    # ---- render -------------------------------------------------
    try:
        images = render_analysis_images(
            normalised_input=analysis.normalised_input,
            prediction=analysis.prediction,
            saliency=analysis.saliency,
        )
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"Could not render the segmentation result: {error}",
            status_code=500,
        ) from error

    model_card = echo_segmenter.describe()

    payload: dict[str, Any] = {
        "success": True,
        "modality": "echo",
        # The device the forward pass actually ran on, which is not always
        # the configured one — saliency can force a CPU fallback.
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": filename,
            "format": loaded.image_format,
            "original_shape": list(loaded.original_shape),
            "analysed_shape": list(
                loaded.oriented_shape or loaded.original_shape
            ),
            "pixel_spacing_mm": (
                list(loaded.pixel_spacing_mm)
                if loaded.pixel_spacing_mm else None
            ),
            "has_spatial_calibration": loaded.has_spatial_calibration,
            "frame_index": loaded.frame_index,
            "frame_count": loaded.frame_count,
        },
        "orientation": {
            "rotation_applied": loaded.rotation_applied,
            "flip_applied": loaded.flip_applied,
            "reoriented": loaded.was_reoriented,
            # True when the format is one whose images are normally stored
            # the way a sonographer sees them, i.e. apex-up, which is a
            # quarter turn away from the training distribution.
            "display_oriented_format": (
                loaded.image_format in DISPLAY_ORIENTED_FORMATS
            ),
            "training_orientation": ECHO_TRAINING_ORIENTATION,
            "note": ORIENTATION_NOTE,
        },
        "structures": [
            structure.to_dict() for structure in analysis.structures
        ],
        "quantification": {
            # Surfaced so "Not identified" can never be read as a clinical
            # absence when it is really a size threshold.
            "presence_threshold_pixels": ECHO_PRESENCE_THRESHOLD_PX,
            "mask_size": model_card["input_size"],
            "presence_rule": (
                f"A structure is reported as present when at least "
                f"{ECHO_PRESENCE_THRESHOLD_PX} pixels of the "
                "segmentation mask carry its label. Below that the "
                "region is treated as noise, not as a finding."
            ),
        },
        "images": images,
        "explainability": {
            "method": "Input-gradient saliency",
            "target_class": ECHO_CLASS_NAMES[1],
            "available": analysis.saliency_available,
            "device": analysis.saliency_device,
            "description": (
                "Absolute gradient of the mean left-ventricular-cavity "
                "probability with respect to the input image. Bright regions "
                "are those the prediction is most sensitive to. This shows "
                "model attribution, not clinical evidence."
            ),
        },
        "inference_ms": round(analysis.inference_ms, 1),
        "notes": loaded.notes + analysis.notes,
    }

    if include_mask:
        payload["mask"] = encode_mask_payload(analysis.prediction)

    _archive(
        payload=payload,
        case_id=case_id,
        filename=filename,
        fallback_name="source",
        data=data,
        failure_note=(
            "The original file could not be archived with this case, so "
            "re-analysing at a different rotation will need it uploaded "
            "again."
        ),
    )

    return payload


# ============================================================
# CCTA
# ============================================================


def analyze_ccta(
    *,
    data: bytes,
    filename: Optional[str],
    max_windows: int = CCTA_MAX_WINDOWS,
    include_gradcam: bool = True,
    include_figures: bool = True,
    case_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Segment the coronary lumen in a CT volume, with 3-D Grad-CAM.

    The window budget is part of the result, not a tuning knob hidden from the
    reader: when it cannot cover the volume the analysis runs on a centred crop
    and ``coverage.complete`` is ``False``, which is a first-class outcome.
    """
    ensure_ccta_model()

    if not data:
        raise AnalysisError("The uploaded file is empty.", status_code=400)

    # ---- decode + resample --------------------------------------
    #
    # Resampling a whole-chest study to 1 mm is itself expensive, so any failure
    # here is reported before the model is touched.
    try:
        loaded = load_ccta_volume(data=data, filename=filename or "")
    except UnsupportedVolumeError as error:
        raise AnalysisError(
            f"{error} Accepted extensions: "
            f"{', '.join(sorted(ALLOWED_CCTA_SUFFIXES))}.",
            status_code=415,
        ) from error
    except MemoryError as error:
        raise AnalysisError(
            "The volume was too large to resample in memory. Crop it to "
            "the cardiac region and upload that instead.",
            status_code=413,
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"Could not read the uploaded volume: {error}",
            status_code=400,
        ) from error

    # ---- segment ------------------------------------------------
    try:
        analysis = ccta_segmenter.analyze(
            volume=loaded.volume,
            spacing_mm=loaded.spacing_mm,
            max_windows=max_windows,
            include_gradcam=include_gradcam,
        )
    except CctaModelUnavailable as error:
        raise AnalysisError(str(error), status_code=503) from error
    except MemoryError as error:
        raise AnalysisError(
            "Ran out of memory during inference. Lower max_windows, or "
            "crop the volume to the cardiac region.",
            status_code=413,
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"CCTA segmentation failed: {error}",
            status_code=500,
        ) from error

    # ---- render -------------------------------------------------
    #
    # Same rule as the other two modalities: the mask and the measurements are
    # what the operator waited minutes for. A renderer failure degrades to JSON
    # with a note, never to an error.
    figures: dict[str, str] = {}
    figure_meta: dict[str, Any] = {}
    figure_notes: list[str] = []

    if include_figures:
        try:
            figures, figure_meta, figure_notes = render_ccta_images(
                volume=loaded.volume,
                mask=analysis.mask,
                probability=analysis.probability,
                analysed=analysis.analysed,
                gradcam=analysis.gradcam,
                gradcam_origin=analysis.gradcam_origin,
            )
        except Exception as error:                      # pragma: no cover
            traceback.print_exc()
            figure_notes = [
                f"The CCTA figures could not be rendered ({error}). The "
                "measurements below are unaffected."
            ]

    model_card = ccta_segmenter.describe()
    complete = analysis.windows_run >= analysis.windows_total

    payload: dict[str, Any] = {
        "success": True,
        "modality": "ccta",
        "analyzed": True,
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": filename,
            **loaded.summary(),
        },
        "threshold": round(analysis.threshold, 4),
        "findings": [finding.to_dict() for finding in analysis.findings],
        "quantification": {
            "presence_threshold_voxels": CCTA_PRESENCE_THRESHOLD_VOXELS,
            "voxel_volume_mm3": 1.0,
            "note": (
                "Volumes are computed on the 1 mm isotropic grid the model ran "
                "on, so one voxel is one cubic millimetre. A mask smaller than "
                "the presence threshold is reported as not present, which is a "
                "size cutoff and not a clinical finding."
            ),
        },
        "coverage": {
            "complete": complete,
            "coverage": round(analysis.coverage, 6),
            "analysed_percent": round(analysis.coverage * 100.0, 2),
            "windows_run": analysis.windows_run,
            "windows_total": analysis.windows_total,
            "patch_size": list(CCTA_PATCH_SIZE),
            "overlap": CCTA_INFERENCE_OVERLAP,
            "note": (
                "Everything outside the analysed region was not examined. In "
                "the mask and the figures it is zero, which means not looked "
                "at — not absent."
                if not complete else
                "The whole volume was covered by the sliding window."
            ),
        },
        "explainability": {
            "available": analysis.gradcam_available,
            "method": "3-D Grad-CAM",
            "target_layer": model_card["explainability"]["target_layer"],
            "scope": model_card["explainability"]["scope"],
            "origin": (
                list(analysis.gradcam_origin) if analysis.gradcam_origin else None
            ),
            "shape": (
                list(analysis.gradcam_shape) if analysis.gradcam_shape else None
            ),
            "device": analysis.gradcam_device,
            "note": (
                "Grad-CAM shows where activation supported the model's own "
                "output. It does not localise disease, and it was computed on "
                "one patch only."
            ),
        },
        "limitations": list(CCTA_WEAK_NOTES),
        "inference_ms": round(analysis.inference_ms, 1),
        "figures": figures,
        "figure_meta": figure_meta,
        "notes": loaded.notes + analysis.notes + figure_notes,
    }

    _archive(
        payload=payload,
        case_id=case_id,
        filename=filename,
        fallback_name="ccta",
        data=data,
        failure_note=(
            "The original volume could not be archived with this case, so "
            "re-analysing it later will need it uploaded again."
        ),
    )

    return payload


# ============================================================
# ECG
# ============================================================


def analyze_ecg(
    *,
    data: bytes,
    filename: Optional[str],
    companions: Optional[dict[str, bytes]] = None,
    sampling_frequency: Optional[float] = None,
    target_class: Optional[str] = None,
    include_figures: bool = True,
    case_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Classify a 12-lead ECG into the five PTB-XL diagnostic superclasses, with
    per-lead gradient attribution.

    ``companions`` is keyed by filename because WFDB headers name their signal
    file explicitly; relying on upload order would break the moment a client
    reordered the files.
    """
    ensure_ecg_model()

    if target_class is not None:
        target_class = target_class.strip().upper() or None

        if target_class is not None and target_class not in ECG_CLASS_NAMES:
            raise AnalysisError(
                f"Unknown target_class {target_class!r}. Expected one of "
                f"{', '.join(ECG_CLASS_NAMES)}.",
                status_code=422,
            )

    if not data:
        raise AnalysisError("The uploaded file is empty.", status_code=400)

    companion_bytes: dict[str, bytes] = dict(companions or {})

    # ---- decode + preprocess ------------------------------------
    try:
        loaded = load_ecg(
            filename=filename or "",
            data=data,
            sampling_frequency=sampling_frequency,
            companion=companion_bytes,
        )
    except UnsupportedEcgError as error:
        raise AnalysisError(
            f"{error} Accepted extensions: "
            f"{', '.join(sorted(ALLOWED_ECG_SUFFIXES))}.",
            status_code=415,
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"Could not read the uploaded ECG: {error}",
            status_code=400,
        ) from error

    # ---- classify -----------------------------------------------
    try:
        analysis = ecg_classifier.analyze(
            signal=loaded.signal,
            target_class=target_class,
        )
    except EcgModelUnavailable as error:
        raise AnalysisError(str(error), status_code=503) from error
    except Exception as error:
        traceback.print_exc()
        raise AnalysisError(
            f"ECG classification failed: {error}",
            status_code=500,
        ) from error

    # ---- render -------------------------------------------------
    #
    # A figure failure must not lose the classification: the probabilities are
    # the result the operator waited for, and the strip is how they are read.
    figures: dict[str, str] = {}
    figure_notes: list[str] = []

    if include_figures:
        try:
            figures = render_ecg_images(
                loaded.display_signal,
                analysis,
                lead_names=loaded.lead_names,
                units=loaded.units,
                sampling_frequency=ECG_TARGET_FS,
                record_name=loaded.record_name,
            )
        except Exception as error:                     # pragma: no cover
            traceback.print_exc()
            figure_notes.append(
                "The waveform figures could not be rendered "
                f"({error}). The classification below is unaffected."
            )

    model_card = ecg_classifier.describe()

    # Weak-class warnings, but only for classes this recording actually called
    # positive. Listing HYP's precision on a recording that came back HYP-
    # negative would be noise, and noise is what gets filtered out by eye.
    weak_warnings = {
        name: ECG_WEAK_CLASSES[name]
        for name in analysis.positive_classes
        if name in ECG_WEAK_CLASSES
    }

    payload: dict[str, Any] = {
        "success": True,
        "modality": "ecg",
        # The device the forward pass actually ran on. Saliency needs
        # gradients, which can force a CPU fallback on MPS.
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": filename,
            "companions": sorted(companion_bytes),
            **loaded.to_dict(),
        },
        "preprocessing": {
            "bandpass_hz": [ECG_BANDPASS_LOW_HZ, ECG_BANDPASS_HIGH_HZ],
            "resampled_to_hz": ECG_TARGET_FS,
            "length_samples": ECG_INPUT_LENGTH,
            "length_seconds": ECG_DURATION_SECONDS,
            "normalization": ECG_NORMALIZATION,
            "clip_range": list(ECG_CLIP_RANGE),
            "note": (
                "Identical to the training chain. The normalisation is "
                "per-lead robust median/IQR, so the model's input is in "
                "dimensionless IQR multiples — which is why the figures plot "
                "the pre-normalisation signal instead."
            ),
        },
        **analysis.to_dict(),
        "weak_class_warnings": weak_warnings,
        "threshold_note": (
            f"A class is called positive at p >= {ECG_THRESHOLD}. Every "
            "probability is returned, so a different operating point can be "
            "applied without re-running the model — but the precision and "
            "recall in the model card were measured at this threshold and do "
            "not carry over to another one."
        ),
        "figures": figures,
        "notes": loaded.notes + analysis.notes + figure_notes,
    }

    _archive(
        payload=payload,
        case_id=case_id,
        filename=filename,
        fallback_name="ecg",
        data=data,
        failure_note=(
            "The original recording could not be archived with this case, "
            "so re-analysing it later will need it uploaded again."
        ),
    )

    return payload
