"""
Echocardiography segmentation.

Model card and analysis for the UNet++ / EfficientNet-B3 segmenter trained on
CAMUS. The inference and rendering modules are imported under aliases because
this router is also called ``echo``.

The response is deliberately verbose about provenance: what was decoded, what
orientation it was analysed in, whether the pixels carry spatial calibration,
and which device the forward pass actually ran on. Every one of those changes
how the numbers below it should be read.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from cardiovision.api.deps import read_upload, require_session
from cardiovision.config import (
    DEVICE,
    ECHO_CLASS_NAMES,
    ECHO_PRESENCE_THRESHOLD_PX,
    ECHO_TRAINING_ORIENTATION,
)
from cardiovision.inference.echo import EchoModelUnavailable, echo_segmenter
from cardiovision.preprocessing.image_io import (
    DISPLAY_ORIENTED_FORMATS,
    ORIENTATION_NOTE,
    UnsupportedImageError,
    load_echo_image,
)
from cardiovision.rendering.echo import (
    encode_mask_payload,
    render_analysis_images,
)
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

router = APIRouter(tags=["echo"])


def _require_model() -> None:
    if not echo_segmenter.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                echo_segmenter.load_error
                or "The echo segmentation model is not loaded."
            ),
        )


@router.get("/api/models/echo")
def echo_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **echo_segmenter.describe()}


@router.post("/api/analyze/echo")
async def analyze_echo(
    file: UploadFile = File(...),
    frame: Optional[int] = Query(
        default=None,
        ge=0,
        description="Frame index for multi-frame DICOM cine loops.",
    ),
    rotate: int = Query(
        default=0,
        description=(
            "Counter-clockwise rotation in degrees (0, 90, 180 or 270) "
            "applied before inference. The model was trained on images "
            "whose sector apex points left; conventional apex-up echo "
            "displays need a quarter turn. Nothing is rotated by default."
        ),
    ),
    flip: bool = Query(
        default=False,
        description="Mirror the image horizontally before inference.",
    ),
    include_mask: bool = Query(
        default=True,
        description="Include the raw class mask for client-side rendering.",
    ),
    case_id: Optional[str] = Query(
        default=None,
        max_length=100,
        description=(
            "Attach the original upload to this case, so it can be "
            "re-analysed at a different rotation later without asking the "
            "operator to find the file again."
        ),
    ),
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Segment an uploaded echocardiography image into background, LV cavity,
    myocardium and left atrium.
    """
    _require_model()

    if rotate not in (0, 90, 180, 270):
        raise HTTPException(
            status_code=422,
            detail=f"rotate must be 0, 90, 180 or 270 degrees, got {rotate}.",
        )

    data = await read_upload(file)

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # ---- decode -------------------------------------------------
    try:
        loaded = load_echo_image(
            filename=file.filename or "",
            data=data,
            frame=frame,
            rotate=rotate,
            flip=flip,
        )
    except UnsupportedImageError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded image: {error}",
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
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Echo segmentation failed: {error}",
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
        raise HTTPException(
            status_code=500,
            detail=f"Could not render the segmentation result: {error}",
        ) from error

    model_card = echo_segmenter.describe()

    response: dict[str, Any] = {
        "success": True,
        "modality": "echo",
        # The device the forward pass actually ran on, which is not always
        # the configured one — saliency can force a CPU fallback.
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": file.filename,
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
        response["mask"] = encode_mask_payload(analysis.prediction)

    # Keep the original bytes with the case, if one is open. Failing to
    # archive the file must not fail the analysis the operator just waited
    # for — the segmentation in hand is the valuable part.
    if case_id and store.is_ready:
        try:
            stored_name = store.store_source_file(
                case_id=case_id,
                filename=file.filename or "source",
                data=data,
            )
            response["source_filename"] = stored_name
        except Exception as error:                     # pragma: no cover
            print(f"[warning] Could not archive the source file: {error}")
            response["notes"] = response["notes"] + [
                "The original file could not be archived with this case, so "
                "re-analysing at a different rotation will need it uploaded "
                "again."
            ]

    return response
