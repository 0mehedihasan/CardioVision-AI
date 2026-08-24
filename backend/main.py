"""
CardioVision AI — FastAPI backend.

Endpoints
---------
GET  /                        service banner
GET  /api/health              load state + which modalities are real
GET  /api/models/echo         echo model card (architecture + metrics)
POST /api/analyze/echo        echo segmentation from an uploaded image
POST /api/clinical-question   MedGemma Q&A, optionally with case context

The two models load independently: a MedGemma failure no longer prevents
echo segmentation from working, and vice versa. Load state is reported
through /api/health so the UI can only advertise what actually works.
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from case_context import build_case_context
from config import (
    DEVICE,
    ECHO_CLASS_NAMES,
    ECHO_PRESENCE_THRESHOLD_PX,
    ECHO_TRAINING_ORIENTATION,
    MAX_UPLOAD_BYTES,
    MEDGEMMA_NAME,
    MODALITY_STATUS,
)
from echo_model import EchoModelUnavailable, echo_segmenter
from image_io import (
    DISPLAY_ORIENTED_FORMATS,
    ORIENTATION_NOTE,
    UnsupportedImageError,
    load_echo_image,
)
from medgemma import MedGemmaUnavailable, medgemma
from rendering import encode_mask_payload, render_analysis_images


# ============================================================
# LIFESPAN
# ============================================================

def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load both models at startup, but never let one failure take down the
    service. Set CARDIOVISION_SKIP_MEDGEMMA=1 to skip the 8.6 GB language
    model while iterating on the imaging pipeline.
    """
    skip_medgemma = _truthy(os.environ.get("CARDIOVISION_SKIP_MEDGEMMA"))
    skip_echo = _truthy(os.environ.get("CARDIOVISION_SKIP_ECHO"))

    if skip_echo:
        print("Skipping echo model (CARDIOVISION_SKIP_ECHO is set).")
    else:
        try:
            echo_segmenter.load()
        except EchoModelUnavailable as error:
            print(f"[warning] Echo model unavailable: {error}")
        except Exception as error:                     # pragma: no cover
            print(f"[warning] Unexpected echo model failure: {error}")
            traceback.print_exc()

    if skip_medgemma:
        print("Skipping MedGemma (CARDIOVISION_SKIP_MEDGEMMA is set).")
    else:
        try:
            medgemma.load()
        except MedGemmaUnavailable as error:
            print(f"[warning] MedGemma unavailable: {error}")
        except Exception as error:                     # pragma: no cover
            print(f"[warning] Unexpected MedGemma failure: {error}")
            traceback.print_exc()

    print("CardioVision AI backend ready.")
    print(f"  Echo segmentation : {'ready' if echo_segmenter.is_loaded else 'unavailable'}")
    print(f"  MedGemma          : {'ready' if medgemma.is_loaded else 'unavailable'}")

    yield


app = FastAPI(
    title="CardioVision AI API",
    description="Backend API for CardioVision AI clinical intelligence.",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS
# ============================================================

class ClinicalQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)

    # Structured case state. The backend renders it into prompt text so
    # prompt construction stays server-side.
    case: Optional[dict[str, Any]] = Field(default=None)

    # Pre-rendered context. Retained for backwards compatibility and for
    # debugging; when present it wins over `case`.
    context: Optional[str] = Field(default=None, max_length=12000)


class ClinicalQuestionResponse(BaseModel):
    success: bool
    answer: str
    model: str
    device: str
    context_used: bool
    context_preview: Optional[str] = None


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health_check() -> dict[str, Any]:
    modalities: dict[str, Any] = {}

    for key, status in MODALITY_STATUS.items():
        entry = dict(status)

        if key == "echo":
            entry["available"] = echo_segmenter.is_loaded
            if not echo_segmenter.is_loaded:
                entry["note"] = (
                    echo_segmenter.load_error
                    or "Echo model is not loaded."
                )

        modalities[key] = entry

    return {
        "status": "healthy",
        "service": "CardioVision AI",
        "device": DEVICE,
        "models": {
            "medgemma": {
                "name": MEDGEMMA_NAME,
                "loaded": medgemma.is_loaded,
                "error": medgemma.load_error,
            },
            "echo": {
                "name": "CardioVision Echo (UNet++ / EfficientNet-B3)",
                "loaded": echo_segmenter.is_loaded,
                "error": echo_segmenter.load_error,
            },
        },
        # Kept for compatibility with the original frontend.
        "medgemma_loaded": medgemma.is_loaded,
        "model": MEDGEMMA_NAME,
        "modalities": modalities,
    }


@app.get("/api/models/echo")
def echo_model_card() -> dict[str, Any]:
    if not echo_segmenter.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=echo_segmenter.load_error or "Echo model is not loaded.",
        )

    return {"success": True, **echo_segmenter.describe()}


# ============================================================
# ECHO ANALYSIS
# ============================================================

async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload while enforcing the size cap."""
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break

        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "File is too large. The limit is "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                ),
            )

        chunks.append(chunk)

    return b"".join(chunks)


@app.post("/api/analyze/echo")
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
) -> dict[str, Any]:
    """
    Segment an uploaded echocardiography image into background, LV cavity,
    myocardium and left atrium.
    """
    if not echo_segmenter.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                echo_segmenter.load_error
                or "The echo segmentation model is not loaded."
            ),
        )

    if rotate not in (0, 90, 180, 270):
        raise HTTPException(
            status_code=422,
            detail=f"rotate must be 0, 90, 180 or 270 degrees, got {rotate}.",
        )

    data = await _read_upload(file)

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

    return response


# ============================================================
# CLINICAL QUESTION
# ============================================================

@app.post("/api/clinical-question", response_model=ClinicalQuestionResponse)
def ask_clinical_question(
    request: ClinicalQuestionRequest,
) -> ClinicalQuestionResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Clinical question cannot be empty.",
        )

    if not medgemma.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                medgemma.load_error
                or "The clinical language model is not loaded."
            ),
        )

    # An explicit context string wins; otherwise render the structured case.
    context = request.context
    if not (context and context.strip()):
        context = build_case_context(request.case)

    try:
        answer = medgemma.generate(question=question, context=context)
    except MedGemmaUnavailable as error:
        print("=" * 60)
        print("MedGemma inference error")
        print(error)
        print("=" * 60)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a clinical response.",
        ) from error

    return ClinicalQuestionResponse(
        success=True,
        answer=answer,
        model=MEDGEMMA_NAME,
        device=DEVICE,
        context_used=bool(context),
        # Surfaced so the UI can show exactly what the model was told.
        context_preview=context,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "CardioVision AI",
        "status": "running",
        "device": DEVICE,
        "models": {
            "medgemma": medgemma.is_loaded,
            "echo_segmentation": echo_segmenter.is_loaded,
        },
        "model": MEDGEMMA_NAME,
        "endpoints": [
            "GET  /api/health",
            "GET  /api/models/echo",
            "POST /api/analyze/echo",
            "POST /api/clinical-question",
        ],
    }
