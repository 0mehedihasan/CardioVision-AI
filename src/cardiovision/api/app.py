"""
CardioVision AI — FastAPI backend.

Endpoints
---------
GET  /                        service banner
GET  /api/health              load state + which modalities are real  (public)
POST /api/auth/login          exchange credentials for a session token (public)
POST /api/auth/logout         revoke the current token
GET  /api/auth/session        who am I, and how long is the session good for
GET  /api/models/echo         echo model card (architecture + metrics)
POST /api/analyze/echo        echo segmentation from an uploaded image
POST /api/clinical-question   MedGemma Q&A, optionally with case context
GET  /api/cases               list / search saved cases
POST /api/cases               create or update a case
GET  /api/cases/{id}          load one case
DELETE /api/cases/{id}        delete a case and its stored images
GET  /api/cases/{id}/images/{name}   serve a stored PNG

Everything except the banner, /api/health and the login endpoint requires a
session token. Health stays public so the login screen can report whether the
backend is up and which models loaded before anyone signs in.

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
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import auth
from case_context import build_case_context
from config import (
    AUTH_SESSION_TTL_SECONDS,
    DEVICE,
    ECHO_CLASS_NAMES,
    ECHO_PRESENCE_THRESHOLD_PX,
    ECHO_TRAINING_ORIENTATION,
    MAX_UPLOAD_BYTES,
    MEDGEMMA_NAME,
    MODALITY_STATUS,
)
from database import CaseStoreError, store
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
    # The case database comes up first. Without it the app can still segment
    # images, so a storage failure is reported rather than fatal.
    try:
        store.connect()
        print(f"Case database ready ({store.count()} saved cases).")
    except Exception as error:                             # pragma: no cover
        print(f"[warning] Case database unavailable: {error}")
        traceback.print_exc()

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
    print(f"  Case storage      : {'ready' if store.is_ready else 'unavailable'}")
    print(f"  Sign in as        : {auth.USERNAME}")

    if auth.USING_DEFAULT_CREDENTIALS:
        print(
            "  [notice] Using the default password. This is an access gate, "
            "not security.\n"
            "           Override it with CARDIOVISION_USER and "
            "CARDIOVISION_PASSWORD,\n"
            "           and keep this service bound to localhost."
        )

    yield

    store.close()


app = FastAPI(
    title="CardioVision AI API",
    description="Backend API for CardioVision AI clinical intelligence.",
    version="3.0.0",
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
# AUTH DEPENDENCY
# ============================================================

def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None

    parts = authorization.split(None, 1)

    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()

    # Tolerate a bare token. The frontend always sends the scheme, but this
    # makes the API usable from curl without quoting gymnastics.
    return authorization.strip() or None


def require_session(
    authorization: Optional[str] = Header(default=None),
) -> auth.Session:
    """
    Reject anything without a valid session token.

    Returning 401 with WWW-Authenticate lets the frontend distinguish "your
    session expired, log in again" from a genuine server error, which matters
    because the two need completely different handling in the UI.
    """
    session = auth.sessions.resolve(_bearer_token(authorization))

    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Not signed in, or the session has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


# ============================================================
# SCHEMAS
# ============================================================

class LoginRequest(BaseModel):
    username: str = Field(..., max_length=200)
    password: str = Field(..., max_length=400)


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


class CasePayload(BaseModel):
    """
    A case as the frontend holds it.

    Every field is optional: a case can legitimately be saved with nothing
    but a patient name, before any imaging exists.
    """

    case_id: Optional[str] = Field(default=None, max_length=100)
    patient: Optional[dict[str, Any]] = Field(default=None)
    clinical: Optional[dict[str, Any]] = Field(default=None)
    echo: Optional[dict[str, Any]] = Field(default=None)
    images: Optional[dict[str, str]] = Field(default=None)
    conversation: Optional[list[dict[str, Any]]] = Field(default=None)


# ============================================================
# AUTHENTICATION
# ============================================================

@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict[str, Any]:
    """
    Exchange credentials for a session token.

    Failures are deliberately vague about *which* half was wrong — naming the
    field would let someone confirm a valid username before attacking the
    password.
    """
    locked_for = auth.throttle.check()

    if locked_for is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many failed sign-in attempts. Try again in "
                f"{locked_for} seconds."
            ),
        )

    if not auth.verify_credentials(request.username, request.password):
        remaining = auth.throttle.record_failure()

        detail = "Incorrect username or password."
        if remaining:
            detail += f" {remaining} attempt{'s' if remaining != 1 else ''} left."

        raise HTTPException(status_code=401, detail=detail)

    auth.throttle.reset()
    session = auth.sessions.create(auth.USERNAME)

    return {
        "success": True,
        "token": session.token,
        "username": session.username,
        "expires_in": session.seconds_remaining,
        # Surfaced so the UI can show a standing reminder rather than letting
        # a default password quietly become permanent.
        "using_default_credentials": auth.USING_DEFAULT_CREDENTIALS,
    }


@app.post("/api/auth/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """
    Revoke the current token.

    Not gated behind require_session: logging out with an already-expired
    token should quietly succeed, not fail with a 401.
    """
    revoked = auth.sessions.revoke(_bearer_token(authorization))

    return {"success": True, "revoked": revoked}


@app.get("/api/auth/session")
def current_session(
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
    """Validate a stored token on page load, without a fresh login."""
    return {
        "success": True,
        "username": session.username,
        "expires_in": session.seconds_remaining,
        "session_ttl_seconds": AUTH_SESSION_TTL_SECONDS,
        "using_default_credentials": auth.USING_DEFAULT_CREDENTIALS,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """
    Public. The login screen needs to report backend and model state before
    anyone has signed in; nothing here is patient data.
    """
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
        "storage": {
            "ready": store.is_ready,
            "error": store.connect_error,
            # Deliberately only a count. A case list is patient data and
            # requires a session.
            "saved_cases": store.count() if store.is_ready else 0,
        },
        "auth": {
            "required": True,
            "username": auth.USERNAME,
            "using_default_credentials": auth.USING_DEFAULT_CREDENTIALS,
        },
        # Kept for compatibility with the original frontend.
        "medgemma_loaded": medgemma.is_loaded,
        "model": MEDGEMMA_NAME,
        "modalities": modalities,
    }


@app.get("/api/models/echo")
def echo_model_card(
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
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
    case_id: Optional[str] = Query(
        default=None,
        max_length=100,
        description=(
            "Attach the original upload to this case, so it can be "
            "re-analysed at a different rotation later without asking the "
            "operator to find the file again."
        ),
    ),
    session: auth.Session = Depends(require_session),
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


# ============================================================
# CASES
# ============================================================

@app.get("/api/cases")
def list_cases(
    search: str = Query(default="", max_length=200),
    limit: int = Query(default=200, ge=1, le=1000),
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
    """Saved case summaries, most recently updated first."""
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The case database is not available.",
        )

    try:
        cases = store.list(search=search, limit=limit)
    except CaseStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"success": True, "cases": cases, "total": store.count()}


@app.post("/api/cases")
def save_case(
    payload: CasePayload,
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Create or update a case.

    An existing case_id updates in place; omitting it mints a new one. The
    caller gets the stored record back, so the frontend can adopt the
    generated ID and timestamps rather than guessing them.
    """
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The case database is not available, so nothing was saved.",
        )

    try:
        stored = store.save(payload.model_dump(exclude_none=True))
    except CaseStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the case: {error}",
        ) from error

    return {"success": True, "case": stored}


@app.get("/api/cases/{case_id}")
def get_case(
    case_id: str,
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The case database is not available.",
        )

    case = store.get(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"No case {case_id}.")

    return {"success": True, "case": case}


@app.delete("/api/cases/{case_id}")
def delete_case(
    case_id: str,
    session: auth.Session = Depends(require_session),
) -> dict[str, Any]:
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The case database is not available.",
        )

    if not store.delete(case_id):
        raise HTTPException(status_code=404, detail=f"No case {case_id}.")

    return {"success": True, "deleted": case_id}


@app.get("/api/cases/{case_id}/images/{name}")
def get_case_image(
    case_id: str,
    name: str,
    session: auth.Session = Depends(require_session),
) -> Response:
    """
    Serve one stored PNG.

    The store validates `name` against a fixed set of render keys, so a
    crafted path cannot reach outside the case directory.
    """
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The case database is not available.",
        )

    payload = store.read_image(case_id, name)

    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stored '{name}' image for case {case_id}.",
        )

    return Response(
        content=payload,
        media_type="image/png",
        # Stored renders are immutable: a new analysis overwrites the case or
        # creates a new one, so the browser can cache these hard.
        headers={"Cache-Control": "private, max-age=86400"},
    )


# ============================================================
# CLINICAL QUESTION
# ============================================================

@app.post("/api/clinical-question", response_model=ClinicalQuestionResponse)
def ask_clinical_question(
    request: ClinicalQuestionRequest,
    session: auth.Session = Depends(require_session),
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
        "auth_required": True,
        "endpoints": [
            "GET  /api/health                    (public)",
            "POST /api/auth/login                (public)",
            "POST /api/auth/logout",
            "GET  /api/auth/session",
            "GET  /api/models/echo",
            "POST /api/analyze/echo",
            "POST /api/clinical-question",
            "GET  /api/cases",
            "POST /api/cases",
            "GET  /api/cases/{case_id}",
            "DELETE /api/cases/{case_id}",
            "GET  /api/cases/{case_id}/images/{name}",
        ],
    }
