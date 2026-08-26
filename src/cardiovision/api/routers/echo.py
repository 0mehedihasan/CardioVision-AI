"""
Echocardiography segmentation.

Model card and analysis for the UNet++ / EfficientNet-B3 segmenter trained on
CAMUS.

The work itself lives in :func:`cardiovision.analysis.analyze_echo`, which the
Streamlit client calls directly. This router owns only what is genuinely HTTP:
the query contract, reading the upload under a size cap, and turning an
:class:`~cardiovision.analysis.AnalysisError` into a status code.

The response is deliberately verbose about provenance: what was decoded, what
orientation it was analysed in, whether the pixels carry spatial calibration,
and which device the forward pass actually ran on. Every one of those changes
how the numbers below it should be read.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile

from cardiovision.analysis import AnalysisError, analyze_echo, ensure_echo_model
from cardiovision.api.deps import as_http_error, read_upload, require_session
from cardiovision.inference.echo import echo_segmenter
from cardiovision.services import auth as auth_service

router = APIRouter(tags=["echo"])


def _require_model() -> None:
    try:
        ensure_echo_model()
    except AnalysisError as error:
        raise as_http_error(error) from error


@router.get("/api/models/echo")
def echo_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **echo_segmenter.describe()}


@router.post("/api/analyze/echo")
async def analyze_echo_endpoint(
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
    # Checked before the body is read, so an unloaded model is refused without
    # first pulling a megabytes-long upload off the wire.
    _require_model()

    data = await read_upload(file)

    try:
        return analyze_echo(
            data=data,
            filename=file.filename,
            frame=frame,
            rotate=rotate,
            flip=flip,
            include_mask=include_mask,
            case_id=case_id,
        )
    except AnalysisError as error:
        raise as_http_error(error) from error
