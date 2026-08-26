"""
Coronary CT angiography lumen segmentation.

Model card and analysis for the Small3DUNet trained on MedHK23/CCA. This is the
weakest of the three models in the project and the endpoint is built to say so
rather than to look confident: the model card carries the three-case test split
on the same level as the Dice score, and every response repeats that the output
is a lumen mask with no stenosis grade, no calcium score and no vessel identity.

The pipeline lives in :func:`cardiovision.analysis.analyze_ccta`, shared with the
Streamlit client. Three things there are unlike the echo and ECG paths.

**The upload is large and the compute is bounded.** A 0.5 mm study is roughly
830x830x580 voxels; resampled to 1 mm it still needs some 490 sliding windows,
which is minutes of CPU. So the request carries a window budget, and when the
budget cannot cover the volume the analysis covers a centred crop and reports
exactly what it skipped. ``coverage.complete: false`` is a first-class result,
not an error.

**Absence of mask is not absence of disease, and unanalysed is not absent.**
The response distinguishes the two: ``analysed_percent`` says how much was
looked at, and the rendered slices are annotated where they extend beyond it.

**The shipped example figures are training data.** ``models/ccta/case_*_xai.png``
came from the notebook's own test cases and are never returned here. Everything
in ``figures`` was rendered from the uploaded volume.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile

from cardiovision.analysis import AnalysisError, analyze_ccta, ensure_ccta_model
from cardiovision.api.deps import as_http_error, read_upload, require_session
from cardiovision.config import CCTA_MAX_WINDOWS, MAX_CCTA_UPLOAD_BYTES
from cardiovision.inference.ccta import ccta_segmenter
from cardiovision.services import auth as auth_service

router = APIRouter(tags=["ccta"])


def _require_model() -> None:
    try:
        ensure_ccta_model()
    except AnalysisError as error:
        raise as_http_error(error) from error


@router.get("/api/models/ccta")
def ccta_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **ccta_segmenter.describe()}


@router.post("/api/analyze/ccta")
async def analyze_ccta_endpoint(
    file: UploadFile = File(
        ...,
        description=(
            "The CT volume: a .nii/.nii.gz file, or a .zip containing one "
            "DICOM series."
        ),
    ),
    max_windows: int = Query(
        default=CCTA_MAX_WINDOWS,
        ge=1,
        le=4000,
        description=(
            "Sliding-window budget. A full-coverage pass over a 1 mm whole-"
            "chest volume needs several hundred windows and takes minutes on "
            "CPU. When the budget is short of that the analysis covers a "
            "centred crop and says so; it does not silently return a partial "
            "mask as if it were complete."
        ),
    ),
    include_gradcam: bool = Query(
        default=True,
        description=(
            "Compute 3-D Grad-CAM over the patch containing the most predicted "
            "lumen. Needs one backward pass, so it adds a few seconds."
        ),
    ),
    include_figures: bool = Query(
        default=True,
        description=(
            "Render slice, overlay, probability and projection panels. Roughly "
            "1-2 MB of PNG; set false for a JSON-only response."
        ),
    ),
    case_id: Optional[str] = Query(
        default=None,
        max_length=100,
        description="Archive the original upload against this case.",
    ),
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Segment the coronary lumen in an uploaded CT volume, with 3-D Grad-CAM.
    """
    # Before the body is read: refusing a 400 MB upload after buffering it is a
    # worse answer than refusing it first.
    _require_model()

    data = await read_upload(file, limit=MAX_CCTA_UPLOAD_BYTES)

    try:
        return analyze_ccta(
            data=data,
            filename=file.filename,
            max_windows=max_windows,
            include_gradcam=include_gradcam,
            include_figures=include_figures,
            case_id=case_id,
        )
    except AnalysisError as error:
        raise as_http_error(error) from error
