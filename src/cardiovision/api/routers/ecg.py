"""
12-lead ECG classification.

Model card and analysis for the ECGResNet1D trained on PTB-XL. Five diagnostic
superclasses, independent sigmoids, so a recording can carry none of them or
several at once.

The pipeline lives in :func:`cardiovision.analysis.analyze_ecg`, shared with the
Streamlit client. Two things there are unlike the echo path.

**Uploads can be a pair.** WFDB — the format PTB-XL ships in, and therefore the
format this model was trained on — splits one recording across a ``.hea``
describing the layout and a ``.dat`` holding the samples. Neither is readable
alone. So the endpoint accepts a list of files and hands the extras to the
analysis as companions, rather than telling an operator holding the native format
to go and convert it.

**Every positive call travels with the precision it was measured at.** Macro
AUROC 0.913 reads as a single reassuring number and conceals that hypertrophy
sits at precision 0.361 — roughly two in three positive HYP calls are false.
``caveat`` on the affected class, and ``weak_class_warnings`` at the top level,
exist so that cannot be missed by a caller that only reads the summary.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile

from cardiovision.analysis import AnalysisError, analyze_ecg, ensure_ecg_model
from cardiovision.api.deps import as_http_error, read_upload, require_session
from cardiovision.inference.ecg import ecg_classifier
from cardiovision.services import auth as auth_service

router = APIRouter(tags=["ecg"])


def _require_model() -> None:
    try:
        ensure_ecg_model()
    except AnalysisError as error:
        raise as_http_error(error) from error


@router.get("/api/models/ecg")
def ecg_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **ecg_classifier.describe()}


@router.post("/api/analyze/ecg")
async def analyze_ecg_endpoint(
    file: UploadFile = File(
        ...,
        description=(
            "The recording. For WFDB upload the .hea here and send the .dat "
            "as a companion, or upload a .zip containing both."
        ),
    ),
    companions: list[UploadFile] = File(
        default=[],
        description=(
            "Additional files belonging to the same recording, e.g. the .dat "
            "that goes with a .hea. Matched by filename, not by order."
        ),
    ),
    sampling_frequency: Optional[float] = Query(
        default=None,
        gt=0,
        le=10000,
        description=(
            "Source sampling rate in Hz, for formats that do not record it "
            "(CSV, NPY). Getting this wrong rescales the whole recording in "
            "time, so the loader reports whatever it ends up using."
        ),
    ),
    target_class: Optional[str] = Query(
        default=None,
        description=(
            "Which class the saliency explains. Defaults to the highest-"
            "probability class, which is the one a reader is asking about."
        ),
    ),
    include_figures: bool = Query(
        default=True,
        description=(
            "Render the 12-lead strip and the lead-attribution chart. "
            "Roughly 160 KB of SVG; set false for a JSON-only response."
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
    Classify an uploaded 12-lead ECG into the five PTB-XL diagnostic
    superclasses, with per-lead gradient attribution.
    """
    _require_model()

    data = await read_upload(file)

    # Read here rather than in the analysis core: ``UploadFile`` is an HTTP
    # concept, and the shared core takes plain bytes so Streamlit can hand it a
    # file from disk.
    companion_bytes: dict[str, bytes] = {}
    for extra in companions:
        if not extra.filename:
            continue
        payload = await read_upload(extra)
        if payload:
            companion_bytes[extra.filename] = payload

    try:
        return analyze_ecg(
            data=data,
            filename=file.filename,
            companions=companion_bytes,
            sampling_frequency=sampling_frequency,
            target_class=target_class,
            include_figures=include_figures,
            case_id=case_id,
        )
    except AnalysisError as error:
        raise as_http_error(error) from error
