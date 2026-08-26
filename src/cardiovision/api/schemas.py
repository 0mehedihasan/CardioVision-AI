"""
Request and response models for the HTTP layer.

These live apart from the routers because more than one router uses them and
because a schema change is an API contract change: keeping them in one file
makes that contract reviewable in a single diff.

Only the JSON bodies are here. Multipart uploads and query parameters are
declared inline on the endpoints that take them, where FastAPI can attach the
per-parameter documentation the OpenAPI schema needs.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# AUTHENTICATION
# ============================================================


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=200)
    password: str = Field(..., max_length=400)


# ============================================================
# CLINICAL Q&A
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
# CASES
# ============================================================


class CasePayload(BaseModel):
    """
    A case as the frontend holds it.

    Every field is optional: a case can legitimately be saved with nothing but
    a patient name, before any imaging exists. ``ccta``, ``echo`` and ``ecg``
    are independent — a case may hold any subset of them, and the store keeps
    whichever arrived rather than requiring a complete workup.

    Figures travel beside their analysis rather than inside it. ``images``
    carries the echo renders, ``ecg_figures`` the ECG ones and ``ccta_figures``
    the CT panels, all as data URLs, and the store writes them out as files so
    they are never held in a JSON column. Sending them nested inside the
    analysis also works — the store lifts them out — but the explicit field is
    what the frontend uses, because it makes a save that deliberately omits the
    figures distinguishable from one that forgot them.
    """

    case_id: Optional[str] = Field(default=None, max_length=100)
    patient: Optional[dict[str, Any]] = Field(default=None)
    clinical: Optional[dict[str, Any]] = Field(default=None)
    ccta: Optional[dict[str, Any]] = Field(default=None)
    echo: Optional[dict[str, Any]] = Field(default=None)
    ecg: Optional[dict[str, Any]] = Field(default=None)
    images: Optional[dict[str, str]] = Field(default=None)
    ecg_figures: Optional[dict[str, str]] = Field(default=None)
    ccta_figures: Optional[dict[str, str]] = Field(default=None)
    conversation: Optional[list[dict[str, Any]]] = Field(default=None)


# ============================================================
# INTEGRATED EVIDENCE AND REPORTS
# ============================================================


class ReportRequest(BaseModel):
    """
    Which case to aggregate or report on.

    Either form works. ``case`` reports on state the frontend is holding, which
    is the common path — analyses exist in the browser before anyone presses
    save, and requiring a save first would make the report lag the screen.
    ``case_id`` alone loads the saved case instead. When both arrive, the body
    wins, because it is the more recent of the two.
    """

    case_id: Optional[str] = Field(default=None, max_length=100)
    case: Optional[dict[str, Any]] = Field(default=None)
