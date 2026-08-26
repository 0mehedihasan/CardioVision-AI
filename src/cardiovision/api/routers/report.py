"""
Integrated evidence and the structured clinical AI report.

Two endpoints over the same deterministic layer.

``POST /api/evidence`` returns the aggregated evidence for a case with no
language model involved at all: what each model reported, what is missing, what
the uncertainties are. It is the honest core of the integration and it works
whether or not MedGemma is installed.

``POST /api/report`` adds the MedGemma narrative on top. The report is assembled
first and the summary is written into it afterwards, so a MedGemma failure costs
one field and returns 200 with ``ai_summary: null`` and a reason — not a 500 that
loses the model findings the operator waited minutes for.

Nothing here re-runs a model. Both endpoints read the analysis results already
on the case, which is what makes them cheap enough to call on every render.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from cardiovision.api.deps import require_session
from cardiovision.api.schemas import ReportRequest
from cardiovision.fusion import build_case_evidence, build_report
from cardiovision.fusion.report import build_report_prompt
from cardiovision.inference.medgemma import MedGemmaUnavailable, medgemma
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

router = APIRouter(tags=["report"])


def _resolve_case(
    request: ReportRequest,
) -> tuple[dict[str, Any], str]:
    """
    The case to report on, from the request body or from storage.

    A body wins over a stored case: the frontend holds analyses that have not
    been saved yet, and refusing to report on them would force a save before
    every report. When only a ``case_id`` arrives, storage is the source.
    """
    if request.case:
        case = dict(request.case)
        if request.case_id and not case.get("case_id"):
            case["case_id"] = request.case_id
        return case, "request"

    if not request.case_id:
        raise HTTPException(
            status_code=422,
            detail="Provide either a case body or a case_id.",
        )

    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Case storage is unavailable, so a case can only be reported "
                "on by sending it in the request body."
            ),
        )

    stored = store.get(request.case_id)

    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"No case {request.case_id!r} is saved.",
        )

    return stored, "storage"


@router.post("/api/evidence")
def integrated_evidence(
    request: ReportRequest,
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Aggregate a case into structured evidence. No language model involved.
    """
    case, source = _resolve_case(request)

    try:
        evidence = build_case_evidence(case)
    except Exception as error:                          # pragma: no cover
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not aggregate the case evidence: {error}",
        ) from error

    return {
        "success": True,
        "case_source": source,
        **evidence.to_dict(),
    }


@router.post("/api/report")
def clinical_report(
    request: ReportRequest,
    include_summary: bool = Query(
        default=True,
        description=(
            "Have MedGemma write the narrative summary. The structured report "
            "is identical either way; set false to skip a slow local "
            "generation."
        ),
    ),
    include_prompt: bool = Query(
        default=False,
        description=(
            "Return the exact text MedGemma was given, so a reader can check "
            "that the narrative claims nothing the evidence did not."
        ),
    ),
    save: bool = Query(
        default=False,
        description="Store the finished report against the case.",
    ),
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Build the structured clinical AI report for a case.
    """
    case, source = _resolve_case(request)

    try:
        evidence = build_case_evidence(case)
    except Exception as error:                          # pragma: no cover
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not aggregate the case evidence: {error}",
        ) from error

    prompt = build_report_prompt(evidence)

    summary: Optional[str] = None
    summary_error: Optional[str] = None

    if not include_summary:
        summary_error = "The narrative summary was not requested."
    elif not evidence.has_any_findings:
        # Nothing has been analysed. Asking a language model to summarise an
        # empty case invites it to fill the gap, and the structured report
        # already says the case is empty.
        summary_error = (
            "No modality has been analysed on this case, so there is nothing "
            "for the language model to summarise."
        )
    elif not medgemma.is_loaded:
        summary_error = (
            medgemma.load_error
            or "The clinical language model is not loaded."
        )
    else:
        try:
            summary = medgemma.generate(
                question=prompt["question"],
                context=prompt["context"],
            )
        except MedGemmaUnavailable as error:
            summary_error = str(error)
        except Exception as error:                      # pragma: no cover
            traceback.print_exc()
            summary_error = f"The summary could not be generated: {error}"

    report = build_report(
        evidence=evidence,
        patient=case.get("patient"),
        ai_summary=summary,
        ai_error=summary_error,
    )

    response: dict[str, Any] = {
        "success": True,
        "case_source": source,
        "report": report,
    }

    if include_prompt:
        response["prompt"] = prompt

    if save:
        case_id = report.get("case_id")

        if not case_id:
            response["saved"] = False
            response["save_error"] = (
                "The report was not saved because the case has no id. Save the "
                "case first."
            )
        elif not store.is_ready:
            response["saved"] = False
            response["save_error"] = "Case storage is unavailable."
        else:
            try:
                store.save_report(case_id=case_id, report=report)
                response["saved"] = True
            except Exception as error:                  # pragma: no cover
                traceback.print_exc()
                response["saved"] = False
                response["save_error"] = str(error)

    return response


@router.get("/api/cases/{case_id}/report")
def stored_report(
    case_id: str,
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    The last report saved for a case, if one was ever saved.

    Returns ``report: null`` rather than 404 when the case exists but has no
    report: "this case has not been reported on" is a normal state, and a 404
    would be indistinguishable from a missing case.
    """
    if not store.is_ready:
        raise HTTPException(status_code=503, detail="Case storage is unavailable.")

    stored = store.get(case_id)

    if not stored:
        raise HTTPException(status_code=404, detail=f"No case {case_id!r} is saved.")

    return {
        "success": True,
        "case_id": case_id,
        "report": stored.get("report"),
    }
