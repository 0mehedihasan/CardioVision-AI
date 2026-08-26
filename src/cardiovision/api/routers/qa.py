"""
MedGemma clinical Q&A.

The prompt is assembled server-side from the structured case, by
:func:`cardiovision.services.case_context.build_case_context`. That is
deliberate: the context is where the model is told which findings are real,
which modalities have no trained model, and which classes are unreliable, and
none of that should be rewritable from the browser.

The rendered context comes back in ``context_preview`` so the UI can show
exactly what the model was told.
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, Depends, HTTPException

from cardiovision.api.deps import require_session
from cardiovision.api.schemas import (
    ClinicalQuestionRequest,
    ClinicalQuestionResponse,
)
from cardiovision.config import DEVICE, MEDGEMMA_NAME
from cardiovision.inference.medgemma import MedGemmaUnavailable, medgemma
from cardiovision.services import auth as auth_service
from cardiovision.services.case_context import build_case_context

router = APIRouter(tags=["qa"])


@router.post("/api/clinical-question", response_model=ClinicalQuestionResponse)
def ask_clinical_question(
    request: ClinicalQuestionRequest,
    session: auth_service.Session = Depends(require_session),
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
