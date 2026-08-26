"""
Patient case CRUD and stored renders.

Every route here is behind both a session and ``require_store``. Route order
matters: ``/api/cases/{case_id}`` would swallow any literal sibling registered
after it, so the literal ``/api/cases`` collection routes come first.
"""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from cardiovision.api.deps import require_session, require_store
from cardiovision.api.schemas import CasePayload
from cardiovision.services import auth as auth_service
from cardiovision.services.database import CaseStoreError, media_type_for, store

router = APIRouter(
    prefix="/api/cases",
    tags=["cases"],
    dependencies=[Depends(require_store)],
)


@router.get("")
def list_cases(
    search: str = Query(default="", max_length=200),
    limit: int = Query(default=200, ge=1, le=1000),
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """Saved case summaries, most recently updated first."""
    try:
        cases = store.list(search=search, limit=limit)
    except CaseStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"success": True, "cases": cases, "total": store.count()}


@router.post("")
def save_case(
    payload: CasePayload,
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """
    Create or update a case.

    An existing case_id updates in place; omitting it mints a new one. The
    caller gets the stored record back, so the frontend can adopt the generated
    ID and timestamps rather than guessing them.
    """
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


@router.get("/{case_id}")
def get_case(
    case_id: str,
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    case = store.get(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"No case {case_id}.")

    return {"success": True, "case": case}


@router.delete("/{case_id}")
def delete_case(
    case_id: str,
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    if not store.delete(case_id):
        raise HTTPException(status_code=404, detail=f"No case {case_id}.")

    return {"success": True, "deleted": case_id}


@router.get("/{case_id}/images/{name}")
def get_case_image(
    case_id: str,
    name: str,
    session: auth_service.Session = Depends(require_session),
) -> Response:
    """
    Serve one stored render.

    The store validates ``name`` against a fixed table of render keys, so a
    crafted path cannot reach outside the case directory. The content type
    comes from that same table rather than being assumed: the echo renders are
    PNG and the ECG figures are SVG, and an SVG served as image/png fails to
    draw in a way that looks exactly like a missing file.
    """
    payload = store.read_image(case_id, name)

    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stored '{name}' figure for case {case_id}.",
        )

    return Response(
        content=payload,
        media_type=media_type_for(name),
        # Stored renders are immutable: a new analysis overwrites the case or
        # creates a new one, so the browser can cache these hard.
        headers={"Cache-Control": "private, max-age=86400"},
    )
