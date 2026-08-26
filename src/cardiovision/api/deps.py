"""
Shared FastAPI dependencies.

Three things every router needs and none of them owns: proving there is a
session, reading an upload without letting it exhaust memory, and refusing
politely when the case database is down. Plus one translation: turning an
:class:`~cardiovision.analysis.AnalysisError` raised by the shared analysis core
into the HTTP status it was labelled with.

Extracted so the auth rule is written once. A second copy of ``require_session``
is how one endpoint ends up quietly unauthenticated.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, UploadFile

from cardiovision.analysis import AnalysisError
from cardiovision.config import MAX_UPLOAD_BYTES
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

__all__ = [
    "as_http_error",
    "bearer_token",
    "require_session",
    "require_store",
    "read_upload",
]


# ============================================================
# ERRORS
# ============================================================


def as_http_error(error: AnalysisError) -> HTTPException:
    """
    Map an analysis failure onto the status it already carries.

    The analysis core is imported by Streamlit as well as by FastAPI, so it
    cannot raise ``HTTPException`` itself. It labels the failure instead, and
    this is the one place that label becomes a response — so 415 for an
    unsupported format and 503 for an unloaded model stay exactly what they were
    when the payload builders lived in the routers.
    """
    return HTTPException(status_code=error.status_code, detail=str(error))



# ============================================================
# AUTHENTICATION
# ============================================================


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
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
) -> auth_service.Session:
    """
    Reject anything without a valid session token.

    Returning 401 with WWW-Authenticate lets the frontend distinguish "your
    session expired, log in again" from a genuine server error, which matters
    because the two need completely different handling in the UI.
    """
    session = auth_service.sessions.resolve(bearer_token(authorization))

    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Not signed in, or the session has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


# ============================================================
# STORAGE
# ============================================================


def require_store() -> None:
    """
    503 when the case database never came up.

    A dependency rather than a line at the top of five endpoints, because the
    one place it gets forgotten is the one that raises an opaque AttributeError
    instead of saying the database is down.
    """
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail=store.connect_error or "The case database is not available.",
        )


# ============================================================
# UPLOADS
# ============================================================


async def read_upload(
    upload: UploadFile,
    limit: Optional[int] = None,
) -> bytes:
    """
    Read an upload while enforcing the size cap.

    Chunked and checked as it goes, so a file over the limit is refused after a
    megabyte rather than after being buffered whole.

    ``limit`` overrides the default cap for endpoints whose input is genuinely
    in a different size class — a CCTA volume is an order of magnitude larger
    than an echo frame, and one shared limit would either reject a real study
    or let a stray gigabyte through the echo endpoint.
    """
    cap = MAX_UPLOAD_BYTES if limit is None else int(limit)

    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break

        total += len(chunk)
        if total > cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    "File is too large. The limit is "
                    f"{cap // (1024 * 1024)} MB."
                ),
            )

        chunks.append(chunk)

    return b"".join(chunks)
