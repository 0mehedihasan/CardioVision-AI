"""
Login, logout, and session validation.

The service module is imported as ``auth_service`` throughout: this module is
also called ``auth``, and a bare ``import auth`` here would resolve to whichever
one won the race.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from cardiovision.api.deps import bearer_token, require_session
from cardiovision.api.schemas import LoginRequest
from cardiovision.config import AUTH_SESSION_TTL_SECONDS
from cardiovision.services import auth as auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
def login(request: LoginRequest) -> dict[str, Any]:
    """
    Exchange credentials for a session token. Public.

    Failures are deliberately vague about *which* half was wrong — naming the
    field would let someone confirm a valid username before attacking the
    password.
    """
    locked_for = auth_service.throttle.check()

    if locked_for is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many failed sign-in attempts. Try again in "
                f"{locked_for} seconds."
            ),
        )

    if not auth_service.verify_credentials(request.username, request.password):
        remaining = auth_service.throttle.record_failure()

        detail = "Incorrect username or password."
        if remaining:
            detail += f" {remaining} attempt{'s' if remaining != 1 else ''} left."

        raise HTTPException(status_code=401, detail=detail)

    auth_service.throttle.reset()
    session = auth_service.sessions.create(auth_service.USERNAME)

    return {
        "success": True,
        "token": session.token,
        "username": session.username,
        "expires_in": session.seconds_remaining,
        # Surfaced so the UI can show a standing reminder rather than letting
        # a default password quietly become permanent.
        "using_default_credentials": auth_service.USING_DEFAULT_CREDENTIALS,
    }


@router.post("/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """
    Revoke the current token.

    Not gated behind require_session: logging out with an already-expired token
    should quietly succeed, not fail with a 401.
    """
    revoked = auth_service.sessions.revoke(bearer_token(authorization))

    return {"success": True, "revoked": revoked}


@router.get("/session")
def current_session(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    """Validate a stored token on page load, without a fresh login."""
    return {
        "success": True,
        "username": session.username,
        "expires_in": session.seconds_remaining,
        "session_ttl_seconds": AUTH_SESSION_TTL_SECONDS,
        "using_default_credentials": auth_service.USING_DEFAULT_CREDENTIALS,
    }
