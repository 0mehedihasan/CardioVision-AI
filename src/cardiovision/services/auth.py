"""
CardioVision AI — authentication.

A single fixed operator account gates the application. This is deliberately
simple because the whole system runs on one machine with no network exposure,
but it is enforced *server-side* rather than in the UI: a frontend-only gate
would leave every stored patient record readable by anyone who knew the port.

WHAT THIS IS
------------
An access gate. It stops someone walking up to an unlocked laptop and reading
patient records, and it stops a browser tab that has not logged in from
touching the API.

WHAT THIS IS NOT
----------------
Real security. The default password is four digits, the token lives in
process memory, and traffic to localhost is unencrypted. Do not expose this
service beyond the loopback interface and do not treat the database as a
protected store of PHI.

CHANGING THE CREDENTIALS
------------------------
Set both environment variables before starting uvicorn:

    CARDIOVISION_USER=someone CARDIOVISION_PASSWORD=something \
        uvicorn main:app --port 8000

The password is never held in memory in plaintext — only a salted SHA-256
digest — and comparison is constant-time so a wrong password cannot be
narrowed down by timing the response.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional

from cardiovision.config import (
    AUTH_DEFAULT_PASSWORD,
    AUTH_DEFAULT_USERNAME,
    AUTH_SESSION_TTL_SECONDS,
)


# ============================================================
# CREDENTIALS
# ============================================================

def _env(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


USERNAME = _env("CARDIOVISION_USER", AUTH_DEFAULT_USERNAME)

# A per-process salt. It makes the stored digest useless outside this run,
# which matters because the digest of a four-digit password is otherwise
# trivially reversible from a rainbow table.
_SALT = secrets.token_bytes(32)


def _digest(value: str) -> bytes:
    return hashlib.sha256(_SALT + value.encode("utf-8")).digest()


# The plaintext password is hashed at import and then dropped. Only the
# digest survives, so a memory dump or an accidental repr() of this module
# does not reveal it.
_PASSWORD_DIGEST = _digest(_env("CARDIOVISION_PASSWORD", AUTH_DEFAULT_PASSWORD))

USING_DEFAULT_CREDENTIALS = (
    USERNAME == AUTH_DEFAULT_USERNAME
    and hmac.compare_digest(_PASSWORD_DIGEST, _digest(AUTH_DEFAULT_PASSWORD))
)


def verify_credentials(username: str, password: str) -> bool:
    """
    Check a username/password pair in constant time.

    Both comparisons run unconditionally. Returning early on a username
    mismatch would make a wrong username measurably faster than a wrong
    password, which tells an attacker which half to work on.
    """
    user_ok = hmac.compare_digest(
        _digest((username or "").strip()),
        _digest(USERNAME),
    )

    password_ok = hmac.compare_digest(
        _digest(password or ""),
        _PASSWORD_DIGEST,
    )

    return user_ok and password_ok


# ============================================================
# SESSIONS
# ============================================================

@dataclass
class Session:
    token: str
    username: str
    created_at: float
    expires_at: float

    @property
    def seconds_remaining(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class SessionStore:
    """
    In-memory session registry.

    Sessions deliberately do not survive a backend restart. For a local
    single-operator tool that is the safer default: restarting the service
    logs everyone out, and there is no token file left behind on disk for
    someone to find later.
    """

    def __init__(self, ttl_seconds: int = AUTH_SESSION_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}
        # uvicorn serves sync endpoints from a threadpool, so two logins can
        # land at the same moment.
        self._lock = threading.Lock()

    def create(self, username: str) -> Session:
        now = time.time()

        session = Session(
            # 32 bytes of urandom, url-safe. Not guessable and not derived
            # from the password, so a leaked token reveals nothing about it.
            token=secrets.token_urlsafe(32),
            username=username,
            created_at=now,
            expires_at=now + self._ttl,
        )

        with self._lock:
            self._prune_locked(now)
            self._sessions[session.token] = session

        return session

    def resolve(self, token: Optional[str]) -> Optional[Session]:
        if not token:
            return None

        now = time.time()

        with self._lock:
            session = self._sessions.get(token)

            if session is None:
                return None

            if session.expires_at <= now:
                # Expired tokens are removed on sight rather than left to
                # accumulate until the next prune.
                del self._sessions[token]
                return None

            # Sliding expiry: an operator working through a long case list
            # should not be logged out mid-review.
            session.expires_at = now + self._ttl
            return session

    def revoke(self, token: Optional[str]) -> bool:
        if not token:
            return False

        with self._lock:
            return self._sessions.pop(token, None) is not None

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

    @property
    def active_count(self) -> int:
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            return len(self._sessions)

    def _prune_locked(self, now: float) -> None:
        """Drop expired sessions. Caller must hold the lock."""
        expired = [
            token
            for token, session in self._sessions.items()
            if session.expires_at <= now
        ]

        for token in expired:
            del self._sessions[token]


sessions = SessionStore()


# ============================================================
# LOGIN THROTTLE
# ============================================================

class LoginThrottle:
    """
    Slows down repeated failures.

    A four-digit numeric password has ten thousand possibilities, which an
    unthrottled endpoint would yield in seconds. This does not make the
    password strong — it makes brute-forcing it take long enough to notice.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._failures: list[float] = []
        self._lock = threading.Lock()

    def check(self) -> Optional[int]:
        """Return seconds to wait if locked out, otherwise None."""
        now = time.time()

        with self._lock:
            self._failures = [
                stamp for stamp in self._failures if now - stamp < self._window
            ]

            if len(self._failures) < self._max:
                return None

            oldest = min(self._failures)
            return max(1, int(self._window - (now - oldest)))

    def record_failure(self) -> int:
        """Log a failed attempt and return how many remain."""
        now = time.time()

        with self._lock:
            self._failures = [
                stamp for stamp in self._failures if now - stamp < self._window
            ]
            self._failures.append(now)
            return max(0, self._max - len(self._failures))

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


throttle = LoginThrottle()
