"""
CardioVision AI — FastAPI application factory.

The routes live in :mod:`cardiovision.api.routers`, one module per concern. This
file only assembles them, which keeps the interesting question — "what does this
endpoint promise?" — next to the endpoint instead of two hundred lines away.

Routes
------
GET    /                              service banner                  (public)
GET    /api/health                    load state + real modalities    (public)
POST   /api/auth/login                credentials -> session token    (public)
POST   /api/auth/logout               revoke the current token
GET    /api/auth/session              who am I, how long is it good for
GET    /api/models/ccta               CCTA model card
GET    /api/models/echo               echo model card
GET    /api/models/ecg                ECG model card
POST   /api/analyze/ccta              coronary lumen segmentation from a volume
POST   /api/analyze/echo              echo segmentation from an image
POST   /api/analyze/ecg               ECG classification from a recording
POST   /api/evidence                  aggregate a case, no language model
POST   /api/report                    structured report + MedGemma narrative
POST   /api/clinical-question         MedGemma Q&A, optionally with case context
GET    /api/cases                     list / search saved cases
POST   /api/cases                     create or update a case
GET    /api/cases/{id}                load one case
DELETE /api/cases/{id}                delete a case and its stored images
GET    /api/cases/{id}/images/{name}  serve a stored render
GET    /api/cases/{id}/report         the last report saved for a case

Everything except the banner, ``/api/health`` and login requires a session
token. Health stays public so the login screen can report whether the backend is
up and which models loaded before anyone signs in.

The four models load independently. A MedGemma failure does not stop echo
segmentation, an echo failure does not stop ECG classification, and load state
is reported through ``/api/health`` so the UI can only advertise what works.
``/api/evidence`` needs no model at all: it aggregates results that were already
computed, so it answers correctly on a server where nothing loaded.
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cardiovision.api.routers import auth as auth_router
from cardiovision.api.routers import cases as cases_router
from cardiovision.api.routers import ccta as ccta_router
from cardiovision.api.routers import ecg as ecg_router
from cardiovision.api.routers import echo as echo_router
from cardiovision.api.routers import health as health_router
from cardiovision.api.routers import qa as qa_router
from cardiovision.api.routers import report as report_router
from cardiovision.config import APP_NAME, APP_VERSION
from cardiovision.inference.ccta import CctaModelUnavailable, ccta_segmenter
from cardiovision.inference.ecg import EcgModelUnavailable, ecg_classifier
from cardiovision.inference.echo import EchoModelUnavailable, echo_segmenter
from cardiovision.inference.medgemma import MedGemmaUnavailable, medgemma
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

__all__ = ["create_app", "app"]

# Vite's dev server. Anything else is a deployment change, not a code change,
# and belongs in CARDIOVISION_CORS_ORIGINS.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _cors_origins() -> list[str]:
    """Allowed origins, overridable with a comma-separated env var."""
    configured = os.environ.get("CARDIOVISION_CORS_ORIGINS", "").strip()

    if not configured:
        return list(DEFAULT_CORS_ORIGINS)

    return [origin.strip() for origin in configured.split(",") if origin.strip()]


# ============================================================
# STARTUP
# ============================================================


def _load_model(label: str, model: object, expected: type[Exception],
                skip_flag: str) -> None:
    """
    Load one model, and never let its failure take down the service.

    Each model is optional at startup: the echo segmenter is no use for an ECG
    and vice versa, so one missing checkpoint should cost exactly one modality.
    The expected unavailability error is reported as a warning; anything else
    gets a traceback, because it is a bug rather than a missing file.
    """
    if _truthy(os.environ.get(skip_flag)):
        print(f"Skipping {label} ({skip_flag} is set).")
        return

    try:
        model.load()                                    # type: ignore[attr-defined]
    except expected as error:
        print(f"[warning] {label} unavailable: {error}")
    except Exception as error:                          # pragma: no cover
        print(f"[warning] Unexpected {label} failure: {error}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring up storage and the four models, then report what worked."""
    # Storage first. Without it the app can still analyse images and
    # recordings, so a storage failure is reported rather than fatal.
    try:
        store.connect()
        print(f"Case database ready ({store.count()} saved cases).")
    except Exception as error:                          # pragma: no cover
        print(f"[warning] Case database unavailable: {error}")
        traceback.print_exc()

    _load_model("CCTA model", ccta_segmenter, CctaModelUnavailable,
                "CARDIOVISION_SKIP_CCTA")
    _load_model("Echo model", echo_segmenter, EchoModelUnavailable,
                "CARDIOVISION_SKIP_ECHO")
    _load_model("ECG model", ecg_classifier, EcgModelUnavailable,
                "CARDIOVISION_SKIP_ECG")
    _load_model("MedGemma", medgemma, MedGemmaUnavailable,
                "CARDIOVISION_SKIP_MEDGEMMA")

    def state(ready: bool) -> str:
        return "ready" if ready else "unavailable"

    print("CardioVision AI backend ready.")
    print(f"  CCTA segmentation  : {state(ccta_segmenter.is_loaded)}")
    print(f"  Echo segmentation  : {state(echo_segmenter.is_loaded)}")
    print(f"  ECG classification : {state(ecg_classifier.is_loaded)}")
    print(f"  MedGemma           : {state(medgemma.is_loaded)}")
    print(f"  Case storage       : {state(store.is_ready)}")
    print(f"  Sign in as         : {auth_service.USERNAME}")

    if auth_service.USING_DEFAULT_CREDENTIALS:
        print(
            "  [notice] Using the default password. This is an access gate, "
            "not security.\n"
            "           Override it with CARDIOVISION_USER and "
            "CARDIOVISION_PASSWORD,\n"
            "           and keep this service bound to localhost."
        )

    yield

    store.close()


# ============================================================
# APPLICATION
# ============================================================


def create_app() -> FastAPI:
    """
    Build the application.

    A factory rather than a module-level singleton so tests can construct an
    app per case — the models are process-wide singletons, but the routing
    table, middleware and dependency overrides are not, and sharing them is how
    one test's override leaks into the next.
    """
    app = FastAPI(
        title=f"{APP_NAME} API",
        description=(
            "Local clinical intelligence backend: coronary CT angiography "
            "lumen segmentation, echocardiography segmentation, 12-lead ECG "
            "classification, deterministic multimodal evidence aggregation, "
            "and MedGemma reporting over the findings."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # health last would work too, but registering the public routes first keeps
    # the OpenAPI page in the order an operator reads it.
    app.include_router(health_router.router)
    app.include_router(auth_router.router)
    app.include_router(ccta_router.router)
    app.include_router(echo_router.router)
    app.include_router(ecg_router.router)
    app.include_router(report_router.router)
    app.include_router(qa_router.router)
    app.include_router(cases_router.router)

    return app


# Module-level instance for `uvicorn cardiovision.api.app:app`, which is what
# the console script and the docs both use.
app = create_app()
