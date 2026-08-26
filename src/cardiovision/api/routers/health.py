"""
Service banner and health.

Both are public. The login screen has to report whether the backend is up and
which models loaded *before* anyone signs in, and nothing here is patient data —
the storage block is deliberately a count and not a case list.

``MODALITY_STATUS`` is the declared truth about what exists; this router
overwrites the ``available`` flag for the trained modalities with whether their
model actually loaded. That way a checkpoint that failed to load shows as
unavailable rather than as a capability the UI will offer and the backend will
then refuse.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from cardiovision.config import (
    APP_NAME,
    APP_VERSION,
    DEVICE,
    MEDGEMMA_NAME,
    MODALITY_STATUS,
)
from cardiovision.inference.ccta import ccta_segmenter
from cardiovision.inference.ecg import ecg_classifier
from cardiovision.inference.echo import echo_segmenter
from cardiovision.inference.medgemma import medgemma
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

router = APIRouter(tags=["health"])

# Modality key -> the loaded model backing it. Only the trained ones appear;
# clinical risk and fusion have no model to consult, so their declared
# `available: False` stands unmodified. Fusion is not an omission — the
# integration layer is deterministic software, and reporting it as a loadable
# model would be the exact overstatement this endpoint exists to prevent.
_BACKED_BY = {
    "ccta": ccta_segmenter,
    "echo": echo_segmenter,
    "ecg": ecg_classifier,
}


@router.get("/api/health")
def health_check() -> dict[str, Any]:
    """Public service, model and storage state."""
    modalities: dict[str, Any] = {}

    for key, status in MODALITY_STATUS.items():
        entry = dict(status)
        model = _BACKED_BY.get(key)

        if model is not None:
            entry["available"] = model.is_loaded
            if not model.is_loaded:
                entry["note"] = (
                    model.load_error
                    or f"The {entry['label'].lower()} model is not loaded."
                )

        modalities[key] = entry

    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
        "device": DEVICE,
        "models": {
            "medgemma": {
                "name": MEDGEMMA_NAME,
                "loaded": medgemma.is_loaded,
                "error": medgemma.load_error,
            },
            "ccta": {
                "name": "CardioVision CCTA (Small3DUNet)",
                "loaded": ccta_segmenter.is_loaded,
                "error": ccta_segmenter.load_error,
            },
            "echo": {
                "name": "CardioVision Echo (UNet++ / EfficientNet-B3)",
                "loaded": echo_segmenter.is_loaded,
                "error": echo_segmenter.load_error,
            },
            "ecg": {
                "name": "CardioVision ECG (ECGResNet1D)",
                "loaded": ecg_classifier.is_loaded,
                "error": ecg_classifier.load_error,
            },
        },
        "storage": {
            "ready": store.is_ready,
            "error": store.connect_error,
            # Deliberately only a count. A case list is patient data and
            # requires a session.
            "saved_cases": store.count() if store.is_ready else 0,
        },
        "auth": {
            "required": True,
            "username": auth_service.USERNAME,
            "using_default_credentials": auth_service.USING_DEFAULT_CREDENTIALS,
        },
        # Kept for compatibility with the original frontend.
        "medgemma_loaded": medgemma.is_loaded,
        "model": MEDGEMMA_NAME,
        "modalities": modalities,
    }


@router.get("/")
def root() -> dict[str, Any]:
    """Service banner. The route list is generated, so it cannot go stale."""
    return {
        "service": APP_NAME,
        "status": "running",
        "version": APP_VERSION,
        "device": DEVICE,
        "models": {
            "medgemma": medgemma.is_loaded,
            "ccta_segmentation": ccta_segmenter.is_loaded,
            "echo_segmentation": echo_segmenter.is_loaded,
            "ecg_classification": ecg_classifier.is_loaded,
        },
        "model": MEDGEMMA_NAME,
        "auth_required": True,
        "docs": "/docs",
    }
