"""
12-lead ECG classification.

Model card and analysis for the ECGResNet1D trained on PTB-XL. Five diagnostic
superclasses, independent sigmoids, so a recording can carry none of them or
several at once.

Two things here are unlike the echo endpoint.

**Uploads can be a pair.** WFDB — the format PTB-XL ships in, and therefore the
format this model was trained on — splits one recording across a ``.hea``
describing the layout and a ``.dat`` holding the samples. Neither is readable
alone. So the endpoint accepts a list of files and hands the extras to the
loader as companions, rather than telling an operator holding the native format
to go and convert it.

**Every positive call travels with the precision it was measured at.** Macro
AUROC 0.913 reads as a single reassuring number and conceals that hypertrophy
sits at precision 0.361 — roughly two in three positive HYP calls are false.
``caveat`` on the affected class, and ``weak_class_warnings`` at the top level,
exist so that cannot be missed by a caller that only reads the summary.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from cardiovision.api.deps import read_upload, require_session
from cardiovision.config import (
    ALLOWED_ECG_SUFFIXES,
    DEVICE,
    ECG_BANDPASS_HIGH_HZ,
    ECG_BANDPASS_LOW_HZ,
    ECG_CLASS_NAMES,
    ECG_CLIP_RANGE,
    ECG_DURATION_SECONDS,
    ECG_INPUT_LENGTH,
    ECG_NORMALIZATION,
    ECG_TARGET_FS,
    ECG_THRESHOLD,
    ECG_WEAK_CLASSES,
)
from cardiovision.inference.ecg import EcgModelUnavailable, ecg_classifier
from cardiovision.preprocessing.ecg_io import UnsupportedEcgError, load_ecg
from cardiovision.rendering.ecg import render_ecg_images
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

router = APIRouter(tags=["ecg"])


def _require_model() -> None:
    if not ecg_classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                ecg_classifier.load_error
                or "The ECG classification model is not loaded."
            ),
        )


@router.get("/api/models/ecg")
def ecg_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **ecg_classifier.describe()}


@router.post("/api/analyze/ecg")
async def analyze_ecg(
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

    if target_class is not None:
        target_class = target_class.strip().upper() or None

        if target_class is not None and target_class not in ECG_CLASS_NAMES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown target_class {target_class!r}. Expected one of "
                    f"{', '.join(ECG_CLASS_NAMES)}."
                ),
            )

    data = await read_upload(file)

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Companions are keyed by filename because WFDB headers name their signal
    # file explicitly; relying on upload order would break the moment a browser
    # reordered the form fields.
    companion_bytes: dict[str, bytes] = {}
    for extra in companions:
        if not extra.filename:
            continue
        payload = await read_upload(extra)
        if payload:
            companion_bytes[extra.filename] = payload

    # ---- decode + preprocess ------------------------------------
    try:
        loaded = load_ecg(
            filename=file.filename or "",
            data=data,
            sampling_frequency=sampling_frequency,
            companion=companion_bytes,
        )
    except UnsupportedEcgError as error:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{error} Accepted extensions: "
                f"{', '.join(sorted(ALLOWED_ECG_SUFFIXES))}."
            ),
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded ECG: {error}",
        ) from error

    # ---- classify -----------------------------------------------
    try:
        analysis = ecg_classifier.analyze(
            signal=loaded.signal,
            target_class=target_class,
        )
    except EcgModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"ECG classification failed: {error}",
        ) from error

    # ---- render -------------------------------------------------
    #
    # A figure failure must not lose the classification: the probabilities are
    # the result the operator waited for, and the strip is how they are read.
    # So a broken renderer degrades to JSON with a note, not to a 500.
    figures: dict[str, str] = {}
    figure_notes: list[str] = []

    if include_figures:
        try:
            figures = render_ecg_images(
                loaded.display_signal,
                analysis,
                lead_names=loaded.lead_names,
                units=loaded.units,
                sampling_frequency=ECG_TARGET_FS,
                record_name=loaded.record_name,
            )
        except Exception as error:                     # pragma: no cover
            traceback.print_exc()
            figure_notes.append(
                "The waveform figures could not be rendered "
                f"({error}). The classification below is unaffected."
            )

    model_card = ecg_classifier.describe()

    # Weak-class warnings, but only for classes this recording actually called
    # positive. Listing HYP's precision on a recording that came back HYP-
    # negative would be noise, and noise is what gets filtered out by eye.
    weak_warnings = {
        name: ECG_WEAK_CLASSES[name]
        for name in analysis.positive_classes
        if name in ECG_WEAK_CLASSES
    }

    response: dict[str, Any] = {
        "success": True,
        "modality": "ecg",
        # The device the forward pass actually ran on. Saliency needs
        # gradients, which can force a CPU fallback on MPS.
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": file.filename,
            "companions": sorted(companion_bytes),
            **loaded.to_dict(),
        },
        "preprocessing": {
            "bandpass_hz": [ECG_BANDPASS_LOW_HZ, ECG_BANDPASS_HIGH_HZ],
            "resampled_to_hz": ECG_TARGET_FS,
            "length_samples": ECG_INPUT_LENGTH,
            "length_seconds": ECG_DURATION_SECONDS,
            "normalization": ECG_NORMALIZATION,
            "clip_range": list(ECG_CLIP_RANGE),
            "note": (
                "Identical to the training chain. The normalisation is "
                "per-lead robust median/IQR, so the model's input is in "
                "dimensionless IQR multiples — which is why the figures plot "
                "the pre-normalisation signal instead."
            ),
        },
        **analysis.to_dict(),
        "weak_class_warnings": weak_warnings,
        "threshold_note": (
            f"A class is called positive at p >= {ECG_THRESHOLD}. Every "
            "probability is returned, so a different operating point can be "
            "applied without re-running the model — but the precision and "
            "recall in the model card were measured at this threshold and do "
            "not carry over to another one."
        ),
        "figures": figures,
        "notes": loaded.notes + analysis.notes + figure_notes,
    }

    # Archive the original bytes against the case, if one is open. Same rule as
    # the echo endpoint: failing to archive must not fail the analysis the
    # operator just waited for.
    if case_id and store.is_ready:
        try:
            response["source_filename"] = store.store_source_file(
                case_id=case_id,
                filename=file.filename or "ecg",
                data=data,
            )
        except Exception as error:                     # pragma: no cover
            print(f"[warning] Could not archive the ECG upload: {error}")
            response["notes"] = response["notes"] + [
                "The original recording could not be archived with this case, "
                "so re-analysing it later will need it uploaded again."
            ]

    return response
