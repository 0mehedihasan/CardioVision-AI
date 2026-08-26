"""
Coronary CT angiography lumen segmentation.

Model card and analysis for the Small3DUNet trained on MedHK23/CCA. This is the
weakest of the three models in the project and the endpoint is built to say so
rather than to look confident: the model card carries the three-case test split
on the same level as the Dice score, and every response repeats that the output
is a lumen mask with no stenosis grade, no calcium score and no vessel identity.

Three things here are unlike the echo and ECG endpoints.

**The upload is large and the compute is bounded.** A 0.5 mm study is roughly
830x830x580 voxels; resampled to 1 mm it still needs some 490 sliding windows,
which is minutes of CPU. So the request carries a window budget, and when the
budget cannot cover the volume the endpoint analyses a centred crop and reports
exactly what it skipped. ``coverage.complete: false`` is a first-class result,
not an error.

**Absence of mask is not absence of disease, and unanalysed is not absent.**
The response distinguishes the two: ``analysed_percent`` says how much was
looked at, and the rendered slices are annotated where they extend beyond it.

**The shipped example figures are training data.** ``models/ccta/case_*_xai.png``
came from the notebook's own test cases and are never returned here. Everything
in ``figures`` was rendered from the uploaded volume.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from cardiovision.api.deps import read_upload, require_session
from cardiovision.config import (
    ALLOWED_CCTA_SUFFIXES,
    CCTA_INFERENCE_OVERLAP,
    CCTA_MAX_WINDOWS,
    CCTA_PATCH_SIZE,
    CCTA_PRESENCE_THRESHOLD_VOXELS,
    CCTA_WEAK_NOTES,
    DEVICE,
    MAX_CCTA_UPLOAD_BYTES,
)
from cardiovision.inference.ccta import CctaModelUnavailable, ccta_segmenter
from cardiovision.preprocessing.ccta_io import (
    UnsupportedVolumeError,
    load_ccta_volume,
)
from cardiovision.rendering.ccta import render_ccta_images
from cardiovision.services import auth as auth_service
from cardiovision.services.database import store

router = APIRouter(tags=["ccta"])


def _require_model() -> None:
    if not ccta_segmenter.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                ccta_segmenter.load_error
                or "The CCTA segmentation model is not loaded."
            ),
        )


@router.get("/api/models/ccta")
def ccta_model_card(
    session: auth_service.Session = Depends(require_session),
) -> dict[str, Any]:
    _require_model()
    return {"success": True, **ccta_segmenter.describe()}


@router.post("/api/analyze/ccta")
async def analyze_ccta(
    file: UploadFile = File(
        ...,
        description=(
            "The CT volume: a .nii/.nii.gz file, or a .zip containing one "
            "DICOM series."
        ),
    ),
    max_windows: int = Query(
        default=CCTA_MAX_WINDOWS,
        ge=1,
        le=4000,
        description=(
            "Sliding-window budget. A full-coverage pass over a 1 mm whole-"
            "chest volume needs several hundred windows and takes minutes on "
            "CPU. When the budget is short of that the analysis covers a "
            "centred crop and says so; it does not silently return a partial "
            "mask as if it were complete."
        ),
    ),
    include_gradcam: bool = Query(
        default=True,
        description=(
            "Compute 3-D Grad-CAM over the patch containing the most predicted "
            "lumen. Needs one backward pass, so it adds a few seconds."
        ),
    ),
    include_figures: bool = Query(
        default=True,
        description=(
            "Render slice, overlay, probability and projection panels. Roughly "
            "1-2 MB of PNG; set false for a JSON-only response."
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
    Segment the coronary lumen in an uploaded CT volume, with 3-D Grad-CAM.
    """
    _require_model()

    data = await read_upload(file, limit=MAX_CCTA_UPLOAD_BYTES)

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # ---- decode + resample --------------------------------------
    #
    # Resampling a whole-chest study to 1 mm is itself expensive, so any failure
    # here is reported before the model is touched.
    try:
        loaded = load_ccta_volume(data=data, filename=file.filename or "")
    except UnsupportedVolumeError as error:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{error} Accepted extensions: "
                f"{', '.join(sorted(ALLOWED_CCTA_SUFFIXES))}."
            ),
        ) from error
    except MemoryError as error:
        raise HTTPException(
            status_code=413,
            detail=(
                "The volume was too large to resample in memory. Crop it to "
                "the cardiac region and upload that instead."
            ),
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded volume: {error}",
        ) from error

    # ---- segment ------------------------------------------------
    try:
        analysis = ccta_segmenter.analyze(
            volume=loaded.volume,
            spacing_mm=loaded.spacing_mm,
            max_windows=max_windows,
            include_gradcam=include_gradcam,
        )
    except CctaModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except MemoryError as error:
        raise HTTPException(
            status_code=413,
            detail=(
                "Ran out of memory during inference. Lower max_windows, or "
                "crop the volume to the cardiac region."
            ),
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"CCTA segmentation failed: {error}",
        ) from error

    # ---- render -------------------------------------------------
    #
    # Same rule as the other two endpoints: the mask and the measurements are
    # what the operator waited minutes for. A renderer failure degrades to JSON
    # with a note, never to a 500.
    figures: dict[str, str] = {}
    figure_meta: dict[str, Any] = {}
    figure_notes: list[str] = []

    if include_figures:
        try:
            figures, figure_meta, figure_notes = render_ccta_images(
                volume=loaded.volume,
                mask=analysis.mask,
                probability=analysis.probability,
                analysed=analysis.analysed,
                gradcam=analysis.gradcam,
                gradcam_origin=analysis.gradcam_origin,
            )
        except Exception as error:                      # pragma: no cover
            traceback.print_exc()
            figure_notes = [
                f"The CCTA figures could not be rendered ({error}). The "
                "measurements below are unaffected."
            ]

    model_card = ccta_segmenter.describe()
    complete = analysis.windows_run >= analysis.windows_total

    response: dict[str, Any] = {
        "success": True,
        "modality": "ccta",
        "analyzed": True,
        "device": analysis.compute_device,
        "configured_device": DEVICE,
        "model": model_card,
        "input": {
            "filename": file.filename,
            **loaded.summary(),
        },
        "threshold": round(analysis.threshold, 4),
        "findings": [finding.to_dict() for finding in analysis.findings],
        "quantification": {
            "presence_threshold_voxels": CCTA_PRESENCE_THRESHOLD_VOXELS,
            "voxel_volume_mm3": 1.0,
            "note": (
                "Volumes are computed on the 1 mm isotropic grid the model ran "
                "on, so one voxel is one cubic millimetre. A mask smaller than "
                "the presence threshold is reported as not present, which is a "
                "size cutoff and not a clinical finding."
            ),
        },
        "coverage": {
            "complete": complete,
            "coverage": round(analysis.coverage, 6),
            "analysed_percent": round(analysis.coverage * 100.0, 2),
            "windows_run": analysis.windows_run,
            "windows_total": analysis.windows_total,
            "patch_size": list(CCTA_PATCH_SIZE),
            "overlap": CCTA_INFERENCE_OVERLAP,
            "note": (
                "Everything outside the analysed region was not examined. In "
                "the mask and the figures it is zero, which means not looked "
                "at — not absent."
                if not complete else
                "The whole volume was covered by the sliding window."
            ),
        },
        "explainability": {
            "available": analysis.gradcam_available,
            "method": "3-D Grad-CAM",
            "target_layer": model_card["explainability"]["target_layer"],
            "scope": model_card["explainability"]["scope"],
            "origin": (
                list(analysis.gradcam_origin) if analysis.gradcam_origin else None
            ),
            "shape": (
                list(analysis.gradcam_shape) if analysis.gradcam_shape else None
            ),
            "device": analysis.gradcam_device,
            "note": (
                "Grad-CAM shows where activation supported the model's own "
                "output. It does not localise disease, and it was computed on "
                "one patch only."
            ),
        },
        "limitations": list(CCTA_WEAK_NOTES),
        "inference_ms": round(analysis.inference_ms, 1),
        "figures": figures,
        "figure_meta": figure_meta,
        "notes": loaded.notes + analysis.notes + figure_notes,
    }

    if case_id and store.is_ready:
        try:
            response["source_filename"] = store.store_source_file(
                case_id=case_id,
                filename=file.filename or "ccta",
                data=data,
            )
        except Exception as error:                      # pragma: no cover
            print(f"[warning] Could not archive the CCTA upload: {error}")
            response["notes"] = response["notes"] + [
                "The original volume could not be archived with this case, so "
                "re-analysing it later will need it uploaded again."
            ]

    return response
