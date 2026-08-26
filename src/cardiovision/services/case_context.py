"""
CardioVision AI — case context construction.

Turns the structured case state sent by the frontend into the plain-text
CASE CONTEXT block handed to MedGemma.

The single most important job here is the "NOT AVAILABLE" section. MedGemma
is told never to invent findings, but the reliable way to hold it to that is
to state explicitly which modalities have no model behind them. Without
this, a question like "what do the CT findings show?" invites a fabricated
answer.
"""

from __future__ import annotations

from typing import Any, Optional

from config import MODALITY_STATUS
from database import derive_age


# ============================================================
# PATIENT SECTION
# ============================================================

def _patient_lines(patient: dict[str, Any]) -> list[str]:
    """
    Demographics as recorded on the case.

    The name is deliberately NOT sent. It adds nothing a clinical answer can
    use, and keeping identifiers out of the prompt limits how far they travel
    even within a local process.
    """
    lines: list[str] = []

    # Derived from the date of birth rather than taken from the typed clinical
    # age, which goes stale the moment a birthday passes.
    age = derive_age(patient.get("dateOfBirth") or "")
    if age is not None:
        lines.append(f"  Age: {age} years (derived from date of birth)")

    sex = (patient.get("sex") or "").strip()
    if sex:
        lines.append(f"  Sex: {sex}")

    study_date = (patient.get("studyDate") or "").strip()
    if study_date:
        lines.append(f"  Study date: {study_date}")

    notes = (patient.get("notes") or "").strip()
    if notes:
        lines.append(f"  Clinician's notes: {notes}")

    return lines


# ============================================================
# CLINICAL SECTION
# ============================================================

_CLINICAL_LABELS = (
    ("age", "Age"),
    ("sex", "Sex"),
    ("bloodPressure", "Blood pressure"),
    ("heartRate", "Heart rate"),
)

_RISK_FACTORS = (
    ("diabetes", "Diabetes"),
    ("hypertension", "Hypertension"),
    ("smoking", "Smoking"),
)


def _clinical_lines(clinical: dict[str, Any], has_patient_age: bool = False) -> list[str]:
    lines: list[str] = []

    for key, label in _CLINICAL_LABELS:
        value = clinical.get(key)
        if value in (None, "", False):
            continue

        # A date of birth is authoritative, so the typed age is redundant and
        # can contradict it. Emitting both would leave the model to pick.
        if key == "age" and has_patient_age:
            continue

        if key == "heartRate":
            lines.append(f"  {label}: {value} bpm")
        else:
            lines.append(f"  {label}: {value}")

    reported = [
        label for key, label in _RISK_FACTORS if clinical.get(key) is True
    ]

    # An unticked checkbox is NOT a clinical denial. The form ships these
    # fields defaulting to false, so "false" means the clinician never
    # touched the box — it does not mean they asked the patient and were
    # told no. Emitting these as "not reported" previously handed MedGemma
    # a negative history that nobody ever took, which it would then repeat
    # back as fact. Absence of a tick is absence of information.
    unticked = [
        label for key, label in _RISK_FACTORS if clinical.get(key) is not True
    ]

    if reported:
        lines.append(f"  Risk factors reported present: {', '.join(reported)}")

    if unticked and lines:
        lines.append(
            f"  Not recorded either way: {', '.join(unticked)}. Treat these "
            "as UNKNOWN, not as absent — the form was simply left blank. Do "
            "not describe the patient as having denied them."
        )

    symptoms = (clinical.get("symptoms") or "").strip()
    if symptoms:
        lines.append(f"  Presenting symptoms: {symptoms}")

    return lines


# ============================================================
# ECHO SECTION
# ============================================================

def _echo_lines(echo: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    model = echo.get("model") or {}
    metrics = model.get("metrics") or {}
    source = echo.get("input") or {}
    orientation = echo.get("orientation") or {}
    quantification = echo.get("quantification") or {}

    architecture = model.get("architecture", "UNet++")
    encoder = model.get("encoder", "EfficientNet-B3")

    lines.append(
        f"  Model: {architecture} / {encoder}, 4-class cardiac structure "
        "segmentation."
    )

    test_dice = metrics.get("test_dice")
    if test_dice is not None:
        lines.append(
            f"  Model accuracy on a held-out test cohort: Dice {test_dice} "
            f"({metrics.get('test_patients', '?')} patients). This describes "
            "the model in general, not the certainty of this particular "
            "segmentation."
        )

    image_format = source.get("format")
    if image_format:
        detail = f"  Source image: {image_format.upper()}"
        frame_index = source.get("frame_index")
        frame_count = source.get("frame_count")
        if frame_index is not None and frame_count:
            detail += f", frame {frame_index} of {frame_count}"
        lines.append(detail + ".")

    structures = echo.get("structures") or []
    calibrated = bool(source.get("has_spatial_calibration"))

    if structures:
        lines.append("  Segmented structures:")

        for structure in structures:
            name = structure.get("name", "Unknown")

            if not structure.get("present"):
                threshold = quantification.get("presence_threshold_pixels")
                if threshold:
                    lines.append(
                        f"    - {name}: fewer than {threshold} mask pixels "
                        "were labelled with this structure, so it is not "
                        "reported. This is a segmentation threshold, NOT "
                        "evidence that the structure is absent or abnormal."
                    )
                else:
                    lines.append(
                        f"    - {name}: not identified in this frame. This "
                        "reflects the segmentation output only, NOT evidence "
                        "that the structure is absent or abnormal."
                    )
                continue

            parts = []
            area_cm2 = structure.get("area_cm2")
            if calibrated and area_cm2:
                parts.append(f"{area_cm2:.2f} cm²")

            area_percent = structure.get("area_percent")
            if area_percent is not None:
                parts.append(f"{area_percent:.1f}% of the image field")

            confidence = structure.get("mean_confidence")
            if confidence is not None:
                parts.append(
                    f"mean segmentation probability {confidence:.2f}"
                )

            lines.append(f"    - {name}: {', '.join(parts)}.")

    if not calibrated:
        lines.append(
            "  NOTE: the source image carried no pixel spacing, so areas are "
            "relative (percentage of field) and NOT absolute measurements. "
            "Do not state chamber sizes in cm or compare against reference "
            "ranges."
        )

    # A display-oriented upload left unrotated is out of distribution, which
    # is a caveat about the reliability of everything above it. The language
    # model must know that before it discusses the findings.
    if orientation.get("display_oriented_format") and not orientation.get(
        "reoriented"
    ):
        lines.append(
            "  WARNING: this image was uploaded in a display-oriented format "
            "and no rotation was applied. The model was trained on images "
            f"with the {orientation.get('training_orientation', 'apex left')}, "
            "so the segmentation above may be unreliable. Treat these "
            "structure findings as provisional and say so if asked about them."
        )
    elif orientation.get("reoriented"):
        applied = []
        if orientation.get("rotation_applied"):
            applied.append(
                f"rotated {orientation['rotation_applied']}° "
                "counter-clockwise"
            )
        if orientation.get("flip_applied"):
            applied.append("mirrored horizontally")
        lines.append(
            "  NOTE: before segmentation the clinician deliberately "
            f"{' and '.join(applied)} the image, to match the orientation "
            "the model was trained on."
        )

    lines.append(
        "  NOTE: this is anatomical segmentation only. The model does not "
        "measure ejection fraction, wall motion, valve function or "
        "haemodynamics, and it does not output a diagnosis."
    )

    return lines


# ============================================================
# UNAVAILABLE MODALITIES
# ============================================================

def _unavailable_lines(
    modalities_provided: dict[str, bool],
    echo_analyzed: bool,
) -> list[str]:
    lines: list[str] = []

    for key in ("ccta", "ecg", "clinical", "fusion"):
        status = MODALITY_STATUS.get(key, {})
        if status.get("available"):
            continue

        label = status.get("label", key.upper())
        uploaded = bool(modalities_provided.get(key))

        if uploaded:
            lines.append(
                f"  {label}: a file was uploaded, but NO trained model exists "
                "for this modality yet, so it has NOT been analysed. No "
                f"{label.lower()} findings are available."
            )
        else:
            lines.append(
                f"  {label}: not available. No trained model exists for this "
                "modality yet."
            )

    if not echo_analyzed:
        lines.append(
            "  Echocardiography: no echo image has been analysed in this "
            "case, so no imaging findings are available."
        )

    return lines


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_case_context(case: Optional[dict[str, Any]]) -> Optional[str]:
    """
    Render the case state as a text block for the prompt.

    Returns None when there is genuinely nothing to say, which makes the
    downstream prompt fall back to general-medical-knowledge mode.
    """
    if not case:
        return None

    sections: list[str] = []

    case_id = (case.get("case_id") or "").strip()
    if case_id:
        sections.append(f"Case ID: {case_id}")

    # ---- patient -------------------------------------------------
    patient = case.get("patient") or {}
    patient_lines = _patient_lines(patient)

    if patient_lines:
        sections.append(
            "PATIENT (recorded by the clinician; the name is withheld from "
            "this prompt):\n" + "\n".join(patient_lines)
        )

    # ---- clinical ------------------------------------------------
    clinical = case.get("clinical") or {}
    clinical_lines = _clinical_lines(
        clinical,
        has_patient_age=derive_age(patient.get("dateOfBirth") or "") is not None,
    )

    if clinical_lines:
        sections.append(
            "CLINICAL DATA (entered by the clinician, not model output):\n"
            + "\n".join(clinical_lines)
        )

    # ---- echo ----------------------------------------------------
    echo = case.get("echo") or {}
    echo_analyzed = bool(echo.get("analyzed"))

    if echo_analyzed:
        sections.append(
            "ECHOCARDIOGRAPHY — AI SEGMENTATION RESULT:\n"
            + "\n".join(_echo_lines(echo))
        )

    # ---- unavailable ---------------------------------------------
    modalities_provided = case.get("modalities_provided") or {}
    unavailable = _unavailable_lines(modalities_provided, echo_analyzed)

    if unavailable:
        sections.append(
            "NOT AVAILABLE FOR THIS CASE — do not infer or invent findings "
            "for any of these:\n" + "\n".join(unavailable)
        )

    # Only the boilerplate sections and nothing real to report.
    if not clinical_lines and not echo_analyzed and not patient_lines:
        return None

    return "\n\n".join(sections).strip()
