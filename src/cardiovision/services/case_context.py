"""
CardioVision AI — case context construction.

Turns the structured case state sent by the frontend into the plain-text
CASE CONTEXT block handed to MedGemma.

The single most important job here is the "NOT AVAILABLE" section. MedGemma
is told never to invent findings, but the reliable way to hold it to that is
to state explicitly which modalities have no model behind them. Without
this, a question like "what do the CT findings show?" invites a fabricated
answer.

The second most important job is that every number arrives with the thing that
qualifies it. A probability is written next to the precision it was measured
at; a segmented area is written next to whether the image had pixel spacing; an
unticked risk-factor box is written as unknown rather than as denied. The point
is not to hedge everything — it is that the language model reads this text and
nothing else, so any caveat left out here is a caveat it cannot apply.

This module serves free-form Q&A, where the question is unknown in advance and
the guard rails therefore have to be exhaustive prose. The report path uses
:func:`cardiovision.fusion.report.build_report_prompt` instead, which renders
the same case through the structured evidence layer for one fixed task. Both
read the stored case dict, so a fact added to one belongs in the other.
"""

from __future__ import annotations

from typing import Any, Optional

from cardiovision.config import (
    CCTA_PRESENCE_THRESHOLD_VOXELS,
    CCTA_TEST_METRICS,
    ECG_ARCHITECTURE,
    ECG_NORMALIZATION,
    ECG_TEST_METRICS,
    ECG_THRESHOLD,
    MODALITY_STATUS,
)
from cardiovision.services.database import derive_age


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
# ECG SECTION
# ============================================================

def _probability_text(prediction: dict[str, Any]) -> str:
    """One class as a line the model can quote without reinterpreting it."""
    name = prediction.get("name", "?")
    label = prediction.get("label") or name
    probability = prediction.get("probability")

    if isinstance(probability, (int, float)):
        shown = f"{float(probability):.2f}"
    else:
        shown = "unknown"

    verdict = "CALLED POSITIVE" if prediction.get("positive") else "not called"

    line = f"    - {label} ({name}): p = {shown} — {verdict}."

    # The operating point only means something for a positive call: it is the
    # answer to "given the model said this, how often is it right", and that
    # question is not being asked about the classes it stayed quiet on.
    operating = prediction.get("operating_point") or {}
    precision = operating.get("precision")

    if prediction.get("positive") and isinstance(precision, (int, float)):
        false_rate = 1.0 - float(precision)
        line += (
            f" On the held-out test split this class had precision "
            f"{float(precision):.2f} at this threshold, so about "
            f"{false_rate * 100:.0f}% of positive calls like it were false."
        )

    return line


def _ecg_lines(ecg: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    model = ecg.get("model") or {}
    metrics = model.get("metrics") or {}
    source = ecg.get("input") or {}
    preprocessing = ecg.get("preprocessing") or {}

    architecture = model.get("architecture") or ECG_ARCHITECTURE

    lines.append(
        f"  Model: {architecture}, a 1-D residual CNN screening a 12-lead ECG "
        "for five PTB-XL diagnostic superclasses. The five outputs are "
        "INDEPENDENT sigmoids, not a softmax, so any number of them can be "
        "positive at once and they do not sum to 1."
    )

    macro_auroc = metrics.get("macro_AUROC") or ECG_TEST_METRICS["macro_AUROC"]
    records = metrics.get("test_records") or ECG_TEST_METRICS["test_records"]
    patients = metrics.get("test_patients") or ECG_TEST_METRICS["test_patients"]

    lines.append(
        f"  Model accuracy on a patient-disjoint held-out test split: macro "
        f"AUROC {float(macro_auroc):.3f} ({records} recordings from {patients} "
        "patients). This describes the model in general, NOT the certainty of "
        "this particular reading, and the macro average hides a wide spread "
        "between classes — see each class's own precision below."
    )

    source_format = source.get("format")
    if source_format:
        detail = f"  Source recording: {str(source_format).upper()}"
        record_name = source.get("record_name")
        if record_name:
            detail += f", record {record_name}"
        sampling = source.get("sampling_frequency_hz")
        if sampling:
            detail += f", sampled at {sampling} Hz"
            resampled = source.get("resampled_to_hz")
            if resampled and resampled != sampling:
                detail += f" and resampled to {resampled} Hz for the model"
        lines.append(detail + ".")

    # ---- the five probabilities ----------------------------------
    predictions = list(ecg.get("predictions") or [])
    threshold = ecg.get("threshold") or model.get("threshold") or ECG_THRESHOLD

    if predictions:
        # Highest probability first, so the calls lead and the near-misses sit
        # directly beneath them rather than being buried in class order.
        predictions.sort(
            key=lambda item: item.get("probability") or 0.0, reverse=True
        )

        lines.append(
            f"  A class is called positive at p >= {threshold}. Every "
            "probability is listed, including the ones that were not called, "
            "so a value sitting just under the line is visible instead of "
            "being reported as a negative:"
        )
        lines.extend(_probability_text(item) for item in predictions)

        positive = [item for item in predictions if item.get("positive")]

        if not positive:
            lines.append(
                "  No superclass reached the threshold. That is a real result "
                "and it is NOT the same as a normal ECG: NORM is itself one of "
                "the five classes and it was not called either. Report this as "
                "'no superclass reached the threshold', and do not upgrade it "
                "to a normal study."
            )

        # NORM alongside an abnormal call is internally contradictory. The
        # inference layer already emits a note; saying it here as well means the
        # contradiction is in front of the model at the point it reads the
        # numbers, not several sections later.
        abnormal = [item for item in positive if item.get("name") != "NORM"]
        if any(item.get("name") == "NORM" for item in positive) and abnormal:
            names = ", ".join(item.get("name", "?") for item in abnormal)
            lines.append(
                f"  CONTRADICTION: NORM was called positive at the same time as "
                f"{names}. The five outputs are independent, so nothing stops "
                "this happening, but it means the model is genuinely uncertain. "
                "Say so rather than choosing whichever answer reads better."
            )

    # ---- weak classes -------------------------------------------
    #
    # Only for classes THIS recording called positive. A standing paragraph
    # about hypertrophy on every reading is the kind of boilerplate that stops
    # being read, and the one time it matters is the one time it looks the same
    # as all the others.
    warnings = ecg.get("weak_class_warnings") or {}

    if not warnings:
        warnings = {
            item.get("name"): item.get("caveat")
            for item in predictions
            if item.get("positive") and item.get("caveat")
        }

    for name, caveat in warnings.items():
        if name and caveat:
            lines.append(f"  WARNING — {name} was called positive: {caveat}")

    # ---- lead attribution ----------------------------------------
    if ecg.get("saliency_available") and ecg.get("lead_attribution"):
        ranked = list(ecg["lead_attribution"])[:4]
        named = ", ".join(
            f"{item.get('name', '?')} ({float(item.get('score') or 0.0):.2f})"
            for item in ranked
        )
        target = ecg.get("saliency_class") or "the leading class"
        lines.append(
            f"  Lead attribution for {target}, strongest first: {named}. These "
            "are input-gradient magnitudes — how much the output moved with "
            "each lead — scaled to the strongest lead. They indicate where the "
            "model looked, NOT where an abnormality is, and a high score is "
            "not evidence that the lead is abnormal."
        )
    elif predictions:
        lines.append(
            "  No saliency was computed for this recording, so there is no "
            "information about which leads drove the result. Do not speculate "
            "about lead involvement."
        )

    # ---- caveats -------------------------------------------------
    if source.get("lead_order_matches_training") is False:
        found = ", ".join(str(name) for name in source.get("lead_names") or [])
        lines.append(
            "  WARNING: the leads did not arrive in the order the model was "
            f"trained on (found: {found}). They were NOT reordered, so every "
            "probability above may be wrong. Treat the whole reading as "
            "unreliable and say so if asked."
        )

    if not source.get("units"):
        lines.append(
            "  NOTE: the recording carried no amplitude units, so the "
            "waveform scale is unverified. Do not quote millivolt amplitudes."
        )

    normalization = preprocessing.get("normalization") or ECG_NORMALIZATION
    lines.append(
        f"  NOTE: the model's input is normalised {normalization}, which "
        "removes absolute voltage. It therefore cannot apply the millimetre "
        "voltage criteria a human reader uses — relevant to how weak its "
        "hypertrophy performance is — and no amplitude in millivolts can be "
        "attributed to it."
    )

    lines.append(
        "  NOTE: this is a screening classifier over five broad superclasses. "
        "It does not measure heart rate, rhythm, PR/QRS/QT intervals or axis; "
        "it does not localise an infarct or separate acute from old; it does "
        "not detect atrial fibrillation, which is not one of its classes; and "
        "it does not output a diagnosis."
    )

    return lines


# ============================================================
# CCTA SECTION
# ============================================================

def _ccta_lines(ccta: dict[str, Any]) -> list[str]:
    """
    The CT segmentation result, written so it cannot be read as a CT report.

    A radiology reader seeing "coronary CT angiography" expects stenosis
    severity and a CAD-RADS category. This model produces neither, and the gap
    between what the modality name promises and what the model delivers is the
    single largest fabrication risk in the project — so the limits are stated
    before the numbers, not after them.
    """
    lines: list[str] = []

    model = ccta.get("model") or {}
    metrics = model.get("metrics") or {}
    source = ccta.get("input") or {}
    coverage = ccta.get("coverage") or {}
    findings = ccta.get("findings") or []

    architecture = model.get("architecture") or "Small3DUNet"

    lines.append(
        f"  Model: {architecture}, a 3-D U-Net performing BINARY segmentation "
        "of the contrast-filled coronary lumen. Its entire output is a mask of "
        "where lumen is."
    )
    lines.append(
        "  This model does NOT grade stenosis, does NOT measure percentage "
        "narrowing, does NOT compute a calcium score, does NOT assign a "
        "CAD-RADS category and does NOT identify or name vessels. If asked for "
        "any of those, say CardioVision does not compute it."
    )

    dice = metrics.get("test_dice") or CCTA_TEST_METRICS["dice"]
    hd95 = metrics.get("test_hd95_mm") or CCTA_TEST_METRICS["hd95_mm"]
    sensitivity = metrics.get("test_sensitivity") or CCTA_TEST_METRICS["sensitivity"]
    test_cases = metrics.get("test_cases") or CCTA_TEST_METRICS["test_cases"]

    lines.append(
        f"  Held-out performance: Dice {dice.get('mean'):.4f} "
        f"(range {dice.get('min'):.4f}-{dice.get('max'):.4f}), sensitivity "
        f"{sensitivity.get('mean'):.4f}, 95th-percentile Hausdorff distance "
        f"{hd95.get('mean'):.1f} mm."
    )
    lines.append(
        f"  WARNING — that test split is {test_cases} CASES. Three cases "
        "support no confidence interval and no claim that the model "
        "generalises. This is the WEAKEST of CardioVision's three models. The "
        "Hausdorff figure means the predicted surface can sit tens of "
        "millimetres from the truth, so the mask's shape and connectivity are "
        "unreliable even where its volume looks reasonable."
    )
    lines.append(
        "  These numbers describe the model on that cohort. They are NOT the "
        "certainty of this particular segmentation."
    )

    threshold = ccta.get("threshold")
    if threshold is not None:
        lines.append(
            f"  A voxel is called lumen at sigmoid p >= {threshold}, chosen on "
            "the validation split. Every metric above was measured at that "
            "threshold."
        )

    shape = source.get("analysed_shape")
    spacing = source.get("analysed_spacing_mm")
    if shape and spacing:
        lines.append(
            f"  Analysed on a {'x'.join(str(d) for d in shape)} grid at "
            f"{spacing[0]} mm isotropic, so one voxel is one cubic millimetre."
        )

    for finding in findings:
        name = finding.get("name") or "Coronary artery lumen"

        if not finding.get("present"):
            lines.append(
                f"  {name}: NOT identified above the reporting cutoff of "
                f"{CCTA_PRESENCE_THRESHOLD_VOXELS} voxels. Given this model's "
                f"measured sensitivity of {sensitivity.get('mean'):.2f}, an "
                "empty mask is as likely to be a miss as a true absence. Do "
                "NOT report this as normal coronary anatomy and do NOT report "
                "it as absent vessels."
            )
            continue

        volume_ml = finding.get("volume_ml")
        percent = finding.get("percent_of_analysed")
        mean_p = finding.get("mean_probability")

        detail = f"  {name}: segmented"
        if volume_ml is not None:
            detail += f", {volume_ml} mL"
        if percent is not None:
            detail += f" ({percent}% of the analysed volume)"
        if mean_p is not None:
            detail += f", mean model output {mean_p}"
        lines.append(detail + ".")

        components = finding.get("components")
        largest = finding.get("largest_component_fraction")
        if components is not None:
            fragmentation = (
                f"  The mask is {components} disconnected component"
                f"{'s' if components != 1 else ''}"
            )
            if largest is not None:
                fragmentation += f", the largest holding {largest:.0%} of the voxels"
            lines.append(
                fragmentation
                + ". A healthy coronary tree is a small number of connected "
                "vessels, so a high component count means the segmentation is "
                "fragmented and should be read as unreliable rather than as "
                "anatomy."
            )

    if coverage and coverage.get("complete") is False:
        lines.append(
            f"  PARTIAL COVERAGE — only {coverage.get('analysed_percent')}% of "
            "the volume was analysed within the compute budget. Everything "
            "outside that region was NOT examined. Do not describe it, and do "
            "not treat the absence of a mask there as a finding."
        )

    explainability = ccta.get("explainability") or {}
    if explainability.get("available"):
        lines.append(
            "  A 3-D Grad-CAM map is available for ONE 96x96x96 patch of the "
            "volume. It shows where activation supported the model's own "
            "output, NOT where disease is, and attention outside that patch "
            "was never computed."
        )
    else:
        lines.append(
            "  No Grad-CAM was computed for this volume, so there is no "
            "information about what the model responded to. Do not speculate "
            "about which region drove the segmentation."
        )

    return lines


# ============================================================
# UNAVAILABLE MODALITIES
# ============================================================

def _unavailable_lines(
    modalities_provided: dict[str, bool],
    echo_analyzed: bool,
    ecg_analyzed: bool = False,
    ccta_analyzed: bool = False,
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

    # Distinct from the loop above, which covers modalities with no model at
    # all. There IS an ECG model; this case simply has no ECG through it, and
    # conflating the two would tell the model the pipeline does not exist.
    if not ecg_analyzed and MODALITY_STATUS.get("ecg", {}).get("available"):
        lines.append(
            "  Electrocardiography: an ECG model is available, but no ECG has "
            "been analysed in this case, so there are no ECG findings. Do not "
            "infer a rhythm or an ECG pattern from the other data."
        )

    # Same distinction for CCTA, which acquired a trained model and so must no
    # longer be described as a modality without one.
    if not ccta_analyzed and MODALITY_STATUS.get("ccta", {}).get("available"):
        lines.append(
            "  Coronary CT angiography: a CT lumen segmentation model is "
            "available, but no CT volume has been analysed in this case, so "
            "there are no CT findings. Do not infer coronary anatomy, stenosis "
            "or calcification from the other data."
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

    # ---- ccta ----------------------------------------------------
    #
    # First, matching the target architecture's order: CCTA -> echo -> ECG.
    ccta = case.get("ccta") or {}
    ccta_analyzed = bool(ccta.get("analyzed")) and bool(ccta.get("findings"))

    if ccta_analyzed:
        sections.append(
            "CORONARY CT ANGIOGRAPHY — AI SEGMENTATION RESULT:\n"
            + "\n".join(_ccta_lines(ccta))
        )

    # ---- echo ----------------------------------------------------
    echo = case.get("echo") or {}
    echo_analyzed = bool(echo.get("analyzed"))

    if echo_analyzed:
        sections.append(
            "ECHOCARDIOGRAPHY — AI SEGMENTATION RESULT:\n"
            + "\n".join(_echo_lines(echo))
        )

    # ---- ecg -----------------------------------------------------
    #
    # Presence of predictions is the test, not an `analyzed` flag: the ECG
    # endpoint has no state in which it returns probabilities it did not
    # compute, and a stored case carries exactly what the endpoint returned.
    ecg = case.get("ecg") or {}
    ecg_analyzed = bool(ecg.get("predictions")) and ecg.get("analyzed") is not False

    if ecg_analyzed:
        sections.append(
            "ELECTROCARDIOGRAPHY — AI SCREENING RESULT:\n"
            + "\n".join(_ecg_lines(ecg))
        )

    # ---- unavailable ---------------------------------------------
    modalities_provided = case.get("modalities_provided") or {}
    unavailable = _unavailable_lines(
        modalities_provided, echo_analyzed, ecg_analyzed, ccta_analyzed
    )

    if unavailable:
        sections.append(
            "NOT AVAILABLE FOR THIS CASE — do not infer or invent findings "
            "for any of these:\n" + "\n".join(unavailable)
        )

    # There is no fusion model, so a case carrying more than one modality must
    # not read as a combined workup. Said once, at the end, where a reader of
    # the prompt has just seen the sections side by side.
    analysed_count = sum((ccta_analyzed, echo_analyzed, ecg_analyzed))
    if analysed_count > 1:
        sections.append(
            "HOW TO READ MORE THAN ONE MODALITY:\n"
            "  The sections above come from independently trained models with "
            "no multimodal model between them, and they were trained on three "
            "unrelated public datasets in which no patient appears twice. That "
            "these inputs belong to one patient is asserted by the operator, "
            "not established by any model.\n"
            "  Report each modality in its own terms. Do NOT combine them into "
            "a single risk score, likelihood or severity grade, and do NOT say "
            "that one modality confirms, corroborates or is consistent with "
            "another. You may state that two findings were observed together."
        )

    # Only the boilerplate sections and nothing real to report.
    if not clinical_lines and not echo_analyzed and not ecg_analyzed \
            and not ccta_analyzed and not patient_lines:
        return None

    return "\n\n".join(sections).strip()
