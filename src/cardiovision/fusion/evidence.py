"""
CardioVision AI — deterministic evidence aggregation.

Reads a case exactly as the frontend and the case store represent it, and
produces a ``CaseEvidence`` object: what each model actually reported, what was
never available, and what should be trusted less than it looks.

Rules this module holds to
--------------------------
1. Nothing is computed that a model did not output. No combined score, no
   weighted average, no derived risk category.
2. Every cross-modal item is co-occurrence, tagged ``inference: "none"``.
3. "Not provided", "no model", "provided but not analysed" and "analysed and
   found nothing" are four different states and stay four different states.
4. A missing clinical field stays missing. It is never defaulted to a negative.

The order of the modality dict follows the target architecture — CCTA, then
echo, then ECG — so a report reads in the same order every time regardless of
which modalities a case happens to have.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from cardiovision.config import (
    CCTA_WEAK_NOTES,
    ECG_CLASS_NAMES,
    ECG_TEST_METRICS,
    ECG_THRESHOLD,
    ECHO_TEST_METRICS,
    MODALITY_STATUS,
)
from cardiovision.fusion.schema import (
    STATUS_ANALYSED,
    STATUS_NO_MODEL,
    STATUS_NOT_PROVIDED,
    STATUS_PROVIDED_NOT_ANALYSED,
    CaseEvidence,
    ClinicalContext,
    CrossModalObservation,
    EVIDENCE_MODALITIES,
    ModalityEvidence,
    Uncertainty,
)
from cardiovision.services.database import derive_age

__all__ = ["build_case_evidence", "normalise_clinical"]


# ============================================================
# CLINICAL NORMALISATION
# ============================================================

# The risk-factor checkboxes the form actually offers. A ticked box is a
# positive history. An unticked box is NOT a denial — the form ships them
# defaulting to false, so false means untouched.
_RISK_FACTORS = (
    ("diabetes", "Diabetes"),
    ("hypertension", "Hypertension"),
    ("smoking", "Smoking"),
)

# Fields the report schema has but this application collects nowhere. Named
# explicitly so an empty list is never read as "none of these".
_NOT_COLLECTED = (
    "medications",
    "family_history",
    "prior_cardiac_events",
    "lipid_profile",
)


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any) -> dict[str, Any]:
    """
    A dict, or an empty one for anything that is not a dict.

    Used instead of ``value or {}`` throughout the builders. The difference only
    shows on bad input, which is exactly the case that matters here: ``or {}``
    passes a list straight through and the next ``.get`` raises, whereas this
    degrades to "that block was not supplied" — which the four-state vocabulary
    already knows how to report.
    """
    return value if isinstance(value, dict) else {}


def _items(value: Any, key: str = "name") -> list[dict[str, Any]]:
    """
    A modality's finding list, as a list of dicts, whatever arrived.

    This layer reads whatever the caller hands it. ``/api/report`` and
    ``/api/evidence`` accept a case body from the browser, and a case can also
    have been saved by an older release whose payload shape differed. Both mean
    the finding lists are untrusted input, so the aggregator must not raise on
    them: an ``AttributeError`` deep in a builder would turn one malformed field
    into a 500 that loses every other modality's result too.

    A list of dicts passes through with non-dicts dropped. A mapping is accepted
    as ``{name: probability}`` or ``{name: {...}}`` and rebuilt into the list
    shape, which is how a class-keyed prediction dict reads. Anything else
    becomes an empty list, and the modality then reports itself un-analysed
    rather than reporting invented findings.
    """
    if isinstance(value, dict):
        rebuilt: list[dict[str, Any]] = []
        for name, entry in value.items():
            if isinstance(entry, dict):
                rebuilt.append({key: name, **entry})
            elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
                rebuilt.append({key: name, "probability": float(entry)})
        return rebuilt

    if isinstance(value, (list, tuple)):
        return [entry for entry in value if isinstance(entry, dict)]

    return []


def _number(value: Any) -> Optional[float]:
    """A float, or None for anything that is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _split_symptoms(value: Any) -> list[str]:
    """
    Free-text symptoms into a list, splitting only on unambiguous separators.

    Commas, semicolons and newlines. Not on "and" — "chest pain and shortness
    of breath on exertion" would fracture into two half-symptoms, one of which
    changes meaning.
    """
    text = _clean_text(value)
    if not text:
        return []

    parts = [text]
    for separator in (";", "\n", ","):
        expanded: list[str] = []
        for part in parts:
            expanded.extend(part.split(separator))
        parts = expanded

    return [part.strip() for part in parts if part.strip()]


def normalise_clinical(
    patient: Optional[dict[str, Any]],
    clinical: Optional[dict[str, Any]],
) -> ClinicalContext:
    """
    Structured clinical context with missing information left missing.

    Age prefers the date of birth over the typed age: a typed age goes stale
    the moment a birthday passes, and emitting both would leave a downstream
    reader to pick between two numbers that disagree.
    """
    patient = _mapping(patient)
    clinical = _mapping(clinical)

    recorded: list[str] = []
    unknown: list[str] = []

    age: Optional[int] = None
    age_source: Optional[str] = None

    derived = derive_age(patient.get("dateOfBirth") or "")
    if derived is not None:
        age = int(derived)
        age_source = "derived from date of birth"
        recorded.append("age")
    else:
        try:
            typed = clinical.get("age")
            if typed not in (None, "", False):
                age = int(float(typed))
                age_source = "entered by the clinician"
                recorded.append("age")
        except (TypeError, ValueError):
            age = None

    if age is None:
        unknown.append("age")

    sex = _clean_text(patient.get("sex")) or _clean_text(clinical.get("sex"))
    if sex:
        recorded.append("sex")
    else:
        unknown.append("sex")

    blood_pressure = _clean_text(clinical.get("bloodPressure"))
    if blood_pressure:
        recorded.append("blood_pressure")
    else:
        unknown.append("blood_pressure")

    heart_rate: Optional[int] = None
    try:
        raw_rate = clinical.get("heartRate")
        if raw_rate not in (None, "", False):
            heart_rate = int(float(raw_rate))
    except (TypeError, ValueError):
        heart_rate = None

    if heart_rate is not None:
        recorded.append("heart_rate")
    else:
        unknown.append("heart_rate")

    symptoms = _split_symptoms(clinical.get("symptoms"))
    if symptoms:
        recorded.append("symptoms")
    else:
        unknown.append("symptoms")

    history = [
        label for key, label in _RISK_FACTORS if clinical.get(key) is True
    ]
    untouched = [
        label for key, label in _RISK_FACTORS if clinical.get(key) is not True
    ]

    if history:
        recorded.append("history")
    if untouched:
        # Deliberately in `unknown` rather than absent from the report. An
        # unticked hypertension box previously read downstream as "no
        # hypertension", which is a clinical claim nobody made.
        unknown.extend(f"history:{label}" for label in untouched)

    notes = _clean_text(patient.get("notes"))
    if notes:
        recorded.append("notes")

    study_date = _clean_text(patient.get("studyDate"))
    if study_date:
        recorded.append("study_date")

    return ClinicalContext(
        age=age,
        age_source=age_source,
        sex=sex,
        blood_pressure=blood_pressure,
        heart_rate=heart_rate,
        symptoms=symptoms,
        history=history,
        medications=[],
        notes=notes,
        study_date=study_date,
        recorded=recorded,
        unknown=unknown,
        not_collected=list(_NOT_COLLECTED),
    )


# ============================================================
# MODALITY EVIDENCE
# ============================================================


def _status_for(
    key: str,
    analysed: bool,
    provided: bool,
) -> str:
    if not MODALITY_STATUS.get(key, {}).get("available"):
        return STATUS_NO_MODEL
    if analysed:
        return STATUS_ANALYSED
    if provided:
        return STATUS_PROVIDED_NOT_ANALYSED
    return STATUS_NOT_PROVIDED


def _empty_modality(key: str, provided: bool) -> ModalityEvidence:
    status = MODALITY_STATUS.get(key, {})

    return ModalityEvidence(
        key=key,
        label=status.get("label", key.upper()),
        status=_status_for(key, analysed=False, provided=provided),
        model_available=bool(status.get("available")),
        input_provided=provided,
        analysed=False,
        task=status.get("task"),
        notes=[status.get("note")] if status.get("note") else [],
    )


def _ccta_evidence(
    data: Optional[dict[str, Any]],
    provided: bool,
) -> ModalityEvidence:
    data = _mapping(data)
    measured = _items(data.get("findings"))
    analysed = bool(data.get("analyzed")) and bool(measured)

    if not analysed:
        return _empty_modality("ccta", provided or bool(data.get("filename")))

    status = MODALITY_STATUS["ccta"]
    model = _mapping(data.get("model"))
    metrics = _mapping(model.get("metrics"))
    coverage = _mapping(data.get("coverage"))
    explainability = _mapping(data.get("explainability"))
    source = _mapping(data.get("input"))

    findings: list[dict[str, Any]] = []
    for item in measured:
        findings.append({
            "name": item.get("name"),
            "observed": bool(item.get("present")),
            "measurement": {
                "voxels": item.get("voxels"),
                "volume_ml": item.get("volume_ml"),
                "percent_of_analysed": item.get("percent_of_analysed"),
                "connected_components": item.get("components"),
                "largest_component_fraction": item.get(
                    "largest_component_fraction"
                ),
            },
            "confidence": {
                "mean_probability": item.get("mean_probability"),
                "max_probability": item.get("max_probability"),
                "scale": (
                    "Per-voxel sigmoid output averaged over the predicted mask. "
                    "It is the model's own output value, not a calibrated "
                    "probability that the finding is correct."
                ),
            },
            "semantics": (
                "Contrast-filled lumen only. This finding carries no stenosis "
                "grade, no calcium score, no CAD-RADS category and no vessel "
                "identity."
            ),
        })

    limitations = list(CCTA_WEAK_NOTES)

    complete = bool(coverage.get("complete", True))
    if not complete:
        limitations.append(
            f"Only {coverage.get('analysed_percent', '?')}% of the volume was "
            "analysed within the request's window budget. The rest was not "
            "examined and is reported as unanalysed, not as normal."
        )

    return ModalityEvidence(
        key="ccta",
        label=status["label"],
        status=STATUS_ANALYSED,
        model_available=True,
        input_provided=True,
        analysed=True,
        task=status["task"],
        model={
            "name": model.get("architecture") or status["model"],
            "task": status["task"],
            "parameters": model.get("parameters"),
            "threshold": data.get("threshold"),
            "dataset": metrics.get("dataset"),
            "test_cases": metrics.get("test_cases"),
            "test_dice": _mapping(metrics.get("test_dice")).get("mean"),
            "test_hd95_mm": _mapping(metrics.get("test_hd95_mm")).get("mean"),
            "metric_scope": metrics.get("scope"),
            "metric_caveat": metrics.get("caveat"),
        },
        findings=findings,
        measurements={
            "analysed_shape": source.get("analysed_shape"),
            "analysed_spacing_mm": source.get("analysed_spacing_mm"),
            "source_format": source.get("format"),
            "voxel_volume_mm3": 1.0,
        },
        confidence={
            "type": "per-voxel sigmoid",
            "threshold": data.get("threshold"),
            "threshold_note": (
                "Chosen on the validation split. Every published metric for "
                "this model is measured at this threshold."
            ),
        },
        explainability=(
            {
                "available": bool(explainability.get("available")),
                "method": explainability.get("method"),
                "target_layer": explainability.get("target_layer"),
                "scope": explainability.get("scope"),
                "note": (
                    "Grad-CAM shows where the network's activation supported "
                    "its own output. It does not localise disease."
                ),
            }
            if explainability else {}
        ),
        coverage={
            "complete": complete,
            "analysed_fraction": coverage.get("coverage"),
            "analysed_percent": coverage.get("analysed_percent"),
            "windows_run": coverage.get("windows_run"),
            "windows_total": coverage.get("windows_total"),
        },
        limitations=limitations,
        notes=list(data.get("notes") or []),
    )


def _echo_evidence(
    data: Optional[dict[str, Any]],
    provided: bool,
) -> ModalityEvidence:
    data = _mapping(data)
    segmented = _items(data.get("structures"))
    analysed = bool(data.get("analyzed")) and bool(segmented)

    if not analysed:
        return _empty_modality("echo", provided or bool(data.get("filename")))

    status = MODALITY_STATUS["echo"]
    model = _mapping(data.get("model"))
    metrics = _mapping(model.get("metrics"))
    source = _mapping(data.get("input"))
    orientation = _mapping(data.get("orientation"))
    quantification = _mapping(data.get("quantification"))

    calibrated = bool(source.get("has_spatial_calibration"))
    threshold_px = quantification.get("presence_threshold_pixels")

    findings: list[dict[str, Any]] = []
    for structure in segmented:
        present = bool(structure.get("present"))

        measurement: dict[str, Any] = {
            "pixels": structure.get("pixels"),
            "percent_of_field": structure.get("area_percent"),
        }
        if calibrated and structure.get("area_cm2"):
            measurement["area_cm2"] = structure.get("area_cm2")

        entry: dict[str, Any] = {
            "name": structure.get("name"),
            "observed": present,
            "measurement": measurement,
            "confidence": {
                "mean_probability": structure.get("mean_confidence"),
                "scale": (
                    "Mean softmax probability over the pixels assigned to this "
                    "structure. Not a calibrated probability of correctness."
                ),
            },
            "semantics": (
                "Anatomical outline only. No ejection fraction, wall motion, "
                "valve function or haemodynamic measurement is produced."
            ),
        }

        if not present:
            entry["absence_meaning"] = (
                f"Fewer than {threshold_px} mask pixels carried this label, so "
                "it is not reported. This is a segmentation size cutoff, NOT "
                "evidence that the structure is absent or abnormal."
                if threshold_px else
                "Not identified in this frame. This reflects the segmentation "
                "output only, NOT evidence of absence or abnormality."
            )

        findings.append(entry)

    limitations = [
        "Anatomical segmentation only: no ejection fraction, no wall motion, "
        "no valve assessment, no diagnosis.",
        "One frame of one view. Nothing is measured across the cardiac cycle.",
    ]

    if not calibrated:
        limitations.append(
            "The source image carried no pixel spacing, so areas are relative "
            "(percentage of field) and not absolute. Chamber sizes in "
            "centimetres cannot be stated and reference ranges do not apply."
        )

    if orientation.get("display_oriented_format") and not orientation.get(
        "reoriented"
    ):
        limitations.append(
            "This image was uploaded in a display-oriented format and was not "
            "rotated, while the model was trained on images with the "
            f"{orientation.get('training_orientation', 'apex left')}. The "
            "segmentation is out of distribution and provisional."
        )

    return ModalityEvidence(
        key="echo",
        label=status["label"],
        status=STATUS_ANALYSED,
        model_available=True,
        input_provided=True,
        analysed=True,
        task=status["task"],
        model={
            "name": model.get("architecture") or status["model"],
            "encoder": model.get("encoder"),
            "task": status["task"],
            "parameters": model.get("parameters"),
            "dataset": metrics.get("dataset") or ECHO_TEST_METRICS["dataset"],
            "test_dice": metrics.get("test_dice") or ECHO_TEST_METRICS["test_dice"],
            "test_patients": (
                metrics.get("test_patients") or ECHO_TEST_METRICS["test_patients"]
            ),
            "metric_scope": metrics.get("scope", "dataset-level, held-out test split"),
        },
        findings=findings,
        measurements={
            "source_format": source.get("format"),
            "frame_index": source.get("frame_index"),
            "frame_count": source.get("frame_count"),
            "spatially_calibrated": calibrated,
            "presence_threshold_pixels": threshold_px,
        },
        confidence={
            "type": "per-pixel softmax over 4 classes",
            "presence_threshold_pixels": threshold_px,
        },
        explainability={
            "available": bool(data.get("saliency_available")),
            "method": "input-gradient saliency",
            "target": data.get("saliency_class_name") or "LV cavity",
            "note": (
                "Saliency shows which input pixels the output responded to. It "
                "does not localise disease."
            ),
        },
        limitations=limitations,
        notes=list(data.get("notes") or []),
    )


def _ecg_evidence(
    data: Optional[dict[str, Any]],
    provided: bool,
) -> ModalityEvidence:
    data = _mapping(data)
    predictions = _items(data.get("predictions"))
    analysed = bool(predictions) and data.get("analyzed") is not False

    if not analysed:
        return _empty_modality("ecg", provided or bool(data.get("filename")))

    status = MODALITY_STATUS["ecg"]
    model = _mapping(data.get("model"))
    metrics = _mapping(model.get("metrics"))
    source = _mapping(data.get("input"))
    threshold = data.get("threshold") or model.get("threshold") or ECG_THRESHOLD

    # Highest probability first. Sorted on a coerced number rather than on the
    # raw field so a null or a string probability sinks to the bottom instead of
    # raising a comparison error mid-sort.
    predictions.sort(
        key=lambda item: _number(item.get("probability")) or 0.0,
        reverse=True,
    )

    findings: list[dict[str, Any]] = []
    for item in predictions:
        operating = _mapping(item.get("operating_point"))
        precision = _number(operating.get("precision"))

        entry: dict[str, Any] = {
            "name": item.get("name"),
            "label": item.get("label"),
            "observed": bool(item.get("positive")),
            "confidence": {
                "probability": item.get("probability"),
                "threshold": threshold,
                "scale": (
                    "Independent sigmoid. The five outputs do not sum to 1 and "
                    "any number of them can be positive at once."
                ),
            },
        }

        if item.get("positive") and precision is not None:
            entry["confidence"]["measured_precision"] = precision
            entry["confidence"]["false_positive_rate_at_threshold"] = round(
                1.0 - precision, 4
            )

        if item.get("caveat"):
            entry["caveat"] = item.get("caveat")

        findings.append(entry)

    positive = [item for item in predictions if item.get("positive")]

    limitations = [
        "Screening over five broad superclasses only. No heart rate, rhythm, "
        "PR/QRS/QT interval or axis measurement; no infarct localisation; no "
        "acute-versus-old distinction; atrial fibrillation is not one of the "
        "classes.",
        "The input is normalised per lead, which removes absolute voltage, so "
        "the millimetre voltage criteria a human reader uses cannot be applied "
        "and no millivolt amplitude can be attributed to the model.",
    ]

    for name, caveat in _mapping(data.get("weak_class_warnings")).items():
        if name and caveat:
            limitations.append(f"{name} called positive: {caveat}")

    if source.get("lead_order_matches_training") is False:
        limitations.append(
            "The leads did not arrive in the training order and were not "
            "reordered, so every probability above may be wrong."
        )

    return ModalityEvidence(
        key="ecg",
        label=status["label"],
        status=STATUS_ANALYSED,
        model_available=True,
        input_provided=True,
        analysed=True,
        task=status["task"],
        model={
            "name": model.get("architecture") or status["model"],
            "task": status["task"],
            "parameters": model.get("parameters"),
            "classes": list(ECG_CLASS_NAMES),
            "threshold": threshold,
            "dataset": metrics.get("dataset") or ECG_TEST_METRICS["dataset"],
            "macro_AUROC": (
                metrics.get("macro_AUROC") or ECG_TEST_METRICS["macro_AUROC"]
            ),
            "test_records": (
                metrics.get("test_records") or ECG_TEST_METRICS["test_records"]
            ),
            "test_patients": (
                metrics.get("test_patients") or ECG_TEST_METRICS["test_patients"]
            ),
            "metric_scope": metrics.get("scope", "dataset-level, held-out test split"),
        },
        findings=findings,
        measurements={
            "source_format": source.get("format"),
            "sampling_frequency_hz": source.get("sampling_frequency_hz"),
            "resampled_to_hz": source.get("resampled_to_hz"),
            "positive_classes": [item.get("name") for item in positive],
            "positive_count": len(positive),
        },
        confidence={
            "type": "five independent sigmoids",
            "threshold": threshold,
            "threshold_note": (
                "Every published F1, precision and recall for this model is "
                "measured at this operating point."
            ),
        },
        explainability={
            "available": bool(data.get("saliency_available")),
            "method": "input-gradient saliency, per lead",
            "target": data.get("saliency_class"),
            "leads": list(data.get("lead_attribution") or [])[:4],
            "note": (
                "Lead attribution shows where the model looked, not where an "
                "abnormality is. A high score is not evidence that the lead is "
                "abnormal."
            ),
        },
        limitations=limitations,
        notes=list(data.get("notes") or []),
    )


_BUILDERS = {
    "ccta": _ccta_evidence,
    "echo": _echo_evidence,
    "ecg": _ecg_evidence,
}


# ============================================================
# CROSS-MODAL OBSERVATIONS
# ============================================================


def _cross_modal(
    modalities: dict[str, ModalityEvidence],
    clinical: ClinicalContext,
) -> list[CrossModalObservation]:
    """
    Co-occurrence only. Nothing here is an inference.

    Three kinds of item are honest to emit: how much of the case was actually
    examined, which observed findings happen to sit alongside which others, and
    where a single model contradicted itself. Anything beyond that — a combined
    likelihood, a corroboration claim, a "consistent with" — would require a
    model trained on paired data, and no such model exists here.
    """
    observations: list[CrossModalObservation] = []

    analysed = [key for key in EVIDENCE_MODALITIES if modalities[key].analysed]
    not_analysed = [
        key for key in EVIDENCE_MODALITIES if not modalities[key].analysed
    ]

    observations.append(
        CrossModalObservation(
            kind="coverage",
            statement=(
                f"{len(analysed)} of {len(EVIDENCE_MODALITIES)} modality models "
                f"ran on this case"
                + (f": {', '.join(analysed)}." if analysed else ".")
                + (
                    f" Not analysed: {', '.join(not_analysed)}."
                    if not_analysed else ""
                )
            ),
            modalities=list(EVIDENCE_MODALITIES),
            basis="modality status recorded on the case",
        )
    )

    if len(analysed) >= 2:
        observations.append(
            CrossModalObservation(
                kind="pairing_provenance",
                statement=(
                    "The models that ran were trained on three unrelated public "
                    "datasets and no patient appears in more than one of them. "
                    "That these inputs belong to the same patient is asserted by "
                    "the operator who uploaded them; it is not established by "
                    "any model, and no model in this project has ever been "
                    "trained on paired multimodal data."
                ),
                modalities=list(analysed),
                basis="training provenance of the three checkpoints",
            )
        )

    # ---- observed findings side by side --------------------------

    positive_ecg = [
        finding["name"]
        for finding in modalities["ecg"].findings
        if finding.get("observed") and finding.get("name") != "NORM"
    ]
    lumen_present = any(
        finding.get("observed") for finding in modalities["ccta"].findings
    )
    echo_structures = [
        finding["name"]
        for finding in modalities["echo"].findings
        if finding.get("observed")
    ]

    if positive_ecg and lumen_present:
        observations.append(
            CrossModalObservation(
                kind="co_occurrence",
                statement=(
                    f"The ECG model called {', '.join(positive_ecg)} positive, "
                    "and the CCTA model produced a lumen mask on the CT volume. "
                    "These are two separate observations listed together. The "
                    "CCTA model outputs only where contrast-filled lumen is, so "
                    "it says nothing about stenosis and cannot corroborate or "
                    "contradict an ECG finding."
                ),
                modalities=["ecg", "ccta"],
                basis="both models' own outputs",
            )
        )

    if positive_ecg and echo_structures:
        observations.append(
            CrossModalObservation(
                kind="co_occurrence",
                statement=(
                    f"The ECG model called {', '.join(positive_ecg)} positive, "
                    f"and the echo model outlined {', '.join(echo_structures)}. "
                    "The echo model produces anatomical outlines only, with no "
                    "function or wall-motion measurement, so it can neither "
                    "support nor refute an ECG classification."
                ),
                modalities=["ecg", "echo"],
                basis="both models' own outputs",
            )
        )

    if clinical.history and positive_ecg:
        observations.append(
            CrossModalObservation(
                kind="co_occurrence",
                statement=(
                    f"The clinician recorded {', '.join(clinical.history)}, and "
                    f"the ECG model called {', '.join(positive_ecg)} positive. "
                    "One is history entered by a person, the other is a model "
                    "output. They are recorded side by side and are not "
                    "combined into a risk estimate — there is no trained "
                    "clinical-risk model in this project."
                ),
                modalities=["clinical", "ecg"],
                basis="clinician entry alongside model output",
            )
        )

    # ---- within-modality contradiction ---------------------------

    ecg_positive_names = {
        finding["name"]
        for finding in modalities["ecg"].findings
        if finding.get("observed")
    }
    if "NORM" in ecg_positive_names and len(ecg_positive_names) > 1:
        others = sorted(ecg_positive_names - {"NORM"})
        observations.append(
            CrossModalObservation(
                kind="contradiction",
                statement=(
                    "The ECG model called NORM positive at the same time as "
                    f"{', '.join(others)}. The five outputs are independent "
                    "sigmoids so nothing prevents this, but it means the model "
                    "is genuinely uncertain on this recording."
                ),
                modalities=["ecg"],
                basis="the model's own output",
            )
        )

    if modalities["ecg"].analysed and not ecg_positive_names:
        observations.append(
            CrossModalObservation(
                kind="negative_result",
                statement=(
                    "No ECG superclass reached the threshold. This is not the "
                    "same as a normal ECG: NORM is itself one of the five "
                    "classes and it was not called either."
                ),
                modalities=["ecg"],
                basis="the model's own output",
            )
        )

    if modalities["ccta"].analysed and not lumen_present:
        observations.append(
            CrossModalObservation(
                kind="negative_result",
                statement=(
                    "The CCTA model produced no lumen mask above the reporting "
                    "size cutoff on the region it analysed. Given this model's "
                    "measured sensitivity of 0.62, an empty mask is as likely to "
                    "be a miss as an absence."
                ),
                modalities=["ccta"],
                basis="the model's own output and its measured sensitivity",
            )
        )

    return observations


# ============================================================
# UNCERTAINTY
# ============================================================


def _uncertainties(
    modalities: dict[str, ModalityEvidence],
    clinical: ClinicalContext,
) -> list[Uncertainty]:
    items: list[Uncertainty] = []

    for key in EVIDENCE_MODALITIES:
        evidence = modalities[key]

        if not evidence.analysed:
            continue

        for limitation in evidence.limitations:
            items.append(
                Uncertainty(
                    scope=key,
                    kind="model_limitation",
                    detail=limitation,
                    severity=(
                        "warning"
                        if limitation.startswith((
                            "This image was uploaded",
                            "The leads did not arrive",
                            "Only ",
                        )) or "called positive:" in limitation
                        else "note"
                    ),
                )
            )

        if evidence.coverage and evidence.coverage.get("complete") is False:
            items.append(
                Uncertainty(
                    scope=key,
                    kind="coverage",
                    detail=(
                        f"{evidence.coverage.get('windows_run')} of "
                        f"{evidence.coverage.get('windows_total')} sliding "
                        "windows were run. Regions outside the analysed area "
                        "were not examined."
                    ),
                    severity="warning",
                )
            )

        if evidence.explainability and evidence.explainability.get(
            "available"
        ) is False:
            items.append(
                Uncertainty(
                    scope=key,
                    kind="input_quality",
                    detail=(
                        "No explainability output was produced for this input, "
                        "so there is no information about what the model "
                        "responded to."
                    ),
                )
            )

    for key in EVIDENCE_MODALITIES:
        evidence = modalities[key]
        if evidence.analysed:
            continue

        if evidence.status == STATUS_NO_MODEL:
            detail = (
                f"{evidence.label}: no trained model exists for this modality "
                "in this project, so no findings can be produced."
            )
        elif evidence.status == STATUS_PROVIDED_NOT_ANALYSED:
            detail = (
                f"{evidence.label}: an input was supplied but has not been "
                "analysed, so no findings are available from it."
            )
        else:
            detail = (
                f"{evidence.label}: no input was supplied, so nothing was "
                "analysed. Do not infer findings for this modality from the "
                "others."
            )

        items.append(
            Uncertainty(scope=key, kind="coverage", detail=detail)
        )

    # ---- clinical -----------------------------------------------

    if clinical.is_empty:
        items.append(
            Uncertainty(
                scope="clinical",
                kind="input_quality",
                detail=(
                    "No clinical information was recorded for this case. Age, "
                    "sex, symptoms and history are all unknown."
                ),
            )
        )
    else:
        history_unknown = [
            name.split(":", 1)[1]
            for name in clinical.unknown
            if name.startswith("history:")
        ]
        if history_unknown:
            items.append(
                Uncertainty(
                    scope="clinical",
                    kind="input_quality",
                    detail=(
                        f"Not recorded either way: {', '.join(history_unknown)}. "
                        "These are UNKNOWN, not absent — the form was left "
                        "blank. Do not describe the patient as having denied "
                        "them."
                    ),
                )
            )

        plain_unknown = [
            name for name in clinical.unknown if not name.startswith("history:")
        ]
        if plain_unknown:
            items.append(
                Uncertainty(
                    scope="clinical",
                    kind="input_quality",
                    detail=(
                        f"Not recorded: {', '.join(plain_unknown)}. Treat as "
                        "unknown rather than normal."
                    ),
                )
            )

    items.append(
        Uncertainty(
            scope="case",
            kind="pairing",
            detail=(
                "No trained multimodal model exists in this project. The "
                "findings above are collected side by side by a deterministic "
                "software layer; they are not weighted, reconciled or combined "
                "into a joint risk estimate."
            ),
        )
    )

    return items


# ============================================================
# PUBLIC BUILDER
# ============================================================


def build_case_evidence(case: Optional[dict[str, Any]]) -> CaseEvidence:
    """
    Aggregate one case into structured evidence.

    Safe on an empty case: the result is a fully-populated object in which
    every modality is missing and every clinical field is unknown. That is a
    valid, meaningful state — "nothing has been analysed yet" — and callers get
    the same shape whether a case has three modalities or none.
    """
    case = _mapping(case)

    provided_flags = _mapping(case.get("modalities_provided"))

    modalities: dict[str, ModalityEvidence] = {}
    for key in EVIDENCE_MODALITIES:
        modalities[key] = _BUILDERS[key](
            case.get(key), bool(provided_flags.get(key))
        )

    clinical = normalise_clinical(case.get("patient"), case.get("clinical"))

    available = [key for key in EVIDENCE_MODALITIES if modalities[key].analysed]
    missing = [key for key in EVIDENCE_MODALITIES if not modalities[key].analysed]

    # Clinical data is not a modality result, but its absence belongs in the
    # missing list because a reader scanning that list is asking "what is not
    # here", and clinical context being absent answers that question.
    if clinical.is_empty:
        missing.append("clinical")

    model_versions: dict[str, Any] = {}
    for key in EVIDENCE_MODALITIES:
        evidence = modalities[key]
        if evidence.analysed and evidence.model:
            model_versions[key] = {
                "model": evidence.model.get("name"),
                "task": evidence.task,
                "dataset": evidence.model.get("dataset"),
            }

    model_versions["fusion"] = {
        "model": None,
        "task": "deterministic software evidence aggregation",
        "note": "No learned fusion model exists in this project.",
    }

    return CaseEvidence(
        case_id=_clean_text(case.get("case_id")),
        modalities=modalities,
        clinical=clinical,
        available_modalities=available,
        missing_modalities=missing,
        cross_modal_evidence=_cross_modal(modalities, clinical),
        uncertainties=_uncertainties(modalities, clinical),
        model_versions=model_versions,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
